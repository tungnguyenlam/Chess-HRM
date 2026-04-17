"""
Batched MCTS implementation for HRMChess.

Includes:
- PUCT selection with virtual loss for batching
- Batched leaf evaluation (GPU efficiency)
- Carry state propagation (HRM specific)
- Dirichlet noise at root
- Temperature scheduling

Implements: PLAN.md step 4.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chessgame.encoding.board_encoder import encode_board
from chessgame.encoding.move_encoder import encode_move, legal_mask, NUM_MOVES
from chessgame.model.hrm_chess import HRMChess


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""

    board: chess.Board
    parent: Optional[MCTSNode] = None
    move: Optional[chess.Move] = None

    # Policy and Value from model
    P: torch.Tensor = field(default_factory=lambda: torch.zeros(NUM_MOVES))
    V: float = 0.0

    # Carry state from HRM (propagated from parent)
    carry: Optional[torch.Tensor] = None

    # MCTS Statistics
    children: Dict[chess.Move, MCTSNode] = field(default_factory=dict)
    N: int = 0  # Real visit count
    W: float = 0.0  # Real total value
    Q: float = 0.0  # Real mean value (W/N)

    # Virtual Loss for batching
    v_loss: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def total_N(self) -> int:
        """Total visits including virtual loss."""
        return self.N + self.v_loss

    def select_child(self, c_puct: float) -> Tuple[chess.Move, MCTSNode]:
        """PUCT selection using total_N (with virtual loss)."""
        best_score = -float("inf")
        best_move = None
        best_node = None

        # Using AlphaZero PUCT formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        # Note: We use total_N for both parent and child during selection
        sqrt_N = math.sqrt(self.total_N)

        for move, child in self.children.items():
            move_idx = encode_move(move)
            p = self.P[move_idx].item()

            # Virtual loss affects child.total_N, making it less attractive
            u_score = c_puct * p * sqrt_N / (1 + child.total_N)
            score = child.Q + u_score

            if score > best_score:
                best_score = score
                best_move = move
                best_node = child

        return best_move, best_node

    def expand(self, policy: torch.Tensor):
        """Expand all legal moves."""
        self.P = policy
        for move in self.board.legal_moves:
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = MCTSNode(board=new_board, parent=self, move=move)

    def add_virtual_loss(self):
        """Add virtual loss to this node and all ancestors."""
        node = self
        while node:
            node.v_loss += 1
            node = node.parent

    def remove_virtual_loss(self):
        """Remove virtual loss from this node and all ancestors."""
        node = self
        while node:
            node.v_loss -= 1
            node = node.parent

    def backup(self, value: float):
        """Update node statistics recursively."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backup(-value)  # Flip value for opponent's turn


class MCTS:
    """
    Batched MCTS for HRMChess.

    Args:
        model: HRMChess instance
        c_puct: Exploration constant
        num_simulations: Total number of simulations per search
        batch_size: Number of simulations to run in parallel on GPU
        device: Device to run model on
    """

    def __init__(
        self,
        model: HRMChess,
        c_puct: float = 1.5,
        num_simulations: int = 800,
        batch_size: int = 16,
        device: str = "cpu",
    ):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.target_dtype = (
            torch.bfloat16 if self.device.type in ("cuda", "mps") else torch.float32
        )

    @torch.no_grad()
    def search(
        self, board: chess.Board, root_carry: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform MCTS search and return move probabilities.
        """
        root = MCTSNode(board=board, carry=root_carry)

        # Initial expansion of root
        self._evaluate_nodes([root])
        self._add_dirichlet_noise(root)

        # Batch simulation loop
        # We run num_simulations / batch_size steps, each evaluating batch_size leaves.
        num_batches = math.ceil(self.num_simulations / self.batch_size)

        for _ in range(num_batches):
            leaves = []

            # Select batch_size leaves using virtual loss
            for _ in range(self.batch_size):
                node = root
                while not node.is_leaf() and not node.board.is_game_over():
                    _, node = node.select_child(self.c_puct)

                # If terminal or already in batch, we might get duplicates,
                # but virtual loss handles selecting different paths.
                node.add_virtual_loss()
                leaves.append(node)

            # Evaluate the unique non-terminal leaves
            eval_batch = [n for n in leaves if not n.board.is_game_over()]
            if eval_batch:
                self._evaluate_nodes(eval_batch)

            # Backup values and remove virtual losses
            for node in leaves:
                node.remove_virtual_loss()
                if node.board.is_game_over():
                    # Re-calculate terminal value
                    res = node.board.result()
                    if res == "1-0":
                        val = 1.0 if node.board.turn == chess.BLACK else -1.0
                    elif res == "0-1":
                        val = 1.0 if node.board.turn == chess.WHITE else -1.0
                    else:
                        val = 0.0
                    node.backup(val)
                else:
                    node.backup(node.V)

        # Return action probabilities based on visit counts
        probs = torch.zeros(NUM_MOVES)
        for move, child in root.children.items():
            probs[encode_move(move)] = child.N

        return probs / probs.sum()

    def _evaluate_nodes(self, nodes: List[MCTSNode]):
        """Batched evaluation of nodes using the model."""
        B = len(nodes)
        board_tensors = []
        for node in nodes:
            board_tensors.append(encode_board(node.board, history=[]))

        inputs = torch.stack(board_tensors).to(self.device).to(self.target_dtype)
        batch = {
            "inputs": inputs,
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=self.device),
        }

        # In MCTS, we simplify carry: start fresh for each leaf OR propagate.
        # Propagation is better for HRM.
        # We take parent carries if they exist.
        parent_carries = [n.parent.carry if n.parent else None for n in nodes]

        # TODO: True batched carry propagation for mixed parent states.
        # For now, we use a single batch initial carry if no parent carries are available.
        # If we have mixed carries, we'd need to cat them.

        if all(c is not None for c in parent_carries):
            # Concatenate parent carries along batch dimension
            # HRM carry structure: z_H [B, S, H], z_L [B, S, H]
            z_H = torch.cat([c.inner_carry.z_H for c in parent_carries], dim=0)
            z_L = torch.cat([c.inner_carry.z_L for c in parent_carries], dim=0)

            from chessmodels.hrm.hrm_act_v1 import (
                HierarchicalReasoningModel_ACTV1InnerCarry,
                HierarchicalReasoningModel_ACTV1Carry,
            )

            inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)
            carry = HierarchicalReasoningModel_ACTV1Carry(
                inner_carry=inner_carry,
                steps=torch.zeros(B, dtype=torch.int32, device=self.device),
                halted=torch.ones(B, dtype=torch.bool, device=self.device),
                current_data={k: v.clone() for k, v in batch.items()},
            )
        else:
            carry = self.model.initial_carry(batch)

        new_carry, outputs = self.model(carry, batch)

        policies = F.softmax(outputs["policy"], dim=-1)  # [B, 4672]
        values = outputs["value"]  # [B, 1]

        for i, node in enumerate(nodes):
            # Save carry for children
            # Slice the batch to get individual carry
            from chessmodels.hrm.hrm_act_v1 import (
                HierarchicalReasoningModel_ACTV1InnerCarry,
                HierarchicalReasoningModel_ACTV1Carry,
            )

            node_inner = HierarchicalReasoningModel_ACTV1InnerCarry(
                z_H=new_carry.inner_carry.z_H[i : i + 1],
                z_L=new_carry.inner_carry.z_L[i : i + 1],
            )
            node.carry = HierarchicalReasoningModel_ACTV1Carry(
                inner_carry=node_inner,
                steps=new_carry.steps[i : i + 1],
                halted=new_carry.halted[i : i + 1],
                current_data={
                    k: v[i : i + 1] for k, v in new_carry.current_data.items()
                },
            )

            node.V = values[i].item()

            # Mask illegal moves and re-normalize
            mask = legal_mask(node.board).to(self.device)
            p = policies[i] * mask
            if p.sum() > 0:
                p /= p.sum()
            else:
                p = mask.float() / mask.sum()

            node.expand(p.cpu())

    def _add_dirichlet_noise(
        self, node: MCTSNode, alpha: float = 0.3, epsilon: float = 0.25
    ):
        """Add noise to the root node to encourage exploration."""
        legal_moves = list(node.board.legal_moves)
        noise = np.random.dirichlet([alpha] * len(legal_moves))

        for i, move in enumerate(legal_moves):
            idx = encode_move(move)
            node.P[idx] = (1 - epsilon) * node.P[idx] + epsilon * noise[i]
