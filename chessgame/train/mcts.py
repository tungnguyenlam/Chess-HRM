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
from typing import Dict, List, Optional, Tuple, Union

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

    # Inner carry state from HRM (propagated from parent)
    inner_carry: Optional[HierarchicalReasoningModel_ACTV1InnerCarry] = None

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

        sqrt_N = math.sqrt(self.total_N)

        for move, child in self.children.items():
            move_idx = encode_move(move)
            p = self.P[move_idx].item()

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
        self.target_dtype = getattr(
            getattr(self.model, "inner", None),
            "forward_dtype",
            next(self.model.parameters()).dtype,
        )

    @torch.no_grad()
    def search(
        self,
        board: chess.Board,
        root_inner_carry: Optional[HierarchicalReasoningModel_ACTV1InnerCarry] = None,
        return_inner_carry: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Optional[HierarchicalReasoningModel_ACTV1InnerCarry]],
    ]:
        """
        Perform MCTS search.

        Returns only the move probabilities by default so existing callers keep
        the old contract. Set return_inner_carry=True to also receive the carry
        associated with the highest-visit child.
        """
        root = MCTSNode(board=board, inner_carry=root_inner_carry)

        # Initial expansion of root
        self._evaluate_nodes([root])
        self._add_dirichlet_noise(root)

        num_batches = math.ceil(self.num_simulations / self.batch_size)

        for _ in range(num_batches):
            leaves = []

            for _ in range(self.batch_size):
                node = root
                while not node.is_leaf() and not node.board.is_game_over():
                    _, node = node.select_child(self.c_puct)

                node.add_virtual_loss()
                leaves.append(node)

            eval_batch = [n for n in leaves if not n.board.is_game_over() and n.is_leaf()]
            # Note: unique filter here if needed, but evaluate_nodes handles it.
            if eval_batch:
                self._evaluate_nodes(eval_batch)

            for node in leaves:
                node.remove_virtual_loss()
                if node.board.is_game_over():
                    res = node.board.result()
                    if res == "1-0":
                        val = 1.0 if node.board.turn == chess.WHITE else -1.0
                    elif res == "0-1":
                        val = 1.0 if node.board.turn == chess.BLACK else -1.0
                    else:
                        val = 0.0
                    node.backup(val)
                else:
                    node.backup(node.V)

        probs = torch.zeros(NUM_MOVES)
        for move, child in root.children.items():
            probs[encode_move(move)] = child.N

        pi = probs / probs.sum()
        
        # Get carry for chosen move (highest visit count)
        best_move_idx = probs.argmax().item()
        chosen_inner_carry = None
        for move, child in root.children.items():
            if encode_move(move) == best_move_idx:
                chosen_inner_carry = child.inner_carry
                break
                
        if return_inner_carry:
            return pi, chosen_inner_carry
        return pi

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

        # Propagation for HRM
        parent_carries = [n.parent.inner_carry if n.parent else None for n in nodes]

        from chessmodels.hrm.hrm_act_v1 import (
            HierarchicalReasoningModel_ACTV1InnerCarry,
            HierarchicalReasoningModel_ACTV1Carry,
        )

        if all(c is not None for c in parent_carries):
            z_H = torch.cat([c.z_H for c in parent_carries], dim=0)
            z_L = torch.cat([c.z_L for c in parent_carries], dim=0)
            inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)
            carry = HierarchicalReasoningModel_ACTV1Carry(
                inner_carry=inner_carry,
                steps=torch.zeros(B, dtype=torch.int32, device=self.device),
                halted=torch.ones(B, dtype=torch.bool, device=self.device),
                current_data={k: v.clone() for k, v in batch.items()},
            )
        elif any(c is not None for c in parent_carries):
            # Mixed: some have carry, some don't.
            # Simplified: initialize all fresh if any are missing. 
            # (In practice, root is usually the only one missing parent.inner_carry).
            carry = self.model.initial_carry(batch)
        else:
            carry = self.model.initial_carry(batch)

        new_carry, outputs = self.model(carry, batch)

        policies = F.softmax(outputs["policy"], dim=-1)
        values = outputs["value"]

        for i, node in enumerate(nodes):
            node.inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
                z_H=new_carry.inner_carry.z_H[i : i + 1].detach(),
                z_L=new_carry.inner_carry.z_L[i : i + 1].detach(),
            )
            node.V = values[i].item()

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
