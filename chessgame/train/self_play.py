"""
Phase 4: Reinforcement Learning via Self-Play and MCTS.

Usage:
  python -m chessgame.train.self_play \
      --checkpoint checkpoints/distill/epoch_2.pt \
      --config mac_mini \
      --num_games 100 \
      --sims 400 \
      --batch_size 32 \
      --lr 1e-5
"""

import argparse
from typing import List, Tuple, Optional

import chess
import torch
import torch.nn as nn

from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.train.mcts import MCTS
from chessgame.data.replay_buffer import ReplayBuffer
from chessgame.encoding.board_encoder import encode_board
from chessgame.train.loss import total_rl
from chessgame.train.runtime import log_runtime, resolve_training_runtime
from chessmodels.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1InnerCarry

try:
    import wandb

    _WANDB = True
except ImportError:
    _WANDB = False


def play_game(
    model: HRMChess,
    num_sims: int,
    device: torch.device,
    temperature: float = 1.0,
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Play one game against self using MCTS and return (board_t, pi, turn) tuples.
    turn is 1 for White, -1 for Black.
    """
    mcts = MCTS(model, num_simulations=num_sims, device=str(device))
    board = chess.Board()
    game_data = []
    
    # Root carry propagation for "unfolding reasoning"
    current_inner_carry: Optional[HierarchicalReasoningModel_ACTV1InnerCarry] = None

    while not board.is_game_over() and board.fullmove_number <= 200:
        # Move probabilities from MCTS search
        # Pass the carry from the winning child of the previous move's search
        pi, current_inner_carry = mcts.search(
            board,
            root_inner_carry=current_inner_carry,
            return_inner_carry=True,
        )

        # Store data: (board_tensor, policy, turn)
        board_t = encode_board(board, history=[])
        turn = 1 if board.turn == chess.WHITE else -1
        game_data.append((board_t, pi, turn))

        # Select move
        if temperature == 0:
            move_idx = pi.argmax().item()
        else:
            p = pi.pow(1.0 / temperature)
            p /= p.sum()
            move_idx = torch.multinomial(p, 1).item()

        from chessgame.encoding.move_encoder import decode_move
        move = decode_move(move_idx, board)
        board.push(move)

    # Outcome from White's perspective
    outcome_white = 0.0
    if board.is_game_over():
        res = board.result()
        if res == "1-0":
            outcome_white = 1.0
        elif res == "0-1":
            outcome_white = -1.0

    return game_data, outcome_white


def train_step(
    model: HRMChess,
    opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    target_dtype: torch.dtype,
) -> float:
    """Run one training step on the replay buffer."""
    if len(buffer) < batch_size:
        return 0.0

    boards, mcts_pis, outcomes = buffer.sample(batch_size)
    boards = boards.to(device).to(target_dtype)
    mcts_pis = mcts_pis.to(device).to(target_dtype)
    outcomes = outcomes.to(device).to(target_dtype)

    B = batch_size
    batch = {
        "inputs": boards,
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=device),
    }

    model.train()
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)

    loss = total_rl(outputs, mcts_pis, outcomes)

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="mac_mini", choices=["full", "mac_mini"])
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--forward_dtype", default="auto")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # Model
    config = (
        HRMChessConfig.full() if args.config == "full" else HRMChessConfig.mac_mini()
    )
    runtime = resolve_training_runtime(
        config=config,
        device_str=args.device,
        forward_dtype_str=args.forward_dtype,
    )
    log_runtime(runtime, lambda message: print(message, flush=True))

    device = runtime.device
    target_dtype = runtime.forward_dtype

    model = HRMChess(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    if args.wandb and _WANDB:
        wandb.init(
            project="hrm-gab-chess",
            name=f"rl-{args.config}",
            config={
                **vars(args),
                "resolved_device": str(device),
                "resolved_forward_dtype": str(target_dtype).replace("torch.", ""),
            },
        )

    for g in range(args.num_games):
        # 1. Play game
        print(f"Playing game {g + 1}/{args.num_games}...")
        game_data, outcome_white = play_game(model, args.sims, device)

        # 2. Add to buffer
        processed_data = []
        for board_t, pi, turn in game_data:
            # outcome for current player
            z = outcome_white * turn
            processed_data.append((board_t, pi, z))

        buffer.add(processed_data)

        # 3. Train
        if len(buffer) >= args.batch_size:
            loss = train_step(model, opt, buffer, args.batch_size, device, target_dtype)
            print(f"Game {g + 1} | Buffer: {len(buffer)} | Loss: {loss:.4f}")
            if args.wandb and _WANDB:
                wandb.log({"loss": loss, "game": g, "buffer_size": len(buffer)})

    if args.wandb and _WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
