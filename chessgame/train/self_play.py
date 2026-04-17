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
from typing import List, Tuple

import chess
import torch
import torch.nn as nn

from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.train.mcts import MCTS
from chessgame.data.replay_buffer import ReplayBuffer
from chessgame.encoding.board_encoder import encode_board
from chessgame.train.loss import total_rl

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
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Play one game against self using MCTS and return (board, pi) pairs."""
    mcts = MCTS(model, num_simulations=num_sims, device=str(device))
    board = chess.Board()
    game_data = []

    while not board.is_game_over() and board.fullmove_number <= 200:
        # Move probabilities from MCTS search
        pi = mcts.search(board)

        # Store data from current perspective
        board_t = encode_board(board, history=[])  # [8, 8, 119]
        game_data.append((board_t, pi))

        # Select move
        if temperature == 0:
            move_idx = pi.argmax().item()
        else:
            # Apply temperature to visit counts
            p = pi.pow(1.0 / temperature)
            p /= p.sum()
            move_idx = torch.multinomial(p, 1).item()

        from chessgame.encoding.move_encoder import decode_move

        move = decode_move(move_idx, board)
        board.push(move)

    # Outcome from perspective of player who just moved
    # res = board.result() -> "1-0", "0-1", "1/2-1/2"
    outcome = 0.0
    if board.is_game_over():
        res = board.result()
        if res == "1-0":
            outcome = 1.0
        elif res == "0-1":
            outcome = -1.0

    # Return (board, pi, outcome_from_white_perspective)
    # We'll flip outcome during training based on who's turn it was
    return game_data, outcome


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

    # outcomes are from White's perspective.
    # We need to flip them if it was Black's turn?
    # Actually, in play_game, we should store outcome from each player's perspective.

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
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    target_dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32

    # Model
    config = (
        HRMChessConfig.full() if args.config == "full" else HRMChessConfig.mac_mini()
    )
    model = HRMChess(config).to(device).to(target_dtype)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    if args.wandb and _WANDB:
        wandb.init(project="hrm-chess-rl", config=vars(args))

    for g in range(args.num_games):
        # 1. Play game
        print(f"Playing game {g + 1}/{args.num_games}...")
        game_data, outcome = play_game(model, args.sims, device)

        # 2. Add to buffer
        # Flip outcome based on whose turn it was
        # White turn: outcome, Black turn: -outcome
        # We need turn info in game_data.
        # Let's assume board_t is encoded from perspective of player whose turn it is.
        # Board encoder already flips for Black.

        processed_data = []
        for i, (board_t, pi) in enumerate(game_data):
            # outcome is for White.
            # If i is even, it's White's turn (in starting board).
            # Actually we should track board.turn during play_game.
            # Simplified: assuming even steps are White, odd are Black.
            z = outcome if i % 2 == 0 else -outcome
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
