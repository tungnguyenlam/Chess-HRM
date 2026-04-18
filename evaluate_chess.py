"""
Evaluate HRMChess against Stockfish at a configurable Elo level.

Usage:
  python evaluate_chess.py \\
      --checkpoint checkpoints/supervised/epoch_5.pt \\
      --config mac_mini \\
      --sf_elo 1500 \\
      --games 20 \\
      --stockfish_path /usr/local/bin/stockfish

Estimates model Elo using the 400-point logistic formula:
  elo_diff = 400 * log10(win_rate / (1 - win_rate))
  estimated_elo = target_elo + elo_diff
"""

import argparse
import math

import chess
import chess.engine
import torch

from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.encoding.board_encoder import encode_board
from chessgame.encoding.move_encoder import decode_move, legal_mask
from chessgame.train.runtime import log_runtime, resolve_training_runtime


_OUTCOME_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


@torch.no_grad()
def get_model_move(
    model: HRMChess,
    board: chess.Board,
    carry,
    device: torch.device,
    target_dtype: torch.dtype,
) -> tuple[chess.Move, object]:
    """Run one forward pass and pick the highest-scoring legal move."""
    board_t = encode_board(board, history=[]).unsqueeze(0).to(device).to(target_dtype)
    B = 1
    batch = {
        "inputs": board_t,
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=device),
    }

    if carry is None:
        carry = model.initial_carry(batch)

    carry, outputs = model(carry, batch)

    policy = outputs["policy"][0]  # [4672]
    mask = legal_mask(board).to(device)
    policy = policy.masked_fill(~mask, float("-inf"))
    move_idx = policy.argmax().item()
    move = decode_move(move_idx, board)

    # Fallback: random legal move if decode fails
    if move is None or move not in board.legal_moves:
        import random

        move = random.choice(list(board.legal_moves))

    return move, carry


def play_game(
    model: HRMChess,
    engine: chess.engine.SimpleEngine,
    model_plays_white: bool,
    sf_time_limit: float,
    device: torch.device,
    target_dtype: torch.dtype,
    max_moves: int = 200,
) -> float:
    """
    Play one game.  Returns result from white's perspective: 1, 0, -1.
    """
    board = chess.Board()
    model_carry = None
    model_color = chess.WHITE if model_plays_white else chess.BLACK

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            move, model_carry = get_model_move(
                model,
                board,
                model_carry,
                device,
                target_dtype,
            )
        else:
            result = engine.play(board, chess.engine.Limit(time=sf_time_limit))
            move = result.move
            model_carry = None  # reset carry after opponent move

        board.push(move)

    outcome_str = board.result()
    if outcome_str == "*":  # max moves reached
        outcome_str = "1/2-1/2"
    return _OUTCOME_MAP[outcome_str]


def evaluate(
    checkpoint_path: str,
    config_name: str = "mac_mini",
    sf_path: str = "stockfish",
    sf_elo: int = 1500,
    n_games: int = 20,
    sf_time_limit: float = 0.1,
    device_str: str = "auto",
    forward_dtype_str: str = "auto",
) -> float:
    # Model
    config = (
        HRMChessConfig.full() if config_name == "full" else HRMChessConfig.mac_mini()
    )
    runtime = resolve_training_runtime(
        config=config,
        device_str=device_str,
        forward_dtype_str=forward_dtype_str,
    )
    log_runtime(runtime, lambda message: print(message, flush=True))

    device = runtime.device
    target_dtype = runtime.forward_dtype

    model = HRMChess(config).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})

    wins = draws = losses = 0

    for i in range(n_games):
        model_white = i % 2 == 0
        result = play_game(
            model,
            engine,
            model_white,
            sf_time_limit,
            device,
            target_dtype,
        )

        if not model_white:
            result = -result  # normalise to model's perspective

        if result > 0:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1

        print(
            f"Game {i + 1}/{n_games}: {'W' if result > 0 else 'D' if result == 0 else 'L'} "
            f"(model {'white' if model_white else 'black'})"
        )

    engine.quit()

    win_rate = (wins + 0.5 * draws) / n_games
    win_rate = max(win_rate, 1e-6)
    win_rate = min(win_rate, 1 - 1e-6)
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate))
    estimated_elo = sf_elo + elo_diff

    print(f"\nResults vs Stockfish {sf_elo}: {wins}W / {draws}D / {losses}L")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Estimated Elo: {estimated_elo:.0f}")

    return estimated_elo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="mac_mini", choices=["full", "mac_mini"])
    parser.add_argument("--stockfish_path", default="stockfish")
    parser.add_argument("--sf_elo", type=int, default=1500)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument(
        "--sf_time", type=float, default=0.1, help="Stockfish time per move in seconds"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--forward_dtype",
        default="auto",
        help="Forward dtype override: auto, float32, float16, or bfloat16",
    )
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        sf_path=args.stockfish_path,
        sf_elo=args.sf_elo,
        n_games=args.games,
        sf_time_limit=args.sf_time,
        device_str=args.device,
        forward_dtype_str=args.forward_dtype,
    )


if __name__ == "__main__":
    main()
