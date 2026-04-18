#!/usr/bin/env python3
"""
Step 3: Evaluate model against Stockfish.

Usage:
    python scripts/s3_eval.py --checkpoint checkpoints/supervised/epoch_5.pt --sf_elo 1500 --games 20

Default config from chess.yaml (mac_mini): 7M params.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from evaluate_chess import evaluate


parser = argparse.ArgumentParser(description="HRMChess Evaluation")
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to model checkpoint"
)
parser.add_argument(
    "--config",
    type=str,
    default="mac_mini",
    choices=["full", "mac_mini"],
    help="Model config",
)
parser.add_argument("--sf_elo", type=int, default=1500, help="Stockfish Elo level")
parser.add_argument("--games", type=int, default=20, help="Number of games to play")
parser.add_argument(
    "--stockfish_path",
    type=str,
    default="/usr/local/bin/stockfish",
    help="Path to Stockfish binary",
)
parser.add_argument(
    "--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu"
)
parser.add_argument(
    "--forward_dtype",
    type=str,
    default="auto",
    help="Forward dtype override: auto, float32, float16, or bfloat16",
)
parser.add_argument(
    "--time_limit",
    type=float,
    default=0.5,
    help="Stockfish time limit per move (seconds)",
)

args = parser.parse_args()

evaluate(
    checkpoint_path=args.checkpoint,
    config_name=args.config,
    sf_path=args.stockfish_path,
    sf_elo=args.sf_elo,
    n_games=args.games,
    sf_time_limit=args.time_limit,
    device_str=args.device,
    forward_dtype_str=args.forward_dtype,
)
