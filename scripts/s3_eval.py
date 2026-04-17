#!/usr/bin/env python3
"""
Step 3: Evaluate model against Stockfish.

Usage:
    python scripts/s3_eval.py --checkpoint checkpoints/supervised/epoch_5.pt --sf_elo 1500 --games 20

Default config from chess.yaml (mac_mini): 7M params.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from omegaconf import OmegaConf

# Load chess.yaml defaults
chess_cfg = OmegaConf.load("config/chess.yaml")

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
    "--time_limit",
    type=float,
    default=0.5,
    help="Stockfish time limit per move (seconds)",
)

args = parser.parse_args()

import torch
from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
import chess.engine

config = HRMChessConfig.full() if args.config == "full" else HRMChessConfig.mac_mini()
model = HRMChess(config)

# Load checkpoint
state = torch.load(args.checkpoint, map_location="cpu")
model.load_state_dict(state["model"])
print(f"Loaded checkpoint: {args.checkpoint}")

# Device
if args.device == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(args.device)

model = model.to(device)
model.eval()

# Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
engine.configure({"Skill Level": max(0, min(20, args.sf_elo // 100))})

from evaluate_chess import play_game

print(f"Playing {args.games} games vs Stockfish @ Elo {args.sf_elo}...")

results = []
model_carry = None

for i in range(args.games):
    model_plays_white = i % 2 == 0
    result = play_game(
        model=model,
        engine=engine,
        model_plays_white=model_plays_white,
        sf_time_limit=args.time_limit,
        device=device,
    )
    results.append(result)
    print(
        f"Game {i + 1}: {'Model wins' if result == 1 else 'Stockfish wins' if result == -1 else 'Draw'} (as {'white' if model_plays_white else 'black'})"
    )

engine.quit()

# Summary
wins = sum(1 for r in results if r == 1)
losses = sum(1 for r in results if r == -1)
draws = sum(1 for r in results if r == 0)

print(f"\nResults: {wins}W / {draws}D / {losses}L")
win_rate = (wins + 0.5 * draws) / len(results)
print(f"Win rate: {win_rate:.2%}")

if win_rate > 0 and win_rate < 1:
    elo_diff = 400 * (win_rate / (1 - win_rate)) ** 0.5  # Simplified
    print(f"Estimated model Elo: ~{args.sf_elo + elo_diff:.0f}")
