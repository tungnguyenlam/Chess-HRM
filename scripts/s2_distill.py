#!/usr/bin/env python3
"""
Step 2: Distillation from Stockfish soft labels.

Usage:
    python scripts/s2_distill.py --data path/to/stockfish_soft.jsonl --checkpoint checkpoints/supervised/epoch_5.pt

Default config from chess.yaml (mac_mini): 7M params.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from omegaconf import OmegaConf

# Load chess.yaml defaults
chess_cfg = OmegaConf.load("config/chess.yaml")

parser = argparse.ArgumentParser(description="HRMChess Distillation Training")
parser.add_argument(
    "--data", type=str, required=True, help="Path to Stockfish soft labels JSONL"
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
)
parser.add_argument(
    "--config",
    type=str,
    default="mac_mini",
    choices=["full", "mac_mini"],
    help="Model config",
)
parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
parser.add_argument("--min_depth", type=int, default=12, help="Min game depth filter")
parser.add_argument(
    "--temperature", type=float, default=100.0, help="Soft label temperature"
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoints/distill",
    help="Checkpoint output dir",
)
parser.add_argument(
    "--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu"
)
parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")

args = parser.parse_args()

# Get config defaults
cfg = chess_cfg[args.config]
distill_cfg = cfg.get("distill", {})

# Override with CLI args
epochs = args.epochs if args.epochs != 2 else distill_cfg.get("epochs", 2)
batch_size = (
    args.batch_size if args.batch_size != 8 else distill_cfg.get("batch_size", 8)
)
lr = args.lr if args.lr != 3e-5 else distill_cfg.get("lr", 3e-5)
weight_decay = (
    args.weight_decay
    if args.weight_decay != 1e-2
    else distill_cfg.get("weight_decay", 1e-2)
)
grad_clip = (
    args.grad_clip if args.grad_clip != 1.0 else distill_cfg.get("grad_clip", 1.0)
)
min_depth = args.min_depth if args.min_depth != 12 else distill_cfg.get("min_depth", 12)
temperature = (
    args.temperature
    if args.temperature != 100.0
    else distill_cfg.get("temperature", 100.0)
)

from chessgame.train.distill import train

train(
    data_path=args.data,
    checkpoint_path=args.checkpoint,
    config_name=args.config,
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    weight_decay=weight_decay,
    grad_clip=grad_clip,
    min_depth=min_depth,
    temperature=temperature,
    checkpoint_dir=args.checkpoint_dir,
    device_str=args.device,
    use_wandb=args.wandb,
)
