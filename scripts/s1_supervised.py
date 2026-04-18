#!/usr/bin/env python3
"""
Step 1: Supervised training on Lichess elite games.

Usage:
    python scripts/s1_supervised.py --data path/to/lichess.jsonl

Default config from chess.yaml (mac_mini): 7M params.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from omegaconf import OmegaConf

# Load chess.yaml defaults
chess_cfg = OmegaConf.load("config/chess.yaml")

parser = argparse.ArgumentParser(description="HRMChess Supervised Training")
parser.add_argument(
    "--data", type=str, required=True, help="Path to Lichess JSONL file"
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
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
parser.add_argument("--min_depth", type=int, default=0, help="Min game depth filter")
parser.add_argument(
    "--warmup_steps", type=int, default=None, help="LR warmup steps override"
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoints/supervised",
    help="Checkpoint output dir",
)
parser.add_argument(
    "--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu"
)
parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
parser.add_argument(
    "--accum_steps", type=int, default=1, help="Gradient accumulation steps"
)
parser.add_argument(
    "--num_workers", type=int, default=None, help="DataLoader workers override"
)
parser.add_argument(
    "--log_every_steps", type=int, default=None, help="Optimizer-step logging interval"
)
parser.add_argument(
    "--forward_dtype",
    type=str,
    default=None,
    help="Forward dtype override: auto, float32, float16, or bfloat16",
)
parser.add_argument("--min_elo", type=int, default=1800, help="Min Elo filter")
parser.add_argument("--curriculum", action="store_true", help="Enable Elo curriculum")

args = parser.parse_args()
provided_flags = set(sys.argv[1:])

# Get config defaults
cfg = chess_cfg[args.config]
supervised_cfg = cfg.get("supervised", {})

# Override with CLI args
epochs = args.epochs if args.epochs != 5 else supervised_cfg.get("epochs", 5)
batch_size = (
    args.batch_size if args.batch_size != 8 else supervised_cfg.get("batch_size", 8)
)
lr = args.lr if args.lr != 1e-4 else supervised_cfg.get("lr", 1e-4)
weight_decay = (
    args.weight_decay
    if args.weight_decay != 1e-2
    else supervised_cfg.get("weight_decay", 1e-2)
)
grad_clip = (
    args.grad_clip if args.grad_clip != 1.0 else supervised_cfg.get("grad_clip", 1.0)
)
# Use CLI arg for min_depth/min_elo if provided, otherwise check config, otherwise use our new defaults
min_depth = args.min_depth
min_elo = args.min_elo
warmup_steps = (
    args.warmup_steps
    if args.warmup_steps is not None
    else supervised_cfg.get("warmup_steps", 500)
)

# If --accum_steps is present on the CLI, always respect it.
# Otherwise fall back to the config value when available.
accum_steps = args.accum_steps
if "--accum_steps" not in provided_flags and "accum_steps" in supervised_cfg:
    accum_steps = supervised_cfg["accum_steps"]

num_workers = (
    args.num_workers
    if args.num_workers is not None
    else supervised_cfg.get("num_workers")
)
log_every_steps = (
    args.log_every_steps
    if args.log_every_steps is not None
    else supervised_cfg.get("log_every_steps", 10)
)
forward_dtype = (
    args.forward_dtype if args.forward_dtype is not None else cfg.get("forward_dtype")
)

from chessgame.train.supervised import train

train(
    data_path=args.data,
    config_name=args.config,
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    weight_decay=weight_decay,
    grad_clip=grad_clip,
    min_depth=min_depth,
    min_elo=min_elo,
    curriculum=args.curriculum,
    checkpoint_dir=args.checkpoint_dir,
    device_str=args.device,
    use_wandb=args.wandb,
    accum_steps=accum_steps,
    warmup_steps=warmup_steps,
    num_workers=num_workers,
    log_every_steps=log_every_steps,
    forward_dtype_str=forward_dtype,
    resume_from=args.checkpoint,
)
