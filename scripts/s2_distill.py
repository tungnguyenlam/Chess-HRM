#!/usr/bin/env python3
"""
Step 2: Distillation training from Stockfish soft labels.
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
    "--data", type=str, required=True, help="Path to Stockfish soft JSONL file"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to supervised checkpoint to start from",
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
parser.add_argument("--min_depth", type=int, default=12, help="Min search depth filter")
parser.add_argument(
    "--temperature", type=float, default=100.0, help="Softmax temperature (cp)"
)
parser.add_argument(
    "--warmup_steps", type=int, default=None, help="LR warmup steps override"
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoints/distill",
    help="Checkpoint output dir",
)
parser.add_argument(
    "--resume_from", type=str, default=None, help="Distillation checkpoint to resume"
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

args = parser.parse_args()
provided_flags = set(sys.argv[1:])

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
min_depth = (
    args.min_depth if args.min_depth != 12 else distill_cfg.get("min_depth", 12)
)
temperature = (
    args.temperature
    if args.temperature != 100.0
    else distill_cfg.get("temperature", 100.0)
)
warmup_steps = (
    args.warmup_steps
    if args.warmup_steps is not None
    else distill_cfg.get("warmup_steps", 200)
)

# If --accum_steps is present on the CLI, always respect it.
# Otherwise fall back to the config value when available.
accum_steps = args.accum_steps
if "--accum_steps" not in provided_flags and "accum_steps" in distill_cfg:
    accum_steps = distill_cfg["accum_steps"]

num_workers = (
    args.num_workers
    if args.num_workers is not None
    else distill_cfg.get("num_workers")
)
log_every_steps = (
    args.log_every_steps
    if args.log_every_steps is not None
    else distill_cfg.get("log_every_steps", 10)
)
forward_dtype = (
    args.forward_dtype if args.forward_dtype is not None else cfg.get("forward_dtype")
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
    resume_from=args.resume_from,
    device_str=args.device,
    use_wandb=args.wandb,
    accum_steps=accum_steps,
    warmup_steps=warmup_steps,
    num_workers=num_workers,
    log_every_steps=log_every_steps,
    forward_dtype_str=forward_dtype,
)
