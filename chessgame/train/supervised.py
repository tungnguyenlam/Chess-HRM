"""
Phase 2: Supervised pretraining from Lichess Elite Database.

Usage:
  python -m chessgame.train.supervised \
      --data path/to/lichess.jsonl \
      --config mac_mini \
      --epochs 5 \
      --batch_size 32 \
      --lr 1e-4 \
      --checkpoint_dir checkpoints/supervised

ACT loss is intentionally excluded here — no reward signal exists yet.
"""

import argparse
import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb

    _WANDB = True
except ImportError:
    _WANDB = False

from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.data.lichess_dataset import LichessEliteDataset, ShardedLichessDataset
from chessgame.train.loss import total_supervised


def _build_batch(board_tensors: torch.Tensor, device: torch.device) -> dict:
    B = board_tensors.shape[0]
    return {
        "inputs": board_tensors.to(device),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=device),
    }


def _cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 500,
    min_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (
        min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def train(
    data_path: str,
    config_name: str = "mac_mini",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    min_depth: int = 18,
    min_elo: int = 0,
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    device_str: str = "auto",
    use_wandb: bool = False,
    accum_steps: int = 1,
    curriculum: bool = False,
):
    # Device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Device: {device}")

    # Config & model
    config = (
        HRMChessConfig.full() if config_name == "full" else HRMChessConfig.mac_mini()
    )
    model = HRMChess(config).to(device)

    # Set target dtype based on device
    target_dtype = torch.float32
    if device.type in ("cuda", "mps"):
        # Use bfloat16 for modern GPUs if available, else float16
        target_dtype = torch.bfloat16
        model = model.to(target_dtype)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Initial Data
    current_min_elo = min_elo
    is_dir = os.path.isdir(data_path)

    def get_loader(elo):
        if is_dir:
            ds = ShardedLichessDataset(data_path, min_depth=min_depth, min_elo=elo)
        else:
            ds = LichessEliteDataset(data_path, min_depth=min_depth, min_elo=elo)

        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0 if device.type == "mps" else 4,
            pin_memory=(device.type == "cuda"),
            shuffle=False if is_dir else True,
        ), ds

    loader, dataset = get_loader(current_min_elo)

    # Estimate total steps if possible
    if hasattr(dataset, "__len__"):
        total_steps = epochs * math.ceil(len(dataset) / batch_size) // accum_steps
    else:
        # Default for streaming
        total_steps = 1000000

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_wandb and _WANDB:
        wandb.init(
            project="hrm-chess-supervised",
            config={
                "config_name": config_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "min_elo": min_elo,
                "curriculum": curriculum,
            },
        )

    step = 0
    start_epoch = 0

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "step" in ckpt:
            step = ckpt["step"]
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]

    carry = None

    for epoch in range(start_epoch, epochs):
        if curriculum and epoch > start_epoch:
            new_min_elo = min_elo + (epoch * 150)
            if new_min_elo != current_min_elo:
                print(f"Curriculum: Increasing min_elo floor to {new_min_elo}")
                current_min_elo = new_min_elo
                loader, dataset = get_loader(current_min_elo)

        model.train()
        running_loss = 0.0

        for i, (board_t, move_idx, value_t) in enumerate(loader):
            # CAST INPUTS TO target_dtype
            batch = _build_batch(board_t.to(target_dtype), device)
            move_idx = move_idx.to(device)
            value_t = value_t.to(device).to(target_dtype)

            if carry is None:
                carry = model.initial_carry(batch)

            carry, outputs = model(carry, batch)

            loss = total_supervised(outputs, move_idx, value_t) / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0:
                step += 1
                current_lr = _cosine_lr(step, total_steps, lr)
                for g in opt.param_groups:
                    g["lr"] = current_lr

                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                opt.zero_grad()

                running_loss += loss.item() * accum_steps
                if step % 100 == 0:
                    avg = running_loss / 100
                    print(
                        f"Epoch {epoch + 1} step {step} | min_elo={current_min_elo} | loss={avg:.4f} lr={current_lr:.2e} norm={grad_norm:.2f}"
                    )
                    running_loss = 0.0
                    if use_wandb and _WANDB:
                        wandb.log(
                            {
                                "loss": avg,
                                "lr": current_lr,
                                "grad_norm": grad_norm,
                                "step": step,
                                "min_elo": current_min_elo,
                            }
                        )

        # Checkpoint after each epoch
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "epoch": epoch + 1,
                    "config_name": config_name,
                    "min_elo": current_min_elo,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    if use_wandb and _WANDB:
        wandb.finish()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="mac_mini", choices=["full", "mac_mini"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_depth", type=int, default=18)
    parser.add_argument("--min_elo", type=int, default=0)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    args = parser.parse_args()

    train(
        data_path=args.data,
        config_name=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_depth=args.min_depth,
        min_elo=args.min_elo,
        curriculum=args.curriculum,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device_str=args.device,
        use_wandb=args.wandb,
        accum_steps=args.accum_steps,
    )


if __name__ == "__main__":
    main()
