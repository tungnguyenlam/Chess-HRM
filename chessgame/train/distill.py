"""
Phase 3: Soft-label distillation from pre-generated Stockfish multipv data.

Reads a JSONL file produced offline by dataset/generate_soft_labels.py.
Trains with KL-divergence policy loss instead of hard cross-entropy.

Usage:
  python -m chessgame.train.distill \
      --data path/to/stockfish_soft.jsonl \
      --checkpoint checkpoints/supervised/epoch_5.pt \
      --config mac_mini \
      --epochs 2 \
      --lr 3e-5 \
      --checkpoint_dir checkpoints/distill
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
from chessgame.data.stockfish_dataset import StockfishSoftDataset
from chessgame.train.loss import total_distill


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
    warmup_steps: int = 200,
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
    checkpoint_path: Optional[str] = None,
    config_name: str = "mac_mini",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 3e-5,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    min_depth: int = 12,
    temperature: float = 100.0,
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    device_str: str = "auto",
    use_wandb: bool = False,
    accum_steps: int = 1,
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

    # Load from supervised checkpoint or distillation resume
    step = 0
    start_epoch = 0

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        # step/epoch will be updated if optimizer exists below
    elif checkpoint_path:
        print(f"Initializing from supervised checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])

    # Data
    dataset = StockfishSoftDataset(
        data_path, min_depth=min_depth, temperature=temperature
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if device.type == "mps" else 4,
        pin_memory=(device.type == "cuda"),
    )

    total_steps = epochs * math.ceil(len(dataset) / batch_size) // accum_steps
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Resume optimizer state if available
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "step" in ckpt:
            step = ckpt["step"]
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]

    if use_wandb and _WANDB:
        wandb.init(
            project="hrm-chess-distill",
            config={
                "config_name": config_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "temperature": temperature,
            },
        )

    carry = None

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for i, (board_t, soft_pi, value_t) in enumerate(loader):
            # CAST INPUTS TO target_dtype
            batch = _build_batch(board_t.to(target_dtype), device)
            soft_pi = soft_pi.to(device).to(target_dtype)
            value_t = value_t.to(device).to(target_dtype)

            if carry is None:
                carry = model.initial_carry(batch)

            carry, outputs = model(carry, batch)

            loss = total_distill(outputs, soft_pi, value_t) / accum_steps
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
                        f"Epoch {epoch + 1} step {step} | loss={avg:.4f} lr={current_lr:.2e} norm={grad_norm:.2f}"
                    )
                    running_loss = 0.0
                    if use_wandb and _WANDB:
                        wandb.log(
                            {
                                "loss": avg,
                                "lr": current_lr,
                                "grad_norm": grad_norm,
                                "step": step,
                            }
                        )

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
    parser.add_argument(
        "--checkpoint", default=None, help="Supervised checkpoint to start from"
    )
    parser.add_argument("--config", default="mac_mini", choices=["full", "mac_mini"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_depth", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=100.0)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument(
        "--resume_from", default=None, help="Distillation checkpoint to resume from"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    args = parser.parse_args()

    train(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_depth=args.min_depth,
        temperature=args.temperature,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device_str=args.device,
        use_wandb=args.wandb,
        accum_steps=args.accum_steps,
    )


if __name__ == "__main__":
    main()
