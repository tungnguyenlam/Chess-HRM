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
import time
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
from chessgame.train.runtime import log_runtime, resolve_training_runtime


def _log(message: str) -> None:
    print(message, flush=True)


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
    warmup_steps: int = 200,
    num_workers: Optional[int] = None,
    log_every_steps: int = 10,
    forward_dtype_str: Optional[str] = None,
):
    log_every_steps = max(1, log_every_steps)

    # Config & model
    config = (
        HRMChessConfig.full() if config_name == "full" else HRMChessConfig.mac_mini()
    )
    # FORCE ACT OFF for distillation training
    config.halt_max_steps = 1

    runtime = resolve_training_runtime(
        config=config,
        device_str=device_str,
        num_workers=num_workers,
        forward_dtype_str=forward_dtype_str,
    )
    log_runtime(runtime, _log)

    device = runtime.device
    target_dtype = runtime.forward_dtype
    resolved_workers = runtime.num_workers

    model = HRMChess(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    _log(f"Parameters: {n_params:,}")
    _log(
        "Training config:"
        f" data={data_path}"
        f" epochs={epochs}"
        f" batch_size={batch_size}"
        f" accum_steps={accum_steps}"
        f" lr={lr:.2e}"
        f" warmup_steps={warmup_steps}"
        f" min_depth={min_depth}"
        f" temperature={temperature}"
        f" checkpoint_dir={checkpoint_dir}"
    )

    # Load from supervised checkpoint or distillation resume
    step = 0
    start_epoch = 0

    if resume_from and os.path.exists(resume_from):
        _log(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        # step/epoch will be updated if optimizer exists below
    elif checkpoint_path:
        _log(f"Initializing from supervised checkpoint: {checkpoint_path}")
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
        num_workers=resolved_workers,
        pin_memory=(device.type == "cuda"),
    )

    batches_per_epoch = math.ceil(len(dataset) / batch_size)
    total_steps = epochs * math.ceil(batches_per_epoch / accum_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt.zero_grad(set_to_none=True)

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
            project="hrm-gab-chess",
            name=f"distill-{config_name}",
            config={
                "config_name": config_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "temperature": temperature,
                "halt_max_steps": config.halt_max_steps,
                "device": str(device),
                "forward_dtype": runtime.forward_dtype_name,
                "num_workers": resolved_workers,
            },
        )
    elif use_wandb and not _WANDB:
        _log("W&B requested but wandb is not installed; continuing without remote logging.")

    carry = None

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_steps = 0
        epoch_loss = 0.0
        epoch_steps = 0
        pending_batches = 0
        pending_loss = 0.0
        pending_samples = 0
        pending_data_wait_s = 0.0
        pending_compute_s = 0.0
        last_lr = opt.param_groups[0]["lr"]
        last_grad_norm = 0.0
        epoch_start = time.perf_counter()
        batch_wait_start = epoch_start
        samples_seen = 0
        _log(f"Epoch {epoch + 1}/{epochs} start | waiting for first batch from DataLoader...")

        for i, (board_t, soft_pi, value_t) in enumerate(loader):
            batch_ready = time.perf_counter()
            data_wait_s = batch_ready - batch_wait_start
            samples_seen += board_t.shape[0]
            if i == 0:
                _log(
                    f"Epoch {epoch + 1}/{epochs} first batch loaded in {data_wait_s:.2f}s | "
                    f"batch_shape={tuple(board_t.shape)}"
                )

            # CAST INPUTS TO target_dtype
            batch = _build_batch(board_t.to(target_dtype), device)
            soft_pi = soft_pi.to(device).to(target_dtype)
            value_t = value_t.to(device).to(target_dtype)

            if carry is None:
                _log(f"Epoch {epoch + 1}/{epochs} initializing recurrent carry state")
                carry = model.initial_carry(batch)

            batch_compute_start = time.perf_counter()
            carry, outputs = model(carry, batch)

            loss = total_distill(outputs, soft_pi, value_t) / accum_steps
            loss.backward()
            pending_samples += board_t.shape[0]
            pending_data_wait_s += data_wait_s
            pending_compute_s += time.perf_counter() - batch_compute_start
            pending_batches += 1
            pending_loss += loss.item() * accum_steps

            if pending_batches == accum_steps:
                step += 1
                current_lr = _cosine_lr(
                    step, total_steps, lr, warmup_steps=warmup_steps
                )
                for g in opt.param_groups:
                    g["lr"] = current_lr

                opt_start = time.perf_counter()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_time_s = time.perf_counter() - opt_start

                step_loss = pending_loss / pending_batches
                running_loss += step_loss
                running_steps += 1
                epoch_loss += step_loss
                epoch_steps += 1
                last_lr = current_lr
                last_grad_norm = float(grad_norm)
                step_samples = pending_samples
                step_data_wait_s = pending_data_wait_s
                step_compute_s = pending_compute_s + opt_time_s
                samples_per_sec = step_samples / max(
                    1e-6, step_data_wait_s + step_compute_s
                )

                pending_batches = 0
                pending_loss = 0.0
                pending_samples = 0
                pending_data_wait_s = 0.0
                pending_compute_s = 0.0
                if step <= 5 or step % log_every_steps == 0:
                    avg = running_loss / max(1, running_steps)
                    _log(
                        f"Epoch {epoch + 1}/{epochs} batch {i + 1} step {step} | "
                        f"samples_seen={samples_seen:,} | data_wait={step_data_wait_s:.2f}s | "
                        f"step_time={step_compute_s:.2f}s | samples_per_sec={samples_per_sec:.1f} | "
                        f"loss={step_loss:.4f} | avg_loss={avg:.4f} | "
                        f"lr={current_lr:.2e} | norm={grad_norm:.2f}"
                    )
                    running_loss = 0.0
                    running_steps = 0
                    if use_wandb and _WANDB:
                        wandb.log(
                            {
                                "loss": avg,
                                "lr": current_lr,
                                "grad_norm": grad_norm,
                                "samples_per_sec": samples_per_sec,
                                "data_wait_s": step_data_wait_s,
                                "step_time_s": step_compute_s,
                                "step": step,
                            }
                        )
            batch_wait_start = time.perf_counter()

        if pending_batches > 0:
            scale = accum_steps / pending_batches
            if scale != 1:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale)

            step += 1
            current_lr = _cosine_lr(step, total_steps, lr, warmup_steps=warmup_steps)
            for g in opt.param_groups:
                g["lr"] = current_lr

            opt_start = time.perf_counter()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_time_s = time.perf_counter() - opt_start

            step_loss = pending_loss / pending_batches
            epoch_loss += step_loss
            epoch_steps += 1
            last_lr = current_lr
            last_grad_norm = float(grad_norm)
            step_data_wait_s = pending_data_wait_s
            step_compute_s = pending_compute_s + opt_time_s
            samples_per_sec = pending_samples / max(
                1e-6, step_data_wait_s + step_compute_s
            )
            _log(
                f"Epoch {epoch + 1}/{epochs} batch {i + 1} step {step} | "
                f"samples_seen={samples_seen:,} | data_wait={step_data_wait_s:.2f}s | "
                f"step_time={step_compute_s:.2f}s | samples_per_sec={samples_per_sec:.1f} | "
                f"loss={step_loss:.4f} | lr={current_lr:.2e} | norm={grad_norm:.2f} "
                "(partial accumulation)"
            )
            if use_wandb and _WANDB:
                wandb.log(
                    {
                        "loss": step_loss,
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                        "samples_per_sec": samples_per_sec,
                        "data_wait_s": step_data_wait_s,
                        "step_time_s": step_compute_s,
                        "step": step,
                    }
                )

        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            epoch_s = time.perf_counter() - epoch_start
            avg_samples_per_sec = samples_seen / max(1e-6, epoch_s)
            _log(
                f"Epoch {epoch + 1}/{epochs} summary | optimizer_steps={epoch_steps} | "
                f"samples_seen={samples_seen:,} | avg_loss={avg_epoch_loss:.4f} | "
                f"avg_samples_per_sec={avg_samples_per_sec:.1f} | "
                f"lr={last_lr:.2e} | norm={last_grad_norm:.2f} | elapsed={epoch_s:.1f}s"
            )
        else:
            _log(f"Epoch {epoch + 1}/{epochs} summary | optimizer_steps=0 | no batches processed")

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
            _log(f"Saved checkpoint: {ckpt_path}")

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
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--min_depth", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=100.0)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument(
        "--resume_from", default=None, help="Distillation checkpoint to resume from"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument(
        "--forward_dtype",
        default=None,
        help="Forward dtype override: auto, float32, float16, or bfloat16",
    )
    args = parser.parse_args()

    train(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        min_depth=args.min_depth,
        temperature=args.temperature,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device_str=args.device,
        use_wandb=args.wandb,
        accum_steps=args.accum_steps,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        log_every_steps=args.log_every_steps,
        forward_dtype_str=args.forward_dtype,
    )


if __name__ == "__main__":
    main()
