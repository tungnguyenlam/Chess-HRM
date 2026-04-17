#!/usr/bin/env python3
"""
Step 0: Pretrain on ARC puzzles (optional foundation training).

Usage:
    python scripts/s0_pretrain.py --data_path data/arc-puzzles

Default config from chess.yaml: uses mac_mini (7M params).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain import launch

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HRMChess Pretrain")
    parser.add_argument(
        "--data_path", type=str, default="data/arc-aug-1000", help="Path to dataset"
    )
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--config", type=str, default="mac_mini", choices=["full", "mac_mini"]
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")

    args, unknown = parser.parse_known_args()

    # Build hydra-style config
    from omegaconf import OmegaConf

    # Load default chess config
    chess_cfg = OmegaConf.load("config/chess.yaml")
    cfg_name = args.config

    base_cfg = OmegaConf.create(
        {
            "arch": {
                "name": "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
                "loss": {
                    "name": "losses@ACTLossHead",
                    "loss_type": "stablemax_cross_entropy",
                },
            },
            "data_path": args.data_path,
            "global_batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_min_ratio": 1.0,
            "lr_warmup_steps": 2000,
            "weight_decay": 0.1,
            "puzzle_emb_lr": 1e-2,
            "puzzle_emb_weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
        }
    )

    # Merge chess config (which may have model params)
    model_cfg = chess_cfg.get(cfg_name, chess_cfg.mac_mini)
    base_cfg.arch.update(model_cfg)

    # Override with args
    if args.checkpoint_path:
        base_cfg.checkpoint_path = args.checkpoint_path

    os.environ["WANDB_MODE"] = "offline" if not args.wandb else "online"

    # Override sys.argv for hydra
    sys.argv = ["pretrain.py"]
    if unknown:
        sys.argv.extend(unknown)

    # Run with config
    from pretrain import PretrainConfig

    config = PretrainConfig(**base_cfg)
    launch()
