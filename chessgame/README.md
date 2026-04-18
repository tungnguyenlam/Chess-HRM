# chessgame

Chess-specific adapter package that wraps the generic HRM-ACT core (`chessmodels/`) into a full chess engine. It handles board/move encoding, dataset loading, model definition, and training across all supervised and distillation phases.

## Package Structure

```
chessgame/
├── encoding/   Board and move encoding (AlphaZero-style)
├── model/      HRMChess model and configuration
├── data/       Dataset classes and replay buffer
└── train/      Loss functions and training scripts
```

## Training Pipeline

The chess engine is trained in phases:

| Phase | Script | Loss | Data source |
|-------|--------|------|-------------|
| 1 — Generic pretraining | `pretrain.py` (root) | Stablemax CE + Q-halt + Q-continue | ARC / maze / sudoku puzzles |
| 2 — Supervised | `scripts/s1_supervised.py` | Policy CE + Value MSE | `data-lichess/phase1` shards |
| 3 — Distillation | `scripts/s2_distill.py` | Soft policy KL + Value MSE | `data-lichess/phase2` labels |
| 4 — RL (planned) | — | Soft policy + Value MSE + ACT Q-loss | Self-play with replay buffer |

## Quick Start

```bash
# Local Apple Silicon environment
bash scripts/setup_mac_env.sh --recreate

# Phase 2 — supervised pretraining from Lichess Elite data
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 5

# Phase 3 — soft-label distillation from Stockfish multipv data
python scripts/s2_distill.py \
    --data data-lichess/phase2 \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --config mac_mini \
    --epochs 2

# CPU/sandbox smoke run
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 1 \
    --device cpu \
    --num_workers 0

# Apple Silicon MPS smoke run
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 1 \
    --device mps \
    --forward_dtype auto \
    --num_workers 0 \
    --wandb

# Evaluate against Stockfish
python evaluate_chess.py \
    --checkpoint checkpoints/distill/epoch_5.pt \
    --sf_elo 1500 \
    --games 20
```

## Subfolders

- [`encoding/`](encoding/README.md) — `encode_board`, `encode_move`, `decode_move`, `legal_mask`
- [`model/`](model/README.md) — `HRMChess`, `HRMChessConfig`, `ChessRoPE2D`
- [`data/`](data/README.md) — `LichessEliteDataset`, `StockfishSoftDataset`, `ReplayBuffer`
- [`train/`](train/README.md) — loss functions, `supervised.py`, `distill.py`

## Runtime Defaults

- `forward_dtype=auto` resolves to stable `float32` on MPS, config-preferred low precision on CUDA, and `float32` on CPU.
- CPU and MPS default to `num_workers=0` because extra macOS DataLoader workers can increase RAM usage and trigger shared-memory failures.
