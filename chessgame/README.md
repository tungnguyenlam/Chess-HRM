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
| 2 — Supervised | `chessgame/train/supervised.py` | Policy CE + Value MSE | Lichess Elite JSONL |
| 3 — Distillation | `chessgame/train/distill.py` | Soft policy KL + Value MSE | Stockfish multipv JSONL |
| 4 — RL (planned) | — | Soft policy + Value MSE + ACT Q-loss | Self-play with replay buffer |

## Quick Start

```bash
# Phase 2 — supervised pretraining from Lichess Elite data
python -m chessgame.train.supervised \
    --data path/to/lichess_elite.jsonl \
    --config mac_mini \
    --epochs 5

# Phase 3 — soft-label distillation from Stockfish multipv data
python -m chessgame.train.distill \
    --data path/to/stockfish_soft.jsonl \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --config mac_mini

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
