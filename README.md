# HRM-GAB Chess

![](./assets/hrm.png)

**HRM-GAB** is a novel chess architecture combining:
1.  **Adaptive Computation Time (ACT)** from the Hierarchical Reasoning Model (HRM).
2.  **Geometric Attention Bias (GAB)** from Chessformer.
3.  **Recurrent GAB Evolution**: Geometric understanding deepens with each reasoning cycle.

---

## Quick Start (HRM-GAB Chess) 🚀

### 1. Setup Environment

```bash
# Recommended on Apple Silicon: build the training venv with Python 3.13
deactivate 2>/dev/null || true
bash scripts/setup_mac_env.sh --recreate
source .venv/bin/activate
```

If you prefer manual setup, use `python3` from your Homebrew path or shell PATH.
Python `3.13` is the recommended local target. The checked-in `.venv` may still
be older. If you are already inside an older virtualenv, deactivate it first or
set `PYTHON_BIN=python3.13` when running the setup script.

### 2. Verify Installation (Smoke Tests)

Run the full suite of unit tests to ensure encoders and model architecture are working correctly:

```bash
# Run all tests (skips Stockfish-dependent tests if not installed)
python -m pytest tests/ -v
```

### 3. Training & Evaluation Pipeline

#### Phase 1: Supervised Training (Lichess Elite)
Train on the Phase 1 shard directory generated under `data-lichess/phase1`.

```bash
# Test with dummy data (Smoke Test)
python tests/test_supervised_train.py

# Full supervised run from chess.yaml defaults
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 5 \
    --checkpoint_dir checkpoints/supervised

# Apple Silicon low-RAM run with MPS and early W&B metrics
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 1 \
    --device mps \
    --forward_dtype auto \
    --num_workers 0 \
    --wandb \
    --checkpoint_dir /tmp/hrm-gab-mps-smoke

# Conservative CPU/sandbox run
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 1 \
    --device cpu \
    --num_workers 0 \
    --checkpoint_dir /tmp/hrm-gab-supervised-smoke
```

#### Phase 2: Stockfish Distillation
Fine-tune using soft-labels from Stockfish MultiPV.

```bash
python scripts/s2_distill.py \
    --data data-lichess/phase2 \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --config mac_mini \
    --epochs 2 \
    --checkpoint_dir checkpoints/distill
```

#### Phase 3: Evaluation (Arena & Puzzles)
Measure the model's strength against Stockfish or its accuracy on tactical puzzles.

```bash
# Arena: Model vs Stockfish (Elo estimation)
python evaluate_chess.py \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --sf_elo 1500 \
    --games 20

# Puzzles: Lichess Puzzle Database
# Download CSV from: https://database.lichess.org/#puzzles
python -c "from chessgame.eval.puzzles import PuzzleEvaluator; from chessgame.model.hrm_chess import HRMChess; import torch; \
model = HRMChess.load_from_checkpoint('checkpoints/supervised/epoch_5.pt'); \
evaluator = PuzzleEvaluator(lambda b: model.get_move(b)); \
report = evaluator.evaluate_file('puzzles.csv', max_puzzles=1000); \
print(f'Accuracy: {report.accuracy:.2%}')"
```

---

## Project Structure

*   `chessgame/model/`: **HRM-GAB** architecture (GAB integration, Attention bias).
*   `chessgame/encoding/`: AlphaZero-style 119-plane board encoding.
*   `chessgame/data/`: Lichess and Stockfish dataset loaders.
*   `chessgame/train/`: Supervised, Distillation, and RL (planned) loops.
*   `chessmodels/hrm/`: Upstream HRM core (Read-only).

---

## Implementation Status

- [x] **Phase 0: Infrastructure** (Encoders, Datasets, SF Annotator, Eval)
- [x] **Phase 1: Architecture** (GAB Module, Attention Bias, HRM Integration)
- [x] **Phase 2: Supervised** (Training loop, Resumption logic, Smoke tests)
- [x] **Phase 3: Distillation** (Training loop, Generation script, Resumption logic)
- [x] **Phase 4: RL Self-Play** (MCTS, Self-Play loop)
- [ ] **Phase 5: Ablation Studies** (Planned)

## Local Performance Notes

- Apple Silicon runs should prefer `--device mps --forward_dtype auto`, which now resolves to stable `float32` on MPS.
- Local CPU and MPS runs default to `num_workers=0` to avoid the higher RAM cost
  and stability issues of spawned DataLoader workers on macOS.
- Training now prints `data_wait`, `step_time`, and `samples_per_sec` so a slow
  run is distinguishable from a real hang.

---

## Original HRM Documentation

(See below for the original Hierarchical Reasoning Model documentation for ARC, Sudoku, and Maze tasks.)

---

# Hierarchical Reasoning Model (Upstream)

... [REST OF ORIGINAL README] ...
