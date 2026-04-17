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
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pytest wandb omegaconf python-chess
```

### 2. Verify Installation (Smoke Tests)

Run the full suite of unit tests to ensure encoders and model architecture are working correctly:

```bash
# Run all tests (skips Stockfish-dependent tests if not installed)
python -m pytest tests/ -v
```

### 3. Training & Evaluation Pipeline

#### Phase 1: Supervised Training (Lichess Elite)
Train on a JSONL dataset of Lichess positions.

```bash
# Test with dummy data (Smoke Test)
python tests/test_supervised_train.py

# Full Supervised Run (Mac Mini config)
python -m chessgame.train.supervised \
    --data data/lichess_elite.jsonl \
    --config mac_mini \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints/supervised
```

#### Phase 2: Stockfish Distillation
Fine-tune using soft-labels from Stockfish MultiPV.

```bash
python -m chessgame.train.distill \
    --data data/stockfish_soft.jsonl \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --config mac_mini \
    --epochs 2 \
    --lr 3e-5 \
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

---

## Original HRM Documentation

(See below for the original Hierarchical Reasoning Model documentation for ARC, Sudoku, and Maze tasks.)

---

# Hierarchical Reasoning Model (Upstream)

... [REST OF ORIGINAL README] ...
