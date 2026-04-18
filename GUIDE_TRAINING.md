# Training Guide — HRM-GAB Chess

This guide details how to execute the training pipeline for the HRM-GAB chess model, utilizing the highly efficient memory-mapped dataset shards we generated from the Lichess Elite database.

## Prerequisites

Ensure you have completed Phase 0 data extraction. You should have two directories containing `.jsonl.zst` shards:
- `data-lichess/phase1` (Supervised training dataset, ~5M positions)
- `data-lichess/phase2` (Distillation dataset, ~2M positions)

Activate your virtual environment before proceeding:
```bash
deactivate 2>/dev/null || true
bash scripts/setup_mac_env.sh --recreate
source .venv/bin/activate
```

For local Apple Silicon training, prefer Python `3.13`, `--device mps`, and
`--forward_dtype auto`. Keep `--num_workers 0` unless you have confirmed that
extra workers help on your machine. If you are already inside an older `.venv`,
deactivate it first or run the setup script with `PYTHON_BIN=python3.13`.

---

## Phase 1: Supervised Pretraining

In this phase, the model learns human intuition and the rules of chess by predicting moves played by high-Elo players.

### Standard Training

Run the config-aware wrapper pointing to the Phase 1 shard directory:

```bash
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 5 \
    --min_elo 1800 \
    --checkpoint_dir checkpoints/supervised
```

### Apple Silicon MPS Run

This is the recommended low-RAM starting point on the M4 Mac Mini. It keeps the
loader single-process, uses MPS, and lets the runtime resolve to stable
`float32`. If you want to experiment with `float16` later, treat it as opt-in.

```bash
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 1 \
    --device mps \
    --forward_dtype auto \
    --num_workers 0 \
    --wandb \
    --checkpoint_dir /tmp/hrm-gab-mps-smoke
```

### Curriculum Learning (Recommended)

To gradually expose the model to harder, higher-quality games, enable curriculum learning. This increases the minimum Elo threshold by 150 points every epoch, dynamically skipping lower-Elo games in the dataset stream.

```bash
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 5 \
    --min_elo 1800 \
    --curriculum \
    --checkpoint_dir checkpoints/supervised
```

*(Note: Checkpoints are saved automatically at the end of each epoch in `checkpoints/supervised/`.)*

---

### Conservative CPU Run

For CI, sandboxed environments, or any CPU-only smoke run, force a single-process
DataLoader. This avoids the shared-memory worker path that can fail on restricted
systems.

```bash
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 1 \
    --device cpu \
    --num_workers 0 \
    --checkpoint_dir /tmp/hrm-gab-supervised-smoke
```

---

## Phase 2: Stockfish Distillation

Once the model has learned basic play from human data, we refine its tactical capabilities by training against high-depth Stockfish annotations.

```bash
python scripts/s2_distill.py \
    --data data-lichess/phase2 \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --config mac_mini \
    --epochs 2 \
    --checkpoint_dir checkpoints/distill
```

---

## Resuming Training

If a training run is interrupted, you can resume seamlessly from the last saved checkpoint. The script will automatically restore the model weights, optimizer state, epoch, and step count.

```bash
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --checkpoint checkpoints/supervised/epoch_2.pt \
    --config mac_mini \
    --checkpoint_dir checkpoints/supervised
```

## Performance & Monitoring

- **W&B Integration:** Use the `--wandb` flag to log loss curves, learning rates, gradient norms, `data_wait`, `step_time`, and `samples_per_sec`.
- **Hardware Acceleration:** The training scripts automatically detect and use Apple Metal Performance Shaders (MPS) or NVIDIA CUDA if available.
- **Forward Dtype:** `--forward_dtype auto` resolves to stable `float32` on MPS, the config-preferred low-precision path on CUDA, and `float32` on CPU. Use `--forward_dtype float16` on MPS only as an explicit experiment.
- **Worker Count:** `num_workers > 0` on macOS uses spawned worker processes. That raises memory pressure and can fail under shared-memory restrictions, so the local default remains `0`.
