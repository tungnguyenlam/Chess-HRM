# chessgame/train

Loss functions and training loop scripts for Phase 2 (supervised) and Phase 3 (distillation) of the chess engine training pipeline.

## Files

### `loss.py`

Pure loss functions with no training loop logic. All functions accept model `outputs` dicts or raw tensors and return scalar loss values.

#### Functions

| Function | Signature | Phase | Description |
|----------|-----------|-------|-------------|
| `policy_hard` | `(logits: Tensor, target_idx: Tensor) -> Tensor` | 2 | Cross-entropy loss against a single best-move index. |
| `policy_soft` | `(logits: Tensor, soft_target: Tensor) -> Tensor` | 3, 4 | KL-divergence-equivalent soft cross-entropy against a move distribution. |
| `value_mse` | `(pred: Tensor, target: Tensor) -> Tensor` | 2, 3, 4 | MSE between predicted and target game value. |
| `act_q_loss` | `(outputs: dict) -> Tensor` | 4 | ACT Q-learning loss: BCE on bootstrapped Q-continue target. |
| `total_supervised` | `(outputs: dict, move_target: Tensor, value_target: Tensor) -> Tensor` | 2 | `policy_hard + value_mse`. |
| `total_distill` | `(outputs: dict, soft_policy: Tensor, value_target: Tensor) -> Tensor` | 3 | `policy_soft + value_mse`. |
| `total_rl` | `(outputs, mcts_pi, outcome, act_weight, kl_sf_loss, kl_weight) -> Tensor` | 4 | `policy_soft + value_mse + act_weight * act_q_loss + kl_weight * kl_sf_loss`. |

#### Usage

```python
from chessgame.train.loss import total_supervised, total_distill

# Phase 2
loss = total_supervised(outputs, move_target=move_idxs, value_target=values)

# Phase 3
loss = total_distill(outputs, soft_policy=soft_policies, value_target=values)

loss.backward()
```

---

### `supervised.py`

Phase 2 training loop: supervised pretraining from Lichess Elite data.

#### Key features

- Cosine LR schedule with linear warmup
- Gradient accumulation (`--accum_steps`)
- CPU/MPS-safe worker defaults with optional `--num_workers` override
- Auto dtype resolution with optional `--forward_dtype` override
- Per-epoch checkpointing to `checkpoint_dir/`
- Optional Weights & Biases logging (`--wandb`)

#### CLI

```bash
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 5 \
    --checkpoint_dir checkpoints/supervised

# Conservative CPU/sandbox run
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 1 \
    --device cpu \
    --num_workers 0 \
    --checkpoint_dir /tmp/hrm-gab-supervised-smoke

# Apple Silicon MPS run
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

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to the Phase 1 shard directory or a single Lichess JSONL file. |
| `--config` | `mac_mini` | Config preset: `mac_mini` or `full`. |
| `--epochs` | `5` | Number of training epochs. |
| `--batch_size` | config value | Per-step batch size. |
| `--lr` | config value | Peak learning rate. |
| `--weight_decay` | config value | AdamW weight decay. |
| `--grad_clip` | config value | Gradient norm clipping threshold. |
| `--min_depth` | wrapper default | Minimum position depth filter. |
| `--checkpoint_dir` | `checkpoints/supervised` | Directory to save epoch checkpoints. |
| `--device` | `auto` | Device string: `cpu`, `mps`, `cuda`, or `auto`. |
| `--forward_dtype` | config value | Forward dtype: `auto`, `float32`, `float16`, or `bfloat16`. |
| `--num_workers` | device-aware | DataLoader workers; defaults to `0` on CPU/MPS and `4` on CUDA. |
| `--wandb` | off | Enable Weights & Biases logging. |
| `--accum_steps` | `1` | Gradient accumulation steps. |

#### Checkpoint format

Each epoch saves `epoch_{n}.pt` containing:

```python
{"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": n, "step": step}
```

---

### `distill.py`

Phase 3 training loop: soft-label distillation from pre-generated Stockfish multipv data.

#### Key features

- Loads from a Phase 2 supervised checkpoint (`--checkpoint`)
- Soft policy KL-divergence loss via `total_distill`
- Cosine LR schedule with 200-step warmup
- CPU/MPS-safe worker defaults with optional `--num_workers` override
- Auto dtype resolution with optional `--forward_dtype` override
- Per-epoch checkpointing

#### CLI

```bash
python scripts/s2_distill.py \
    --data data-lichess/phase2 \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --config mac_mini \
    --epochs 2 \
    --checkpoint_dir checkpoints/distill
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to the Phase 2 labels file used for distillation. |
| `--checkpoint` | `None` | Path to a supervised checkpoint to resume from. |
| `--config` | `mac_mini` | Config preset: `mac_mini` or `full`. |
| `--epochs` | config value | Number of training epochs. |
| `--batch_size` | config value | Per-step batch size. |
| `--lr` | config value | Peak learning rate. |
| `--weight_decay` | config value | AdamW weight decay. |
| `--grad_clip` | config value | Gradient norm clipping threshold. |
| `--min_depth` | config value | Minimum Stockfish depth to include a position. |
| `--temperature` | config value | Softmax temperature for soft policy targets. |
| `--checkpoint_dir` | `checkpoints/distill` | Directory to save epoch checkpoints. |
| `--device` | `auto` | Device string: `cpu`, `mps`, `cuda`, or `auto`. |
| `--forward_dtype` | config value | Forward dtype: `auto`, `float32`, `float16`, or `bfloat16`. |
| `--num_workers` | device-aware | DataLoader workers; defaults to `0` on CPU/MPS and `4` on CUDA. |
| `--wandb` | off | Enable Weights & Biases logging. |

## Typical training sequence

```bash
# 1. Phase 2 — supervised from Lichess data
python scripts/s1_supervised.py \
    --data data-lichess/phase1 \
    --config mac_mini \
    --epochs 5

# 2. Phase 3 — distill from Stockfish soft labels
python scripts/s2_distill.py \
    --data data-lichess/phase2 \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --config mac_mini \
    --epochs 2

# 3. Evaluate against Stockfish
python evaluate_chess.py \
    --checkpoint checkpoints/distill/epoch_5.pt \
    --sf_elo 1500 --games 20
```
