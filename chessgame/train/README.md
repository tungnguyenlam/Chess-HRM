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
- Per-epoch checkpointing to `checkpoint_dir/`
- Optional Weights & Biases logging (`--wandb`)

#### CLI

```bash
python -m chessgame.train.supervised \
    --data path/to/lichess_elite.jsonl \
    --config mac_mini \
    --epochs 5 \
    --batch_size 256 \
    --lr 3e-4 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --min_depth 12 \
    --checkpoint_dir checkpoints/supervised \
    --device mps
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to Lichess Elite JSONL file. |
| `--config` | `mac_mini` | Config preset: `mac_mini` or `full`. |
| `--epochs` | `5` | Number of training epochs. |
| `--batch_size` | `256` | Per-step batch size. |
| `--lr` | `3e-4` | Peak learning rate. |
| `--weight_decay` | `0.01` | AdamW weight decay. |
| `--grad_clip` | `1.0` | Gradient norm clipping threshold. |
| `--min_depth` | `10` | Minimum Stockfish depth to include a position. |
| `--checkpoint_dir` | `checkpoints/supervised` | Directory to save epoch checkpoints. |
| `--device` | `cpu` | Device string: `cpu`, `mps`, or `cuda`. |
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
- Per-epoch checkpointing

#### CLI

```bash
python -m chessgame.train.distill \
    --data path/to/stockfish_soft.jsonl \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --config mac_mini \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --temperature 50.0 \
    --checkpoint_dir checkpoints/distill \
    --device mps
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to Stockfish multipv JSONL file. |
| `--checkpoint` | `None` | Path to a supervised checkpoint to resume from. |
| `--config` | `mac_mini` | Config preset: `mac_mini` or `full`. |
| `--epochs` | `5` | Number of training epochs. |
| `--batch_size` | `128` | Per-step batch size. |
| `--lr` | `1e-4` | Peak learning rate. |
| `--weight_decay` | `0.01` | AdamW weight decay. |
| `--grad_clip` | `1.0` | Gradient norm clipping threshold. |
| `--min_depth` | `10` | Minimum Stockfish depth to include a position. |
| `--temperature` | `50.0` | Softmax temperature for soft policy targets. |
| `--checkpoint_dir` | `checkpoints/distill` | Directory to save epoch checkpoints. |
| `--device` | `cpu` | Device string: `cpu`, `mps`, or `cuda`. |
| `--wandb` | off | Enable Weights & Biases logging. |

## Typical training sequence

```bash
# 1. Phase 2 — supervised from Lichess data
python -m chessgame.train.supervised \
    --data data/lichess_elite.jsonl \
    --config mac_mini --epochs 5

# 2. Phase 3 — distill from Stockfish soft labels
python -m chessgame.train.distill \
    --data data/stockfish_soft.jsonl \
    --checkpoint checkpoints/supervised/epoch_5.pt \
    --config mac_mini --epochs 5

# 3. Evaluate against Stockfish
python evaluate_chess.py \
    --checkpoint checkpoints/distill/epoch_5.pt \
    --sf_elo 1500 --games 20
```
