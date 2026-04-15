# chessmodels

Chess-adapted copy of the generic `models/` package. Contains the core neural network building blocks (layers, losses, sparse embeddings) used by `HRMChess`. The key difference from `models/` is that attention has a **MPS/CPU fallback** — it uses `torch.nn.functional.scaled_dot_product_attention` when FlashAttention is unavailable, making it compatible with Apple Silicon and CPU-only environments.

## Relationship to `models/`

| | `chessmodels/` | `models/` |
|---|---|---|
| Attention backend | `F.scaled_dot_product_attention` (MPS/CPU safe) | Requires FlashAttention (GPU only) |
| Used by | `chessgame/` (chess engine) | `pretrain.py` (generic puzzle pretraining) |

## Files

### `common.py`

Mathematically correct truncated normal weight initialization, compatible with JAX/Flax conventions.

#### Function

| Function | Signature | Description |
|----------|-----------|-------------|
| `trunc_normal_init_` | `(tensor, std, lower, upper) -> None` | In-place truncated normal initialization via inverse CDF method. Used as the default weight init throughout the model. |

---

### `layers.py`

Core transformer building blocks. All linear layers use `CastedLinear` (weights cast to input dtype at forward time) and are initialized with `trunc_normal_init_`.

#### Type alias

`CosSin = Tuple[torch.Tensor, torch.Tensor]` — used as the return type of rotary embedding modules.

#### Classes

| Class | Description |
|-------|-------------|
| `CastedLinear(nn.Module)` | Linear layer that casts its weights to the input tensor's dtype at forward time. Enables mixed-precision training without explicit dtype management. |
| `CastedEmbedding(nn.Module)` | Embedding layer that casts looked-up vectors to a target dtype. |
| `RotaryEmbedding(nn.Module)` | Standard 1D Rotary Position Embedding (RoPE). Pre-computes `(cos, sin)` buffers for a given `max_seq_len` and `head_dim`. |
| `Attention(nn.Module)` | Multi-head self-attention with optional RoPE. Uses FlashAttention if available, otherwise falls back to `F.scaled_dot_product_attention` (supports MPS and CPU). |
| `SwiGLU(nn.Module)` | SwiGLU feed-forward network. Intermediate size is rounded up to the nearest multiple of 256. |

#### Free functions

| Function | Description |
|----------|-------------|
| `rotate_half(x)` | Rotates the last dimension by half for RoPE application. |
| `apply_rotary_pos_emb(q, k, cos, sin)` | Applies rotary position embeddings to query and key tensors. |
| `rms_norm(hidden_states, variance_epsilon)` | Root-mean-square layer normalization without learnable scale. |

---

### `losses.py`

Loss head that wraps an HRM model for puzzle-type tasks. Computes the combined LM + Q-halt + Q-continue loss and tracks evaluation metrics.

#### Constants

| Name | Value | Description |
|------|-------|-------------|
| `IGNORE_LABEL_ID` | `-100` | Label value masked out of loss computation (same convention as PyTorch CE). |

#### Class: `ACTLossHead(nn.Module)`

Wraps any HRM model. On `forward()`:
1. Runs the model for one ACT step.
2. Computes `lm_loss` using stablemax or softmax cross-entropy.
3. Computes `q_halt_loss` (BCE against the halt target).
4. Optionally computes `q_continue_loss` (bootstrapped Q-continue target).
5. Returns `(new_carry, total_loss, metrics_dict, detached_outputs, all_finished)`.

#### Stablemax utilities

| Function | Description |
|----------|-------------|
| `s(x, epsilon)` | Numerically stable stablemax activation. |
| `log_stablemax(x, dim)` | Log of the stablemax distribution. |
| `stablemax_cross_entropy(logits, labels, ignore_index)` | Stablemax-based cross-entropy (numerically more stable than softmax CE for sparse targets). |
| `softmax_cross_entropy(logits, labels, ignore_index)` | Standard softmax cross-entropy. |

---

### `sparse_embedding.py`

Sparse puzzle-ID embeddings with a custom distributed SignSGD optimizer. Used during generic pretraining (`pretrain.py`) to give each puzzle a learnable identity vector.

#### Class: `CastedSparseEmbedding(nn.Module)`

Maintains a persistent `weights` buffer (full embedding table) and non-persistent `local_weights` / `local_ids` buffers (the currently looked-up rows). During training, the looked-up rows are copied to `local_weights` which has a gradient; at inference `weights` is used directly.

**Constructor:** `CastedSparseEmbedding(num_embeddings, embedding_dim, dtype)`

#### Class: `CastedSparseEmbeddingSignSGD_Distributed(Optimizer)`

Custom optimizer for `CastedSparseEmbedding` parameters in multi-GPU training:
1. All-gathers gradients across all ranks.
2. Deduplicates embedding IDs.
3. Applies SignSGD with decoupled weight decay directly to the sparse slice.

**Constructor:** `CastedSparseEmbeddingSignSGD_Distributed(params, lr, weight_decay, world_size)`

## Subfolder

- [`hrm/`](hrm/README.md) — HRM-ACT V1 model: hierarchical reasoning with adaptive computation time.
