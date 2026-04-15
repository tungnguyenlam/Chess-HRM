# chessmodels/hrm

Generic Hierarchical Reasoning Model with Adaptive Computation Time (HRM-ACT V1). This is the chess-adapted copy of `models/hrm/hrm_act_v1.py` — functionally identical but imports from `chessmodels.*` for MPS/CPU compatibility. `HRMChessInner` in `chessgame/model/hrm_chess.py` subclasses `HierarchicalReasoningModel_ACTV1_Inner` from this file.

## File

### `hrm_act_v1.py`

Implements the full two-level hierarchical reasoning architecture with Q-learning-based adaptive halting.

---

## Architecture Overview

```
Input tokens
     │
     ▼
 embed_tokens
     │
     ▼
┌─────────────────────────────────────────────┐
│  ACT loop (up to halt_max_steps iterations) │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │  H-level reasoning stack            │   │
│  │  (h_layers blocks × h_cycles times) │   │
│  │  Input injection: z_H += x_emb      │   │
│  └──────────────┬───────────────────────┘   │
│                 │ z_H                        │
│  ┌──────────────▼───────────────────────┐   │
│  │  L-level reasoning stack            │   │
│  │  (l_layers blocks × l_cycles times) │   │
│  │  Input injection: z_L += z_H        │   │
│  └──────────────┬───────────────────────┘   │
│                 │ z_L                        │
│  Q-head: halt? continue?                    │
│  Halt if Q(halt) > Q(continue)              │
└─────────────────────────────────────────────┘
     │
     ▼
  lm_head → logits
```

The **1-step gradient trick**: all ACT iterations except the last run under `torch.no_grad()`. Only the final step's activations carry gradients, reducing memory cost while preserving learning signal.

---

## Data Classes

### `HierarchicalReasoningModel_ACTV1InnerCarry`

Holds the recurrent hidden state for one ACT step.

| Field | Shape | Description |
|-------|-------|-------------|
| `z_H` | `[B, seq_len, hidden_size]` | High-level reasoning hidden state. |
| `z_L` | `[B, seq_len, hidden_size]` | Low-level reasoning hidden state. |

### `HierarchicalReasoningModel_ACTV1Carry`

Full outer carry that tracks per-sequence halting state across ACT steps.

| Field | Shape | Description |
|-------|-------|-------------|
| `inner_carry` | `HierarchicalReasoningModel_ACTV1InnerCarry` | Current H/L hidden states. |
| `steps` | `[B]` | Number of ACT steps taken so far per sequence. |
| `halted` | `[B]` | Boolean mask: `True` if the sequence has halted. |
| `current_data` | `[B, seq_len, hidden_size]` | The most recent output (frozen for halted sequences). |

---

## Classes

### `HierarchicalReasoningModel_ACTV1Config(BaseModel)`

Pydantic config for the full model. Key fields:

| Field | Description |
|-------|-------------|
| `hidden_size` | Transformer hidden dimension. |
| `h_layers` | Number of transformer blocks in the H-level stack. |
| `l_layers` | Number of transformer blocks in the L-level stack. |
| `h_cycles` | Number of times the H-level stack is applied per ACT step. |
| `l_cycles` | Number of times the L-level stack is applied per ACT step. |
| `num_heads` | Number of attention heads. |
| `halt_max_steps` | Maximum number of ACT iterations before forced halt. |
| `vocab_size` | Vocabulary size (for the generic LM head). |
| `max_seq_len` | Maximum sequence length. |

---

### `HierarchicalReasoningModel_ACTV1Block(nn.Module)`

Single transformer block: **post-norm** self-attention followed by a SwiGLU MLP.

```
x → Attention(RMSNorm(x)) + x → SwiGLU(RMSNorm(x)) + x
```

---

### `HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module)`

A stack of `HierarchicalReasoningModel_ACTV1Block` layers with **additive input injection**. At each cycle, the original input embedding is added back to the hidden state before processing:

```
z = z + x_input   # inject input
z = stack_of_blocks(z)
```

---

### `HierarchicalReasoningModel_ACTV1_Inner(nn.Module)`

The inner model that executes one ACT step. Subclassed by `HRMChessInner`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `empty_carry` | `(batch_size, seq_len, device) -> InnerCarry` | Creates zeroed `z_H` and `z_L` tensors. |
| `reset_carry` | `(carry, mask) -> InnerCarry` | Zeros out carry for sequences indicated by `mask` (used to reset halted sequences). |
| `forward` | `(carry, batch) -> (new_carry, output, q_logits)` | Runs H-level then L-level stacks for one ACT step. Returns updated carry, output hidden states, and `(q_halt_logits, q_continue_logits)`. |

**1-step gradient trick:** The inner model runs all but the last iteration under `torch.no_grad()`. The final iteration runs normally so gradients flow back through exactly one step.

---

### `HierarchicalReasoningModel_ACTV1(nn.Module)`

Outer ACT wrapper. Manages the carry lifecycle and halting logic.

| Method | Signature | Description |
|--------|-----------|-------------|
| `initial_carry` | `(batch_size) -> Carry` | Creates a fresh zeroed carry for a new batch. |
| `forward` | `(carry, batch) -> (new_carry, outputs)` | Resets halted sequences, runs `_Inner` for one step, applies halting logic, returns updated carry and output dict. |

**Halting logic:**
- During training: sequences halt stochastically with probability `sigmoid(q_halt - q_continue)` plus an exploration epsilon.
- During inference: sequences halt deterministically when `q_halt > q_continue`.
- All sequences are forced to halt at `halt_max_steps`.
- Halted sequences keep their `current_data` frozen; only active sequences are updated.

## Usage

The inner model is not used directly — it is subclassed by `HRMChessInner`. To use the outer wrapper generically:

```python
from chessmodels.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)

config = HierarchicalReasoningModel_ACTV1Config(
    hidden_size=256,
    h_layers=2,
    l_layers=2,
    h_cycles=2,
    l_cycles=4,
    num_heads=8,
    halt_max_steps=4,
    vocab_size=512,
    max_seq_len=128,
)

model = HierarchicalReasoningModel_ACTV1(config)
carry = model.initial_carry(batch_size=4)

batch = {"input_ids": token_ids}  # [B, seq_len]
carry, outputs = model(carry, batch)
# outputs["logits"]: [B, seq_len, vocab_size]
```

For the chess-specific usage, see [`chessgame/model/README.md`](../../chessgame/model/README.md).
