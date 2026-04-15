# chessgame/model

Defines the full `HRMChess` model: configuration presets, 2D rotary position embeddings, and the chess-specific inner/outer model classes that subclass the generic HRM-ACT core in `chessmodels/`.

## Files

### `hrm_chess_config.py`

Chess-specific configuration that extends `HierarchicalReasoningModel_ACTV1Config` with chess input/output dimensions.

#### Class: `HRMChessConfig`

Inherits from `HierarchicalReasoningModel_ACTV1Config`. Adds two fixed fields:

| Field | Value | Description |
|-------|-------|-------------|
| `board_input_dim` | `119` | Number of feature planes per square (AlphaZero encoding). |
| `policy_size` | `4672` | Number of possible move indices. |

#### Preset class methods

| Method | Params | Description |
|--------|--------|-------------|
| `HRMChessConfig.full()` | `hidden_size=512`, 4 H-layers, 4 L-layers, 3 H-cycles, 6 L-cycles, `halt_max_steps=8` | Server-scale config (~28M parameters). |
| `HRMChessConfig.mac_mini()` | `hidden_size=256`, 2 H-layers, 2 L-layers, 2 H-cycles, 4 L-cycles, `halt_max_steps=4` | Local dev/training config (~7M parameters). |

#### Usage

```python
from chessgame.model.hrm_chess_config import HRMChessConfig

config = HRMChessConfig.mac_mini()   # for local development
config = HRMChessConfig.full()       # for server training
```

---

### `rope_2d.py`

2D Rotary Position Embedding for the 8×8 chess board plus one CLS token (65 tokens total).

#### Class: `ChessRoPE2D(nn.Module)`

Pre-computes cos/sin buffers of shape `[65, head_dim]` by concatenating row-axis and column-axis RoPE. The CLS token gets zero-valued embeddings (no positional bias).

| Method | Returns | Description |
|--------|---------|-------------|
| `forward()` | `CosSin` (tuple of two `[65, head_dim]` tensors) | Returns the precomputed `(cos, sin)` buffers. |

---

### `hrm_chess.py`

The main chess model. `HRMChessInner` replaces the generic token embedding and LM head with chess-specific projections; `HRMChess` wraps it in the ACT outer loop.

#### Class: `HRMChessInner(HierarchicalReasoningModel_ACTV1_Inner)`

Subclasses the generic inner model and replaces its I/O layers with:

| Layer | Type | Description |
|-------|------|-------------|
| `board_proj` | `Linear(119 → hidden_size)` | Projects each of the 64 squares from 119 feature planes to the model's hidden dimension. |
| `rope_2d` | `ChessRoPE2D` | 2D rotary position embeddings for the 8×8 board. |
| `cls_token` | `nn.Parameter` | Learnable CLS token prepended to the 64-square sequence (index 0). |
| `policy_head` | `Linear(hidden_size → 4672)` | Outputs raw logits over all 4672 moves. |
| `value_head` | `MLP(hidden_size → 256 → 1 → tanh)` | Outputs a scalar game value in `[-1, 1]`. |

`forward(carry, batch)` runs the H/L reasoning cycles and returns `(new_carry, cls_hidden, (q_halt_logits, q_continue_logits))`.

#### Class: `HRMChess(nn.Module)`

Outer ACT wrapper that owns `HRMChessInner` and manages the halting loop.

| Method | Signature | Description |
|--------|-----------|-------------|
| `initial_carry` | `(batch_size: int) -> HierarchicalReasoningModel_ACTV1Carry` | Creates zeroed carry state for a new batch. |
| `forward` | `(carry, batch: dict) -> (new_carry, outputs)` | Resets halted sequences, runs the inner model, applies chess heads, executes ACT halting logic, returns updated carry and output dict. |

**`batch` dict keys:**

| Key | Shape | Description |
|-----|-------|-------------|
| `"board"` | `[B, 8, 8, 119]` | Encoded board tensors from `encode_board`. |

**`outputs` dict keys:**

| Key | Shape | Description |
|-----|-------|-------------|
| `"policy"` | `[B, 4672]` | Raw move logits (apply `legal_mask` before argmax). |
| `"value"` | `[B, 1]` | Predicted game value in `[-1, 1]`. |
| `"q_halt_logits"` | `[B, 1]` | Q-value logits for the halt decision. |
| `"q_continue_logits"` | `[B, 1]` | Q-value logits for the continue decision. |

## Usage

```python
import chess
import torch
from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.encoding.board_encoder import encode_board
from chessgame.encoding.move_encoder import legal_mask

device = torch.device("cpu")  # or "mps" / "cuda"
config = HRMChessConfig.mac_mini()
model = HRMChess(config).to(device)

board = chess.Board()
board_tensor = encode_board(board, history=[])          # [8, 8, 119]
board_tensor = board_tensor.unsqueeze(0).to(device)    # [1, 8, 8, 119]

carry = model.initial_carry(batch_size=1)
batch = {"board": board_tensor}

with torch.no_grad():
    carry, outputs = model(carry, batch)

policy_logits = outputs["policy"][0]                   # [4672]
mask = legal_mask(board).to(device)
policy_logits[~mask] = float("-inf")
best_move_idx = policy_logits.argmax().item()

value = outputs["value"][0].item()                     # float in [-1, 1]
```

## Loading a checkpoint

```python
checkpoint = torch.load("checkpoints/distill/epoch_5.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
```
