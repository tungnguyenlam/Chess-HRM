# chessgame/encoding

Converts between `python-chess` objects and the tensor representations consumed by `HRMChess`. Implements the AlphaZero-style board encoding and the 4672-move index scheme.

## Files

### `board_encoder.py`

Encodes a chess board position into a float32 tensor of shape `[8, 8, 119]`.

#### Function

| Function | Signature | Description |
|----------|-----------|-------------|
| `encode_board` | `(board: chess.Board, history: list[chess.Board]) -> Tensor` | Builds 119 feature planes from the current board and up to 7 prior positions. |

**Plane layout (119 total):**

- **Planes 0–111** (14 planes × 8 time-steps): For each time-step `t`, 6 planes for white piece types (P, N, B, R, Q, K), 6 planes for black piece types, and 2 repetition planes.
- **Planes 112–118** (7 auxiliary planes): side to move, total move count, castling rights (K/Q/k/q), halfmove clock.

#### Usage

```python
import chess
from chessgame.encoding.board_encoder import encode_board

board = chess.Board()
history = []  # list of previous chess.Board positions (up to 7)

tensor = encode_board(board, history)
# tensor.shape == (8, 8, 119), dtype=torch.float32
```

---

### `move_encoder.py`

Maps `chess.Move` objects to integer indices in `[0, 4671]` and back. Follows the AlphaZero 4672-move encoding.

#### Constants

| Name | Value | Description |
|------|-------|-------------|
| `NUM_MOVES` | `4672` | Total number of possible move indices. |

#### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `encode_move` | `(move: chess.Move) -> int` | Encodes a move to an integer index. Handles queen-type moves (0–55 per square), knight moves (56–63), and underpromotions (64–72). |
| `decode_move` | `(idx: int, board: chess.Board) -> chess.Move` | Decodes an integer index back to a `chess.Move`. Auto-detects queen promotion from board context. |
| `legal_mask` | `(board: chess.Board) -> Tensor` | Returns a `[4672]` bool tensor with `True` at every legal move index for the current position. |

#### Usage

```python
import chess
from chessgame.encoding.move_encoder import encode_move, decode_move, legal_mask, NUM_MOVES

board = chess.Board()
move = chess.Move.from_uci("e2e4")

idx = encode_move(move)           # int in [0, 4671]
recovered = decode_move(idx, board)  # chess.Move

mask = legal_mask(board)          # torch.BoolTensor, shape [4672]
# Use mask to zero out illegal moves before argmax:
# logits[~mask] = -inf
```

#### Move type categories (per source square)

| Category | Plane offset | Description |
|----------|-------------|-------------|
| Queen-type | 0–55 | 8 directions × 7 distances |
| Knight | 56–63 | 8 knight deltas |
| Underpromotion | 64–72 | 3 pieces (N/B/R) × 3 directions |
