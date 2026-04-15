# chessgame/data

Dataset classes for all training phases and a circular replay buffer for self-play RL.

## Files

### `lichess_dataset.py`

PyTorch `Dataset` for Phase 2 supervised pretraining from the [Lichess Elite Database](https://database.nikonoel.fr/).

#### Class: `LichessEliteDataset(Dataset)`

Reads a JSONL file where each line is a JSON object with the following fields:

```jsonc
{"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "move": "e7e5", "cp": 15, "depth": 18}
```

| Field | Type | Description |
|-------|------|-------------|
| `fen` | `str` | Board position in FEN notation. |
| `move` | `str` | Best move in UCI notation. |
| `cp` | `int` | Centipawn evaluation (positive = white advantage). |
| `depth` | `int` | Stockfish search depth used to generate the move. |

Rows with `depth < min_depth` are filtered out. The value target is computed as `tanh(cp / 400)`.

**Constructor:**

```python
LichessEliteDataset(data_path: str, min_depth: int = 10)
```

**`__getitem__` returns:**

| Index | Shape | Description |
|-------|-------|-------------|
| `0` | `[8, 8, 119]` | Encoded board tensor. |
| `1` | `int` | Best move index in `[0, 4671]`. |
| `2` | `float` | Value target in `[-1, 1]`. |

#### Usage

```python
from torch.utils.data import DataLoader
from chessgame.data.lichess_dataset import LichessEliteDataset

dataset = LichessEliteDataset("data/lichess_elite.jsonl", min_depth=12)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

for boards, move_idxs, values in loader:
    # boards: [B, 8, 8, 119]
    # move_idxs: [B]
    # values: [B]
    ...
```

---

### `stockfish_dataset.py`

PyTorch `Dataset` for Phase 3 soft-label distillation from pre-generated Stockfish multipv data.

#### Class: `StockfishSoftDataset(Dataset)`

Reads a JSONL file where each line contains multiple candidate moves with their evaluations:

```jsonc
{"fen": "...", "moves": [{"move": "e2e4", "cp": 30}, {"move": "d2d4", "cp": 25}, ...], "depth": 20}
```

| Field | Type | Description |
|-------|------|-------------|
| `fen` | `str` | Board position in FEN notation. |
| `moves` | `list` | List of `{move, cp}` objects from Stockfish multipv analysis. |
| `depth` | `int` | Search depth. |

Builds a soft policy distribution over all 4672 moves by applying softmax to `cp / temperature`.

**Constructor:**

```python
StockfishSoftDataset(data_path: str, min_depth: int = 10, temperature: float = 50.0)
```

A higher `temperature` produces a more uniform distribution; lower values sharpen it toward the top move.

**`__getitem__` returns:**

| Index | Shape | Description |
|-------|-------|-------------|
| `0` | `[8, 8, 119]` | Encoded board tensor. |
| `1` | `[4672]` | Soft policy distribution (sums to 1). |
| `2` | `float` | Value target in `[-1, 1]`. |

#### Usage

```python
from chessgame.data.stockfish_dataset import StockfishSoftDataset

dataset = StockfishSoftDataset("data/stockfish_soft.jsonl", temperature=50.0)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

for boards, soft_policies, values in loader:
    # boards: [B, 8, 8, 119]
    # soft_policies: [B, 4672]
    # values: [B]
    ...
```

---

### `replay_buffer.py`

Fixed-capacity circular replay buffer for Phase 4 self-play RL.

#### Class: `ReplayBuffer`

Stores `(board_tensor, mcts_pi, outcome)` tuples in a circular buffer. Overwrites the oldest entries once full.

**Constructor:**

```python
ReplayBuffer(capacity: int = 100_000)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `(samples: list[tuple]) -> None` | Appends a list of `(board [8,8,119], mcts_pi [4672], outcome float)` tuples. Wraps around when capacity is reached. |
| `sample` | `(batch_size: int) -> tuple[Tensor, Tensor, Tensor]` | Returns randomly sampled `(boards, pis, outcomes)` as stacked tensors. |

#### Usage

```python
from chessgame.data.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(capacity=50_000)

# After a self-play game, add the collected samples
buffer.add([(board_t, pi_t, outcome) for board_t, pi_t, outcome in game_samples])

# Sample a training batch
boards, pis, outcomes = buffer.sample(batch_size=256)
# boards: [256, 8, 8, 119]
# pis:    [256, 4672]
# outcomes: [256]
```
