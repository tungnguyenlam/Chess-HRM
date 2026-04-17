"""Tests for chessgame.data.* — PLAN step 0.5.

Tests dataset loaders and replay buffer without requiring actual data files.
We create temporary JSONL fixtures inline.
"""

import json

import numpy as np
import torch
import pytest

from chessgame.data.lichess_dataset import LichessEliteDataset
from chessgame.data.stockfish_dataset import StockfishSoftDataset
from chessgame.data.replay_buffer import ReplayBuffer
from chessgame.encoding.move_encoder import NUM_MOVES


# ---------------------------------------------------------------------------
# Fixtures: in-memory JSONL files
# ---------------------------------------------------------------------------


@pytest.fixture
def lichess_jsonl(tmp_path):
    """Create a small Lichess-format JSONL file."""
    records = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "move": "e7e5",
            "cp": -30,
            "depth": 20,
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "move": "e2e4",
            "cp": 30,
            "depth": 22,
        },
        {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            "move": "b8c6",
            "cp": -10,
            "depth": 25,
        },
        # Shallow record — should be filtered out with min_depth=18
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "move": "d2d4",
            "cp": 20,
            "depth": 5,
        },
    ]
    path = tmp_path / "test_lichess.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


@pytest.fixture
def stockfish_jsonl(tmp_path):
    """Create a small Stockfish MultiPV JSONL file."""
    records = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "moves": [
                {"move": "e7e5", "cp": -30},
                {"move": "c7c5", "cp": -50},
                {"move": "d7d5", "cp": -60},
            ],
            "depth": 18,
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": [
                {"move": "e2e4", "cp": 30},
                {"move": "d2d4", "cp": 25},
            ],
            "depth": 20,
        },
        # Shallow — filtered at min_depth=15
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": [{"move": "e2e4", "cp": 10}],
            "depth": 10,
        },
    ]
    path = tmp_path / "test_sf.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# LichessEliteDataset
# ---------------------------------------------------------------------------


class TestLichessEliteDataset:
    """Tests for the Lichess supervised dataset."""

    def test_loads_records(self, lichess_jsonl):
        ds = LichessEliteDataset(lichess_jsonl, min_depth=18)
        # 3 records with depth >= 18, 1 filtered
        assert len(ds) == 3

    def test_min_depth_filter(self, lichess_jsonl):
        ds = LichessEliteDataset(lichess_jsonl, min_depth=25)
        assert len(ds) == 1  # only depth=25 record passes

    def test_max_records(self, lichess_jsonl):
        ds = LichessEliteDataset(lichess_jsonl, min_depth=18, max_records=2)
        assert len(ds) == 2

    def test_item_shapes(self, lichess_jsonl):
        ds = LichessEliteDataset(lichess_jsonl, min_depth=18)
        board_t, move_idx, value_t = ds[0]
        assert board_t.shape == (8, 8, 119)
        assert board_t.dtype == torch.float32
        assert move_idx.dtype == torch.long
        assert 0 <= move_idx.item() < NUM_MOVES
        assert value_t.dtype == torch.float32
        assert -1.0 <= value_t.item() <= 1.0

    def test_value_target_range(self, lichess_jsonl):
        """Value target should be tanh(cp/400)."""
        ds = LichessEliteDataset(lichess_jsonl, min_depth=18)
        _, _, value_t = ds[0]
        expected = float(np.tanh(-30 / 400.0))
        assert abs(value_t.item() - expected) < 1e-5

    def test_move_is_valid(self, lichess_jsonl):
        """Encoded move should correspond to the original UCI string."""
        ds = LichessEliteDataset(lichess_jsonl, min_depth=18)
        _, move_idx, _ = ds[0]
        from chessgame.encoding.move_encoder import decode_move

        decoded = decode_move(move_idx.item())
        assert decoded is not None
        assert decoded.uci() == "e7e5"


# ---------------------------------------------------------------------------
# StockfishSoftDataset
# ---------------------------------------------------------------------------


class TestStockfishSoftDataset:
    """Tests for the Stockfish soft-label distillation dataset."""

    def test_loads_records(self, stockfish_jsonl):
        ds = StockfishSoftDataset(stockfish_jsonl, min_depth=15)
        assert len(ds) == 2  # 1 filtered out by depth

    def test_item_shapes(self, stockfish_jsonl):
        ds = StockfishSoftDataset(stockfish_jsonl, min_depth=15)
        board_t, soft_policy, value_t = ds[0]
        assert board_t.shape == (8, 8, 119)
        assert soft_policy.shape == (NUM_MOVES,)
        assert soft_policy.dtype == torch.float32
        assert value_t.dtype == torch.float32

    def test_soft_policy_is_distribution(self, stockfish_jsonl):
        """Soft policy should sum to ~1.0 (valid probability distribution)."""
        ds = StockfishSoftDataset(stockfish_jsonl, min_depth=15)
        _, soft_policy, _ = ds[0]
        total = soft_policy.sum().item()
        assert abs(total - 1.0) < 1e-4, f"Soft policy sums to {total}"

    def test_soft_policy_nonnegative(self, stockfish_jsonl):
        ds = StockfishSoftDataset(stockfish_jsonl, min_depth=15)
        _, soft_policy, _ = ds[0]
        assert (soft_policy >= 0).all()

    def test_value_target_bounded(self, stockfish_jsonl):
        ds = StockfishSoftDataset(stockfish_jsonl, min_depth=15)
        _, _, value_t = ds[0]
        assert -1.0 <= value_t.item() <= 1.0


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    """Tests for the circular replay buffer."""

    def test_empty_buffer(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=100)
        samples = [
            (torch.randn(8, 8, 119), torch.randn(NUM_MOVES), 1.0),
            (torch.randn(8, 8, 119), torch.randn(NUM_MOVES), -1.0),
        ]
        buf.add(samples)
        assert len(buf) == 2

    def test_circular_overwrite(self):
        """Buffer should not grow beyond capacity."""
        buf = ReplayBuffer(capacity=3)
        for i in range(10):
            buf.add([(torch.randn(8, 8, 119), torch.randn(NUM_MOVES), float(i))])
        assert len(buf) == 3

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100)
        samples = [
            (torch.randn(8, 8, 119), torch.randn(NUM_MOVES), 1.0) for _ in range(10)
        ]
        buf.add(samples)
        boards, pis, outcomes = buf.sample(4)
        assert boards.shape == (4, 8, 8, 119)
        assert pis.shape == (4, NUM_MOVES)
        assert outcomes.shape == (4,)

    def test_sample_clamps_to_buffer_size(self):
        """Requesting more than available should return all."""
        buf = ReplayBuffer(capacity=100)
        buf.add(
            [(torch.randn(8, 8, 119), torch.randn(NUM_MOVES), 0.0) for _ in range(3)]
        )
        boards, pis, outcomes = buf.sample(100)
        assert boards.shape[0] == 3
