"""conftest.py — shared fixtures for all test modules."""

import sys
from pathlib import Path

import pytest

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def starting_board():
    """Standard chess starting position."""
    import chess

    return chess.Board()


@pytest.fixture
def italian_game_board():
    """Italian Game position after 1.e4 e5 2.Nf3 Nc6 3.Bc4."""
    import chess

    b = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]:
        b.push_uci(uci)
    return b


@pytest.fixture
def promotion_board():
    """White pawn on e7, Black king on h8. White to move, can promote."""
    import chess

    return chess.Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")


@pytest.fixture
def board_with_history():
    """A board with 10 moves of history (enough for 8-ply lookback)."""
    import chess

    b = chess.Board()
    moves = [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1c4",
        "g8f6",
        "d2d3",
        "f8e7",
        "e1g1",
        "e8g8",
    ]
    history = []
    for uci in moves:
        history.append(b.copy())
        b.push_uci(uci)
    # Return board + history in most-recent-first order
    return b, list(reversed(history))
