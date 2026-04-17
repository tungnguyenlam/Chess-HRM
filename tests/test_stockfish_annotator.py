"""Tests for chessgame.data.stockfish_annotator — PLAN step 0.4.

These tests require Stockfish to be installed and available in PATH.
If not available, they are skipped automatically.
"""

import json
import shutil
import tempfile
from pathlib import Path

import chess
import pytest

from chessgame.data.stockfish_annotator import (
    StockfishAnnotator,
    AnnotatedPosition,
    _score_to_cp,
)

# Marker for tests that need Stockfish
SF_PATH = shutil.which("stockfish")
requires_stockfish = pytest.mark.skipif(
    SF_PATH is None,
    reason="Stockfish binary not found in PATH",
)


@pytest.fixture
def annotator():
    """Create an annotator with fast settings for testing."""
    ann = StockfishAnnotator(
        stockfish_path=SF_PATH,
        depth=5,  # shallow for speed
        multipv=3,  # few PVs for speed
        threads=1,
        hash_mb=16,
    )
    yield ann
    ann.close()


@requires_stockfish
class TestStockfishAnnotatorInit:
    """Annotator initialization."""

    def test_engine_starts(self, annotator):
        """Engine should start without error."""
        engine = annotator._start_engine()
        assert engine is not None

    def test_missing_binary_raises(self):
        """Non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            StockfishAnnotator(stockfish_path="/nonexistent/stockfish")

    def test_context_manager(self):
        """Should work as context manager."""
        with StockfishAnnotator(stockfish_path=SF_PATH, depth=1) as ann:
            assert ann is not None


@requires_stockfish
class TestAnnotateOne:
    """Single position annotation."""

    def test_starting_position(self, annotator):
        board = chess.Board()
        result = annotator.annotate_one(board)
        assert isinstance(result, AnnotatedPosition)
        assert result.fen == board.fen()
        assert len(result.moves) > 0
        assert result.depth >= 5

    def test_moves_have_required_fields(self, annotator):
        board = chess.Board()
        result = annotator.annotate_one(board)
        for m in result.moves:
            assert "move" in m, "Missing 'move' field"
            assert "cp" in m, "Missing 'cp' field"
            # move should be valid UCI
            chess.Move.from_uci(m["move"])
            # cp should be an int
            assert isinstance(m["cp"], int)

    def test_multipv_count(self, annotator):
        """Should return at most multipv moves."""
        board = chess.Board()
        result = annotator.annotate_one(board)
        assert len(result.moves) <= annotator.multipv

    def test_moves_are_legal(self, annotator):
        """All returned moves should be legal in the position."""
        board = chess.Board()
        result = annotator.annotate_one(board)
        legal_ucis = {m.uci() for m in board.legal_moves}
        for m in result.moves:
            assert m["move"] in legal_ucis, f"Illegal move returned: {m['move']}"

    def test_italian_game(self, annotator, italian_game_board):
        result = annotator.annotate_one(italian_game_board)
        assert len(result.moves) > 0

    def test_checkmate_position(self, annotator):
        """Checkmate position should still annotate (0 legal moves)."""
        b = chess.Board()
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            b.push_uci(uci)
        # White is checkmated. Annotating from White's perspective.
        result = annotator.annotate_one(b)
        assert isinstance(result, AnnotatedPosition)
        # Stockfish should return 0 moves for checkmate
        assert len(result.moves) == 0


@requires_stockfish
class TestAnnotateMultiple:
    """Batch annotation."""

    def test_multiple_positions(self, annotator):
        boards = [
            chess.Board(),
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ]
        results = annotator.annotate_positions(boards)
        assert len(results) == 2
        assert all(isinstance(r, AnnotatedPosition) for r in results)


@requires_stockfish
class TestJsonlIO:
    """Save and load JSONL files."""

    def test_roundtrip(self, annotator):
        """Save annotations, reload, verify match."""
        board = chess.Board()
        results = annotator.annotate_positions([board])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.jsonl")
            annotator.save_jsonl(results, path)

            # Verify file exists and has content
            assert Path(path).exists()
            lines = Path(path).read_text().strip().split("\n")
            assert len(lines) == 1

            # Reload
            loaded = StockfishAnnotator.load_jsonl(path)
            assert len(loaded) == 1
            assert loaded[0].fen == results[0].fen
            assert loaded[0].depth == results[0].depth
            assert len(loaded[0].moves) == len(results[0].moves)

    def test_jsonl_format(self, annotator):
        """Each line should be valid JSON."""
        board = chess.Board()
        results = annotator.annotate_positions([board])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.jsonl")
            annotator.save_jsonl(results, path)
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    assert "fen" in d
                    assert "moves" in d
                    assert "depth" in d


class TestScoreConversion:
    """_score_to_cp helper — always available, no Stockfish needed."""

    def test_positive_cp(self):
        """Positive score from White's perspective."""
        score = chess.engine.PovScore(chess.engine.Cp(150), chess.WHITE)
        assert _score_to_cp(score, chess.WHITE) == 150

    def test_negative_cp(self):
        score = chess.engine.PovScore(chess.engine.Cp(-200), chess.WHITE)
        assert _score_to_cp(score, chess.WHITE) == -200

    def test_mate_score(self):
        """Mate in 3 should return large positive value."""
        score = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
        cp = _score_to_cp(score, chess.WHITE)
        # python-chess: Mate(3).score(mate_score=30000) = 30000 - 3 = 29997
        assert cp == 29997

    def test_mated_score(self):
        score = chess.engine.PovScore(chess.engine.Mate(-3), chess.WHITE)
        cp = _score_to_cp(score, chess.WHITE)
        # Mate(-3).score(mate_score=30000) = -30000 + 3 = -29997
        assert cp == -29997

    def test_black_perspective(self):
        """Score should be from White's perspective regardless of turn."""
        score = chess.engine.PovScore(chess.engine.Cp(100), chess.BLACK)
        cp = _score_to_cp(score, chess.BLACK)
        # PovScore.white() flips sign for BLACK
        assert cp == -100
