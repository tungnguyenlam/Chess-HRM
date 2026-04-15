"""Tests for chessgame.eval.* — PLAN step 0.6.

Tests puzzle evaluator and interpretability scaffolding.
Arena tests (evaluate_chess.py) require Stockfish — skipped if not available.
"""
import csv
import tempfile
from pathlib import Path

import chess
import pytest

from chessgame.eval.puzzles import (
    PuzzleEvaluator,
    PuzzleReport,
    PuzzleResult,
    parse_puzzle_csv_line,
)
from chessgame.eval.interpretability import (
    GABSnapshot,
    ACTDepthRecord,
    classify_game_phase,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def puzzle_csv(tmp_path):
    """Create a small Lichess-format puzzle CSV."""
    path = tmp_path / "puzzles.csv"
    rows = [
        # Scholar's mate puzzle: after 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6??
        # Position after Nf6, puzzle is Qxf7#
        {
            "PuzzleId": "00001",
            "FEN": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "Moves": "c4f7 e8f7 h5f7",  # dummy setup move + expected: Qxf7+
            "Rating": "600",
            "RatingDeviation": "100",
            "Popularity": "99",
            "NbPlays": "10000",
            "Themes": "mate mateIn1",
            "GameUrl": "https://lichess.org/test",
            "OpeningTags": "",
        },
        # Simple tactic
        {
            "PuzzleId": "00002",
            "FEN": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "Moves": "e7e5 d2d4",  # setup: e5, expected: d4
            "Rating": "1200",
            "RatingDeviation": "50",
            "Popularity": "80",
            "NbPlays": "5000",
            "Themes": "opening",
            "GameUrl": "https://lichess.org/test2",
            "OpeningTags": "",
        },
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


# ---------------------------------------------------------------------------
# parse_puzzle_csv_line
# ---------------------------------------------------------------------------

class TestParsePuzzle:
    def test_valid_row(self):
        row = {
            "PuzzleId": "001",
            "FEN": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "Moves": "e7e5 d2d4",
            "Rating": "1200",
        }
        result = parse_puzzle_csv_line(row)
        assert result is not None
        board, expected, rating, pid = result
        assert isinstance(board, chess.Board)
        assert expected == chess.Move.from_uci("d2d4")
        assert rating == 1200
        assert pid == "001"

    def test_applies_setup_move(self):
        """The first move in Moves should be applied to reach puzzle position."""
        row = {
            "PuzzleId": "002",
            "FEN": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "Moves": "e7e5 g1f3",
            "Rating": "800",
        }
        result = parse_puzzle_csv_line(row)
        board, _, _, _ = result
        # After e7e5, it's White's turn
        assert board.turn == chess.WHITE
        # e5 pawn should be on the board
        assert board.piece_at(chess.E5) is not None

    def test_invalid_row(self):
        result = parse_puzzle_csv_line({"bad": "data"})
        assert result is None

    def test_single_move_returns_none(self):
        """Need at least 2 moves (setup + answer)."""
        row = {
            "PuzzleId": "003",
            "FEN": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "Moves": "e7e5",
            "Rating": "800",
        }
        assert parse_puzzle_csv_line(row) is None


# ---------------------------------------------------------------------------
# PuzzleEvaluator
# ---------------------------------------------------------------------------

class TestPuzzleEvaluator:
    def test_perfect_model(self, puzzle_csv):
        """A model that always picks the expected move should get 100%."""
        # Track which move to return
        expected_moves = {}

        def precompute(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    parsed = parse_puzzle_csv_line(row)
                    if parsed:
                        board, expected, _, pid = parsed
                        expected_moves[board.fen()] = expected

        precompute(puzzle_csv)

        def perfect_move(board):
            return expected_moves[board.fen()]

        evaluator = PuzzleEvaluator(get_model_move=perfect_move)
        report = evaluator.evaluate_file(puzzle_csv)
        assert report.accuracy == 1.0
        assert report.total == 2
        assert report.correct == 2

    def test_wrong_model(self, puzzle_csv):
        """A model that always picks a1a2 gets 0% (it's never the right answer)."""
        def bad_move(board):
            return chess.Move.from_uci("a1a2")

        evaluator = PuzzleEvaluator(get_model_move=bad_move)
        report = evaluator.evaluate_file(puzzle_csv)
        assert report.accuracy == 0.0

    def test_max_puzzles(self, puzzle_csv):
        def dummy(board):
            return chess.Move.from_uci("a1a2")

        evaluator = PuzzleEvaluator(get_model_move=dummy)
        report = evaluator.evaluate_file(puzzle_csv, max_puzzles=1)
        assert report.total == 1

    def test_rating_buckets(self, puzzle_csv):
        def dummy(board):
            return chess.Move.from_uci("a1a2")

        evaluator = PuzzleEvaluator(get_model_move=dummy)
        report = evaluator.evaluate_file(puzzle_csv)
        buckets = report.accuracy_by_rating_bucket(bucket_size=400)
        assert len(buckets) >= 1  # At least one bucket


class TestPuzzleReport:
    def test_empty_report(self):
        r = PuzzleReport()
        assert r.accuracy == 0.0

    def test_accuracy_calculation(self):
        r = PuzzleReport(total=10, correct=7)
        assert abs(r.accuracy - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# Interpretability scaffolding
# ---------------------------------------------------------------------------

class TestInterpretability:
    def test_classify_opening(self):
        assert classify_game_phase(5, 32) == "opening"

    def test_classify_middlegame(self):
        assert classify_game_phase(20, 20) == "middlegame"

    def test_classify_endgame(self):
        assert classify_game_phase(40, 6) == "endgame"

    def test_gab_snapshot_dataclass(self):
        import torch
        snap = GABSnapshot(cycle=1, bias=torch.zeros(8, 65, 65))
        assert snap.cycle == 1
        assert snap.bias.shape == (8, 65, 65)

    def test_act_depth_record(self):
        rec = ACTDepthRecord(
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            halt_step=3,
            phase="opening",
            sf_eval_variance=0.5,
        )
        assert rec.halt_step == 3
