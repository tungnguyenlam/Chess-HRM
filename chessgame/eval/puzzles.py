"""
Puzzle evaluation: measure model accuracy on the Lichess puzzle database.

Input:  CSV from https://database.lichess.org/#puzzles
        Columns: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

Usage:
    evaluator = PuzzleEvaluator(model, device)
    results = evaluator.evaluate_file("puzzles.csv", max_puzzles=1000)

Implements: PLAN.md step 0.6
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import chess


@dataclass
class PuzzleResult:
    """Result of evaluating a single puzzle."""

    puzzle_id: str
    rating: int
    correct: bool
    model_move: str
    expected_move: str


@dataclass
class PuzzleReport:
    """Aggregate puzzle evaluation results."""

    total: int = 0
    correct: int = 0
    results: List[PuzzleResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    def accuracy_by_rating_bucket(self, bucket_size: int = 400) -> Dict[str, float]:
        """Accuracy grouped by rating intervals."""
        buckets: Dict[str, List[bool]] = {}
        for r in self.results:
            bucket = (r.rating // bucket_size) * bucket_size
            key = f"{bucket}-{bucket + bucket_size}"
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(r.correct)
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}


def parse_puzzle_csv_line(
    row: dict,
) -> Optional[Tuple[chess.Board, chess.Move, int, str]]:
    """
    Parse one row from the Lichess puzzle CSV.

    Returns:
        (board, expected_move, rating, puzzle_id) or None on parse error.

    The CSV 'Moves' column contains the full sequence: the opponent's last move
    followed by the solution moves. The first move is played to reach the puzzle
    position, and the second move is the expected answer.
    """
    try:
        fen = row["FEN"]
        moves_str = row["Moves"].strip().split()
        rating = int(row["Rating"])
        puzzle_id = row["PuzzleId"]

        board = chess.Board(fen)
        # Apply opponent's last move to reach the puzzle position
        if len(moves_str) < 2:
            return None
        board.push_uci(moves_str[0])

        # Expected answer is the second move
        expected = chess.Move.from_uci(moves_str[1])
        return board, expected, rating, puzzle_id
    except (KeyError, ValueError, IndexError):
        return None


class PuzzleEvaluator:
    """
    Evaluate model accuracy on Lichess puzzles.

    Args:
        get_model_move: Callable that takes (board: chess.Board) and returns chess.Move
    """

    def __init__(self, get_model_move: Callable[[chess.Board], chess.Move]):
        self.get_model_move = get_model_move

    def evaluate_one(
        self,
        board: chess.Board,
        expected: chess.Move,
        rating: int,
        puzzle_id: str,
    ) -> PuzzleResult:
        """Evaluate a single puzzle."""
        model_move = self.get_model_move(board)
        correct = model_move == expected
        return PuzzleResult(
            puzzle_id=puzzle_id,
            rating=rating,
            correct=correct,
            model_move=model_move.uci(),
            expected_move=expected.uci(),
        )

    def evaluate_file(
        self,
        csv_path: str,
        max_puzzles: Optional[int] = None,
    ) -> PuzzleReport:
        """Evaluate puzzles from a Lichess puzzle CSV file."""
        report = PuzzleReport()
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = parse_puzzle_csv_line(row)
                if parsed is None:
                    continue
                board, expected, rating, pid = parsed
                result = self.evaluate_one(board, expected, rating, pid)
                report.results.append(result)
                report.total += 1
                if result.correct:
                    report.correct += 1
                if max_puzzles and report.total >= max_puzzles:
                    break
        return report
