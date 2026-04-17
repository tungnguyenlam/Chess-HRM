"""
Stockfish annotation pipeline for generating soft-label training data.

Produces JSONL files where each line is:
  {
    "fen":   "<FEN string>",
    "moves": [{"move": "<uci>", "cp": <centipawns>}, ...],
    "depth": <int>
  }

Usage:
    annotator = StockfishAnnotator(stockfish_path="/path/to/stockfish", depth=18, multipv=8)
    results = annotator.annotate_positions([board1, board2, ...])
    annotator.save_jsonl(results, "output.jsonl")

Implements: PLAN.md step 0.4
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import chess
import chess.engine


@dataclass
class AnnotatedPosition:
    """One annotated position with MultiPV Stockfish evaluation."""

    fen: str
    moves: List[dict] = field(default_factory=list)  # [{"move": uci, "cp": int}]
    depth: int = 0


class StockfishAnnotator:
    """
    Annotates chess positions with Stockfish MultiPV evaluations.

    Args:
        stockfish_path: Path to the Stockfish binary. If None, searches PATH.
        depth:          Search depth per position.
        multipv:        Number of principal variations to return.
        threads:        Number of Stockfish threads.
        hash_mb:        Hash table size in MB.
    """

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 18,
        multipv: int = 8,
        threads: int = 1,
        hash_mb: int = 128,
    ):
        if stockfish_path is None:
            stockfish_path = shutil.which("stockfish")
        if stockfish_path is None or not Path(stockfish_path).exists():
            raise FileNotFoundError(
                "Stockfish binary not found. Install Stockfish or pass stockfish_path."
            )

        self.stockfish_path = stockfish_path
        self.depth = depth
        self.multipv = multipv
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _start_engine(self) -> chess.engine.SimpleEngine:
        """Start (or return existing) Stockfish engine process."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self._engine.configure(
                {
                    "Threads": self.threads,
                    "Hash": self.hash_mb,
                }
            )
        return self._engine

    def close(self) -> None:
        """Shut down the engine process."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def annotate_one(self, board: chess.Board) -> AnnotatedPosition:
        """
        Annotate a single position.

        Returns:
            AnnotatedPosition with MultiPV moves sorted by score (best first).
        """
        engine = self._start_engine()

        result = engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=self.multipv,
        )

        moves = []
        actual_depth = 0
        for info in result:
            pv = info.get("pv")
            score = info.get("score")
            depth = info.get("depth", 0)
            actual_depth = max(actual_depth, depth)

            if pv and score:
                # Normalize score to centipawns from White's perspective
                cp = _score_to_cp(score, board.turn)
                moves.append(
                    {
                        "move": pv[0].uci(),
                        "cp": cp,
                    }
                )

        return AnnotatedPosition(
            fen=board.fen(),
            moves=moves,
            depth=actual_depth,
        )

    def annotate_positions(self, boards: List[chess.Board]) -> List[AnnotatedPosition]:
        """Annotate multiple positions sequentially."""
        results = []
        for board in boards:
            results.append(self.annotate_one(board))
        return results

    def save_jsonl(
        self,
        annotations: List[AnnotatedPosition],
        path: str,
    ) -> None:
        """Write annotations to a JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for ann in annotations:
                f.write(json.dumps(asdict(ann)) + "\n")

    @staticmethod
    def load_jsonl(path: str) -> List[AnnotatedPosition]:
        """Read annotations from a JSONL file."""
        results = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                results.append(AnnotatedPosition(**d))
        return results

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    """
    Convert a PovScore to centipawns from White's perspective.
    Mate scores are clamped to ±30000.
    """
    white_score = score.white()
    cp = white_score.score(mate_score=30000)
    if cp is None:
        cp = 0
    return cp
