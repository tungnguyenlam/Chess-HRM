"""
Dataset for supervised pretraining from the Lichess Elite Database.

Expected data format: one JSON record per line (JSONL):
  {"fen": "...", "move": "e2e4", "cp": 30, "depth": 18}

  fen   - FEN string for the position
  move  - best move in UCI notation (e.g. "e2e4", "g1f3")
  cp    - centipawn evaluation from white's perspective
  depth - Stockfish search depth used

You can also generate this from Lichess PGN files that have %eval annotations
using the companion script dataset/build_chess_dataset.py (to be added).

Download pre-annotated data:
  https://database.nikonoel.fr/   (Lichess Elite games with Stockfish evals)
"""
import json
from typing import Optional

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from chessgame.encoding.board_encoder import encode_board
from chessgame.encoding.move_encoder import encode_move


class LichessEliteDataset(Dataset):
    """
    Reads a JSONL file and returns (board_tensor, move_idx, value_target) triples.

    Args:
        path:       Path to a .jsonl file
        min_depth:  Minimum Stockfish depth; shallower records are skipped
        max_records: Cap the dataset size (None = all)
    """

    def __init__(self, path: str, min_depth: int = 18,
                 max_records: Optional[int] = None):
        self.records: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("depth", 0) < min_depth:
                    continue
                self.records.append(rec)
                if max_records is not None and len(self.records) >= max_records:
                    break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        board = chess.Board(rec["fen"])
        move = chess.Move.from_uci(rec["move"])

        board_tensor = encode_board(board, history=[])          # [8, 8, 119]
        move_idx = encode_move(move)                            # int
        value_target = float(np.tanh(rec["cp"] / 400.0))       # [-1, 1]

        return board_tensor, torch.tensor(move_idx, dtype=torch.long), \
               torch.tensor(value_target, dtype=torch.float32)
