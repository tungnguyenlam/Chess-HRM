"""
Dataset for soft-label distillation (Phase 3).

Reads pre-generated JSONL files where each record has multipv Stockfish output:
  {
    "fen":   "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "moves": [
      {"move": "e7e5", "cp": -30},
      {"move": "c7c5", "cp": -50},
      ...
    ],
    "depth": 15
  }

Generate this file offline using dataset/generate_soft_labels.py (to be added),
which calls Stockfish with multipv=8 on a set of positions.
"""
import json
from typing import Optional

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from chessgame.encoding.board_encoder import encode_board
from chessgame.encoding.move_encoder import encode_move, NUM_MOVES


_TEMPERATURE_CP = 100.0   # centipawn temperature for softmax


class StockfishSoftDataset(Dataset):
    """
    Returns (board_tensor, soft_policy[4672], value_target) triples.

    Args:
        path:        Path to a .jsonl file
        min_depth:   Minimum Stockfish depth; shallower records are skipped
        temperature: Softmax temperature in centipawns
        max_records: Cap dataset size
    """

    def __init__(self, path: str, min_depth: int = 12,
                 temperature: float = _TEMPERATURE_CP,
                 max_records: Optional[int] = None):
        self.temperature = temperature
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
        moves_data = rec["moves"]

        board_tensor = encode_board(board, history=[])  # [8, 8, 119]

        # Build sparse soft policy over 4672 moves
        indices = []
        scores = []
        for entry in moves_data:
            try:
                move = chess.Move.from_uci(entry["move"])
                midx = encode_move(move)
                indices.append(midx)
                scores.append(float(entry["cp"]))
            except (KeyError, ValueError):
                continue

        soft_policy = torch.zeros(NUM_MOVES, dtype=torch.float32)
        if indices:
            scores_t = torch.tensor(scores, dtype=torch.float32)
            probs = F.softmax(scores_t / self.temperature, dim=0)
            for i, midx in enumerate(indices):
                if 0 <= midx < NUM_MOVES:
                    soft_policy[midx] = probs[i].item()

        # Value from top-move score
        top_cp = moves_data[0]["cp"] if moves_data else 0
        import numpy as np
        value_target = float(np.tanh(top_cp / 400.0))

        return board_tensor, soft_policy, \
               torch.tensor(value_target, dtype=torch.float32)
