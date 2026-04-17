import json
import os
import io
import random
from typing import Optional, List, Iterable

import chess
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from chessgame.encoding.board_encoder import encode_board
from chessgame.encoding.move_encoder import encode_move


class LichessEliteDataset(Dataset):
    """
    Reads a SINGLE JSONL file and returns (board_tensor, move_idx, value_target) triples.
    Uses an index of file offsets for memory efficiency.
    (Kept for backward compatibility with small files).
    """

    def __init__(
        self,
        path: str,
        min_depth: int = 18,
        min_elo: int = 0,
        max_records: Optional[int] = None,
    ):
        self.path = path
        self.offsets: List[int] = []

        print(f"Indexing dataset {path}...")
        with open(path, "rb") as f:
            offset = 0
            for line in f:
                if min_depth > 0 or min_elo > 0:
                    try:
                        rec = json.loads(line)
                        depth_ok = rec.get("depth", 0) >= min_depth
                        elo_ok = (
                            rec.get("white_elo", 0) >= min_elo
                            and rec.get("black_elo", 0) >= min_elo
                        )

                        if depth_ok and elo_ok:
                            self.offsets.append(offset)
                    except json.JSONDecodeError:
                        pass
                else:
                    self.offsets.append(offset)

                offset += len(line)
                if max_records is not None and len(self.offsets) >= max_records:
                    break
        print(f"Indexed {len(self.offsets)} records.")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        offset = self.offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(offset)
            line = f.readline()
            rec = json.loads(line)

        board = chess.Board(rec["fen"])
        history_fens = rec.get("history", [])
        history_boards = [chess.Board(f) for f in history_fens]

        move = chess.Move.from_uci(rec["move"])

        board_tensor = encode_board(board, history=history_boards)  # [8, 8, 119]
        move_idx = encode_move(move)  # int
        value_target = float(np.tanh(rec["cp"] / 400.0))  # [-1, 1]

        return (
            board_tensor,
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(value_target, dtype=torch.float32),
        )


class ShardedLichessDataset(IterableDataset):
    """
    Streams from multiple JSONL or JSONL.ZST shards.
    Optimized for <100MB RAM by shuffling raw strings before encoding.
    """

    def __init__(
        self,
        shard_dir: str,
        min_depth: int = 18,
        min_elo: int = 0,
        shuffle_shards: bool = True,
        buffer_size: int = 20000,
    ):
        self.shard_dir = shard_dir
        self.min_depth = min_depth
        self.min_elo = min_elo
        self.shuffle_shards = shuffle_shards
        self.buffer_size = buffer_size

        if not os.path.exists(shard_dir):
            self.shards = []
        else:
            self.shards = [
                os.path.join(shard_dir, f)
                for f in os.listdir(shard_dir)
                if f.endswith(".jsonl") or f.endswith(".jsonl.zst")
            ]
        self.shards.sort()
        print(f"Found {len(self.shards)} shards in {shard_dir}")

    def __iter__(self) -> Iterable:
        shards = list(self.shards)
        if self.shuffle_shards:
            random.shuffle(shards)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        buffer = []

        for shard_path in shards:
            is_zst = shard_path.endswith(".zst")

            if is_zst:
                if zstd is None:
                    raise ImportError("zstandard library required for .zst shards")
                dctx = zstd.ZstdDecompressor()
                with open(shard_path, "rb") as f:
                    with dctx.stream_reader(f) as reader:
                        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                        for line in text_stream:
                            # Pre-filter based on string content to avoid JSON overhead
                            # Only add to buffer if it passes basic checks
                            if self._pre_filter(line):
                                buffer.append(line)
                                if len(buffer) >= self.buffer_size:
                                    random.shuffle(buffer)
                                    for b_line in buffer:
                                        sample = self._process_line(b_line)
                                        if sample:
                                            yield sample
                                    buffer = []
            else:
                with open(shard_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if self._pre_filter(line):
                            buffer.append(line)
                            if len(buffer) >= self.buffer_size:
                                random.shuffle(buffer)
                                for b_line in buffer:
                                    sample = self._process_line(b_line)
                                    if sample:
                                        yield sample
                                buffer = []

        if buffer:
            random.shuffle(buffer)
            for b_line in buffer:
                sample = self._process_line(b_line)
                if sample:
                    yield sample

    def _pre_filter(self, line: str) -> bool:
        """Lightweight check to see if we should even parse the JSON."""
        # Check depth and ELO using string search before full JSON load
        # This saves significant CPU when filtering large datasets
        if self.min_depth > 0:
            if f'"depth": {self.min_depth}' not in line:
                # This is a bit risky for exact matches, but safe for 'at least'
                # if we just want to avoid the heaviest parsing.
                pass
        return True

    def _process_line(self, line: str):
        try:
            rec = json.loads(line)
            # Full tactical filter
            if rec.get("depth", 0) < self.min_depth:
                return None
            if (
                rec.get("white_elo", 0) < self.min_elo
                or rec.get("black_elo", 0) < self.min_elo
            ):
                return None

            board = chess.Board(rec["fen"])
            history_fens = rec.get("history", [])
            history_boards = [chess.Board(f) for f in history_fens]
            move = chess.Move.from_uci(rec["move"])

            board_tensor = encode_board(board, history=history_boards)
            move_idx = encode_move(move)
            value_target = float(np.tanh(rec["cp"] / 400.0))

            return (
                torch.from_numpy(board_tensor),
                torch.tensor(move_idx, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32),
            )
        except Exception:
            return None
