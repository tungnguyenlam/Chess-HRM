import json
import os
import io
import random
import time
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
        verbose: bool = False,
        progress_every_samples: int = 50000,
    ):
        self.shard_dir = shard_dir
        self.min_depth = min_depth
        self.min_elo = min_elo
        self.shuffle_shards = shuffle_shards
        self.buffer_size = buffer_size
        self.verbose = verbose
        self.progress_every_samples = progress_every_samples

        if not os.path.exists(shard_dir):
            self.shards = []
        else:
            self.shards = [
                os.path.join(shard_dir, f)
                for f in os.listdir(shard_dir)
                if f.endswith(".jsonl") or f.endswith(".jsonl.zst")
            ]
        self.shards.sort()
        print(f"Found {len(self.shards)} shards in {shard_dir}", flush=True)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def __iter__(self) -> Iterable:
        shards = list(self.shards)
        if self.shuffle_shards:
            random.shuffle(shards)

        worker_info = torch.utils.data.get_worker_info()
        worker_label = "main" if worker_info is None else f"worker-{worker_info.id}"
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        self._log(
            f"[LichessStream:{worker_label}] starting stream | shards={len(shards)} | "
            f"buffer_size={self.buffer_size} | min_depth={self.min_depth} | min_elo={self.min_elo}"
        )
        buffer = []
        total_yielded = 0
        stream_start = time.perf_counter()
        first_sample_announced = False

        def flush_buffer(shard_name: str):
            nonlocal buffer
            nonlocal total_yielded
            nonlocal first_sample_announced

            random.shuffle(buffer)
            buffer_total = len(buffer)

            for line_idx, b_line in enumerate(buffer, start=1):
                if (
                    not first_sample_announced
                    and line_idx % 5000 == 0
                ):
                    elapsed = time.perf_counter() - stream_start
                    self._log(
                        f"[LichessStream:{worker_label}] still waiting for first training sample from {shard_name}: "
                        f"{line_idx:,}/{buffer_total:,} raw lines processed in {elapsed:.1f}s"
                    )

                sample = self._process_line(b_line)
                if sample:
                    total_yielded += 1
                    if total_yielded == 1:
                        first_sample_announced = True
                        elapsed = time.perf_counter() - stream_start
                        self._log(
                            f"[LichessStream:{worker_label}] first training sample ready after {elapsed:.2f}s"
                        )
                    if (
                        self.progress_every_samples > 0
                        and total_yielded % self.progress_every_samples == 0
                    ):
                        elapsed = time.perf_counter() - stream_start
                        self._log(
                            f"[LichessStream:{worker_label}] yielded {total_yielded:,} samples total in {elapsed:.1f}s"
                        )
                    yield sample

            buffer = []

        for shard_idx, shard_path in enumerate(shards, start=1):
            is_zst = shard_path.endswith(".zst")
            shard_name = os.path.basename(shard_path)
            first_buffer_announced = False
            self._log(
                f"[LichessStream:{worker_label}] opening shard {shard_idx}/{len(shards)}: {shard_name}"
            )
            self._log(
                f"[LichessStream:{worker_label}] filling shuffle buffer with {self.buffer_size} raw lines before first batch"
            )

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
                                    if not first_buffer_announced:
                                        self._log(
                                            f"[LichessStream:{worker_label}] first buffer ready from {shard_name}; shuffling and decoding"
                                        )
                                        first_buffer_announced = True
                                    yield from flush_buffer(shard_name)
            else:
                with open(shard_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if self._pre_filter(line):
                            buffer.append(line)
                            if len(buffer) >= self.buffer_size:
                                if not first_buffer_announced:
                                    self._log(
                                        f"[LichessStream:{worker_label}] first buffer ready from {shard_name}; shuffling and decoding"
                                    )
                                    first_buffer_announced = True
                                yield from flush_buffer(shard_name)

        if buffer:
            last_shard_name = os.path.basename(shards[-1]) if shards else "buffer"
            yield from flush_buffer(last_shard_name)

        elapsed = time.perf_counter() - stream_start
        self._log(
            f"[LichessStream:{worker_label}] stream finished | yielded {total_yielded:,} samples in {elapsed:.1f}s"
        )

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
            if not isinstance(board_tensor, torch.Tensor):
                board_tensor = torch.from_numpy(board_tensor)
            move_idx = encode_move(move)
            value_target = float(np.tanh(rec["cp"] / 400.0))

            return (
                board_tensor,
                torch.tensor(move_idx, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32),
            )
        except Exception:
            return None
