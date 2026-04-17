#!/usr/bin/env python3
"""
Convert Lichess PGN (standard or elite) to Sharded JSONL format.
Supports streaming directly from .zst compressed PGN files.
Handles games without evaluation annotations by using game result as value target.
"""

import argparse
import json
import re
import os
import io
import chess
import chess.pgn

try:
    import zstandard as zstd

    _ZSTD = True
except ImportError:
    _ZSTD = False


def parse_pgn_eval(comment):
    """Parse %eval from PGN comment, e.g., '#+2.5' or '0.5'."""
    if not comment:
        return None, 0
    match = re.search(r"([#][+-]?|[+-]?)(\d+\.?\d*)", comment)
    if not match:
        return None, 0
    sign = match.group(1)
    value = float(match.group(2))
    if "#" in sign:
        cp = 10000 if "+" in sign else -10000
    else:
        cp = int(value * 100)
    return cp, 1


def get_result_value(result_str):
    """Convert PGN Result to numeric value: 1.0, 0.0, or 0.5."""
    if result_str == "1-0":
        return 1.0
    elif result_str == "0-1":
        return -1.0
    return 0.0


def get_shard_path(output_dir, shard_idx, compress=False):
    ext = ".jsonl.zst" if compress else ".jsonl"
    return os.path.join(output_dir, f"shard_{shard_idx:04d}{ext}")


def convert_pgn(
    pgn_path,
    output_dir,
    min_depth=0,
    max_games=None,
    games_per_shard=50000,
    compress=True,
    min_elo=1800,
):
    os.makedirs(output_dir, exist_ok=True)

    game_count = 0
    record_count = 0
    shard_idx = 0

    cctx = zstd.ZstdCompressor() if compress else None

    def open_shard(idx):
        path = get_shard_path(output_dir, idx, compress)
        print(f"Opening shard {idx}: {path}")
        if compress:
            f = open(path, "wb")
            return cctx.stream_writer(f), f
        else:
            f = open(path, "w")
            return f, f

    out_stream, out_file = open_shard(shard_idx)

    # Handle .zst input streaming
    if pgn_path.endswith(".zst"):
        if not _ZSTD:
            raise ImportError("zstandard library required for .zst files")
        raw_f = open(pgn_path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(raw_f)
        pgn_f = io.TextIOWrapper(stream_reader, encoding="utf-8")
    else:
        pgn_f = open(pgn_path, "r", encoding="utf-8")

    try:
        while True:
            game = chess.pgn.read_game(pgn_f)
            if game is None:
                break

            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            result_str = game.headers.get("Result", "*")

            # Use game outcome as default value target if no eval
            outcome_val = get_result_value(result_str)

            if white_elo < min_elo or black_elo < min_elo:
                continue

            board = game.board()
            history_fens = []
            for move in game.mainline():
                comment = move.comment
                cp, depth = parse_pgn_eval(comment)

                # If no depth info, we still take the move for Policy training
                # and use game outcome for Value training.
                target_cp = (
                    cp if depth > 0 else int(outcome_val * 100)
                )  # placeholder scale
                target_depth = depth

                # If min_depth is set, only keep annotated moves.
                # If min_depth is 0, we keep ALL moves from elite games.
                if min_depth == 0 or target_depth >= min_depth:
                    record = {
                        "fen": board.fen(),
                        "history": list(history_fens),
                        "move": move.move.uci(),
                        "cp": target_cp,
                        "depth": target_depth,
                        "white_elo": white_elo,
                        "black_elo": black_elo,
                        "outcome": outcome_val,
                    }
                    line = json.dumps(record) + "\n"
                    if compress:
                        out_stream.write(line.encode("utf-8"))
                    else:
                        out_stream.write(line)
                    record_count += 1

                history_fens.append(board.fen())
                if len(history_fens) > 7:
                    history_fens.pop(0)
                board.push(move.move)

            game_count += 1

            if game_count % games_per_shard == 0:
                out_stream.flush()
                out_file.close()
                shard_idx += 1
                out_stream, out_file = open_shard(shard_idx)

            if max_games and game_count >= max_games:
                break

            if game_count % 1000 == 0:
                print(f"Processed {game_count} games, {record_count} moves...")

    finally:
        out_stream.flush()
        out_file.close()
        pgn_f.close()
        if pgn_path.endswith(".zst"):
            raw_f.close()

    print(f"Done! Wrote {record_count} records across {shard_idx + 1} shards.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input PGN file (.pgn or .pgn.zst)")
    parser.add_argument("output_dir", help="Output directory for shards")
    parser.add_argument("--min_depth", type=int, default=0)
    parser.add_argument("--min_elo", type=int, default=1800)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--games_per_shard", type=int, default=50000)
    parser.add_argument("--compress", action="store_true", default=True)
    parser.add_argument("--no_compress", action="store_false", dest="compress")
    args = parser.parse_args()

    convert_pgn(
        args.input,
        args.output_dir,
        min_depth=args.min_depth,
        max_games=args.max_games,
        games_per_shard=args.games_per_shard,
        compress=args.compress,
        min_elo=args.min_elo,
    )
