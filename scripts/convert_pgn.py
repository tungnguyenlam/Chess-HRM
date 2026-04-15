#!/usr/bin/env python3
"""
Convert Lichess PGN with %eval annotations to JSONL format.

Usage:
    python scripts/convert_pgn.py input.pgn output.jsonl --min_depth 18
"""

import argparse
import json
import re
import chess
import chess.pgn


def parse_pgn_eval(comment):
    """Parse %eval from PGN comment, e.g., '#+2.5' or '0.5'."""
    if not comment:
        return None, 0
    # Match patterns like: (0.5), (#+2.5), (-0.3)
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


def convert_pgn(pgn_path, output_path, min_depth=18, max_games=None):
    count = 0
    with open(output_path, "w") as out:
        with open(pgn_path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                # Get headers
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))

                # Filter for elite players
                if white_elo < 1800 or black_elo < 1800:
                    continue

                board = game.board()
                for move in game.mainline():
                    # Get comment (evaluation)
                    comment = move.comment
                    cp, depth = parse_pgn_eval(comment)

                    if depth >= min_depth:
                        record = {
                            "fen": board.fen(),
                            "move": move.move.uci(),
                            "cp": cp,
                            "depth": depth,
                        }
                        out.write(json.dumps(record) + "\n")
                        count += 1

                    board.push(move.move)

                if max_games and count >= max_games:
                    break

                if count % 10000 == 0:
                    print(f"Processed {count} moves...")

    print(f"Done! Wrote {count} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input PGN file")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument("--min_depth", type=int, default=18)
    parser.add_argument("--max_games", type=int, default=None)
    args = parser.parse_args()
    convert_pgn(args.input, args.output, args.min_depth, args.max_games)
