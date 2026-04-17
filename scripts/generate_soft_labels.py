"""
Generate soft labels for distillation (Phase 3).
Takes a JSONL file of FENs (e.g. from Phase 2) and runs Stockfish MultiPV 
to produce the soft policy targets.

Refactored to stream input and output to disk for memory efficiency.

Usage:
    python scripts/generate_soft_labels.py \
        --input data/lichess_elite.jsonl \
        --output data/stockfish_soft.jsonl \
        --depth 20 \
        --multipv 8 \
        --cpus 4
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Iterator

import chess
from tqdm import tqdm

from chessgame.data.stockfish_annotator import StockfishAnnotator


def worker_init(stockfish_path, depth, multipv, threads_per_sf):
    global annotator
    annotator = StockfishAnnotator(
        stockfish_path=stockfish_path,
        depth=depth,
        multipv=multipv,
        threads=threads_per_sf,
    )


def worker_task(fen: str) -> str:
    """Run annotation and return a JSON string to keep results small in memory."""
    global annotator
    board = chess.Board(fen)
    result = annotator.annotate_one(board)
    return json.dumps({"fen": result.fen, "moves": result.moves, "depth": result.depth})


def fen_generator(path: str, max_records: int = None) -> Iterator[str]:
    """Stream FENs from input file without loading full list into memory."""
    count = 0
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "fen" in rec:
                yield rec["fen"]
                count += 1
            if max_records and count >= max_records:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL with 'fen' fields")
    parser.add_argument("--output", required=True, help="Output JSONL for soft labels")
    parser.add_argument("--stockfish_path", default=None)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--cpus", type=int, default=mp.cpu_count())
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--threads_per_sf", type=int, default=1)
    args = parser.parse_args()

    # Process in parallel with streaming
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Estimate total for progress bar (fast because we don't parse JSON)
    print(f"Estimating total records in {args.input}...")
    total_est = 0
    with open(args.input, "rb") as f:
        for _ in f:
            total_est += 1
    if args.max_records:
        total_est = min(total_est, args.max_records)

    print(
        f"Starting annotation with {args.cpus} workers. Results stream to {args.output}..."
    )

    with open(args.output, "w") as out_f:
        with mp.Pool(
            processes=args.cpus,
            initializer=worker_init,
            initargs=(
                args.stockfish_path,
                args.depth,
                args.multipv,
                args.threads_per_sf,
            ),
        ) as pool:
            # imap returns results as they are ready.
            # Use a chunksize > 1 to reduce IPC overhead if tasks are very short.
            for result_json in tqdm(
                pool.imap(worker_task, fen_generator(args.input, args.max_records)),
                total=total_est,
            ):
                out_f.write(result_json + "\n")
                # Optional: out_f.flush() every N records if you want high safety

    print("Done!")


if __name__ == "__main__":
    main()
