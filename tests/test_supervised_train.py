import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import json
import chess
import shutil
from chessgame.model.hrm_chess import HRMChess
from chessgame.train.supervised import train


def create_dummy_dataset(path, num_records=20):
    board = chess.Board()
    records = []
    history = []
    for _ in range(num_records):
        legal_moves = list(board.legal_moves)
        move = legal_moves[0]

        rec = {
            "fen": board.fen(),
            "history": list(history),
            "move": move.uci(),
            "cp": 10,
            "depth": 20,
        }
        records.append(rec)

        history.append(board.fen())
        if len(history) > 7:
            history.pop(0)

        board.push(move)
        if board.is_game_over():
            board.reset()
            history = []

    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run_test():
    tmp_dir = "tmp_test_data"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    try:
        # 1. Create dummy data
        data_path = os.path.join(tmp_dir, "dummy.jsonl")
        create_dummy_dataset(data_path)

        # 2. Run training
        device = "cpu"  # Force CPU for smoke test to avoid Metal/MPS issues

        checkpoint_dir = os.path.join(tmp_dir, "checkpoints")

        print(f"Starting smoke test training on {device}...")
        model = train(
            data_path=str(data_path),
            config_name="mac_mini",
            epochs=1,
            batch_size=4,
            lr=1e-3,
            checkpoint_dir=str(checkpoint_dir),
            device_str=device,
            use_wandb=False,
            accum_steps=1,
        )

        print("Training completed successfully!")
        assert isinstance(model, HRMChess)
        assert os.path.exists(os.path.join(checkpoint_dir, "epoch_1.pt"))
        print("Verification passed!")

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    run_test()
