import os
import sys
import json
import chess
import shutil
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from chessgame.model.hrm_chess import HRMChess
from chessgame.train.distill import train
import subprocess


def create_dummy_fens(path, num_records=10):
    board = chess.Board()
    with open(path, "w") as f:
        for _ in range(num_records):
            f.write(json.dumps({"fen": board.fen()}) + "\n")
            board.push(list(board.legal_moves)[0])
            if board.is_game_over():
                board.reset()


def run_test():
    tmp_dir = "tmp_distill_test"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    try:
        # 1. Create dummy input FENs
        input_fens = os.path.join(tmp_dir, "fens.jsonl")
        create_dummy_fens(input_fens)

        # 2. Generate soft labels
        # If stockfish is not found, we'll create a fake soft label file to test the training loop
        sf_path = shutil.which("stockfish")
        soft_labels = os.path.join(tmp_dir, "soft_labels.jsonl")

        if sf_path:
            print(f"Found Stockfish at {sf_path}. Running generator...")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent.parent)
            cmd = [
                sys.executable,
                "scripts/generate_soft_labels.py",
                "--input",
                input_fens,
                "--output",
                soft_labels,
                "--depth",
                "1",
                "--multipv",
                "2",
                "--cpus",
                "2",
            ]
            subprocess.run(cmd, env=env, check=True)
        else:
            print("Stockfish not found. Creating fake soft labels...")
            with open(input_fens, "r") as f_in, open(soft_labels, "w") as f_out:
                for line in f_in:
                    rec = json.loads(line)
                    res = {
                        "fen": rec["fen"],
                        "moves": [
                            {"move": "e2e4", "cp": 10},
                            {"move": "d2d4", "cp": 5},
                        ],
                        "depth": 1,
                    }
                    f_out.write(json.dumps(res) + "\n")

        # 3. Run distillation training
        device = "cpu"
        checkpoint_dir = os.path.join(tmp_dir, "checkpoints")

        print(f"Starting distillation smoke test on {device}...")
        model = train(
            data_path=soft_labels,
            config_name="mac_mini",
            epochs=1,
            batch_size=2,
            lr=1e-4,
            min_depth=1,
            checkpoint_dir=checkpoint_dir,
            device_str=device,
            use_wandb=False,
            accum_steps=1,
        )

        print("Distillation training completed successfully!")
        assert isinstance(model, HRMChess)
        assert os.path.exists(os.path.join(checkpoint_dir, "epoch_1.pt"))
        print("Verification passed!")

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    run_test()
