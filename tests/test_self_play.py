import os
import sys
import torch
import shutil
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.train.self_play import play_game, train_step, ReplayBuffer


def test_play_game():
    config = HRMChessConfig.mac_mini()
    device = torch.device("cpu")
    model = HRMChess(config).to(device)
    model.eval()

    # Play a very short game with 2 sims
    game_data, outcome = play_game(model, num_sims=2, device=device)

    assert len(game_data) > 0
    assert isinstance(game_data[0][0], torch.Tensor)  # board_t
    assert isinstance(game_data[0][1], torch.Tensor)  # pi
    assert isinstance(outcome, float)


def test_train_step():
    config = HRMChessConfig.mac_mini()
    device = torch.device("cpu")
    model = HRMChess(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    buffer = ReplayBuffer(capacity=100)

    # Add dummy data
    board_t = torch.randn(8, 8, 119)
    pi = torch.zeros(4672)
    pi[0] = 1.0
    outcome = 1.0
    buffer.add([(board_t, pi, outcome)])

    loss = train_step(
        model, opt, buffer, batch_size=1, device=device, target_dtype=torch.float32
    )
    assert loss > 0


def test_self_play_main_smoke():
    # We can't easily run main() because it requires a checkpoint file.
    # But we can test the logic by creating a dummy checkpoint.
    tmp_dir = "tmp_self_play_test"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    try:
        config = HRMChessConfig.mac_mini()
        model = HRMChess(config)
        ckpt_path = os.path.join(tmp_dir, "dummy.pt")
        torch.save({"model": model.state_dict()}, ckpt_path)

        # We'll run a few games via subprocess to test the CLI
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "chessgame.train.self_play",
            "--checkpoint",
            ckpt_path,
            "--num_games",
            "2",
            "--sims",
            "2",
            "--batch_size",
            "2",
            "--device",
            "cpu",
        ]
        # Set PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent)

        print("Running self-play smoke test...")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
        assert result.returncode == 0

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
