import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import chess
import torch

from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.train.supervised import (
    find_latest_epoch_checkpoint,
    resolve_resume_checkpoint,
    train,
)


def _create_dummy_dataset(path: Path, num_records: int = 8, elo: int = 2500) -> None:
    board = chess.Board()
    history = []
    with path.open("w", encoding="utf-8") as f:
        for _ in range(num_records):
            move = list(board.legal_moves)[0]
            record = {
                "fen": board.fen(),
                "history": list(history),
                "move": move.uci(),
                "cp": 10,
                "depth": 20,
                "white_elo": elo,
                "black_elo": elo,
            }
            f.write(json.dumps(record) + "\n")

            history.append(board.fen())
            if len(history) > 7:
                history.pop(0)

            board.push(move)
            if board.is_game_over():
                board.reset()
                history = []


def _write_checkpoint(path: Path, epoch: int = 2) -> None:
    model = HRMChess(HRMChessConfig.mac_mini())
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "step": epoch,
            "config_name": "mac_mini",
        },
        path,
    )


def test_find_latest_epoch_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch_1.pt").touch()
    (checkpoint_dir / "epoch_3.pt").touch()
    (checkpoint_dir / "notes.txt").write_text("keep me", encoding="utf-8")

    latest = find_latest_epoch_checkpoint(str(checkpoint_dir))

    assert latest == str(checkpoint_dir / "epoch_3.pt")


def test_resolve_resume_checkpoint_rerun_clears_epoch_checkpoints(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch_1.pt").touch()
    (checkpoint_dir / "epoch_2.pt").touch()
    marker = checkpoint_dir / "notes.txt"
    marker.write_text("keep me", encoding="utf-8")

    messages = []
    resolved = resolve_resume_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        rerun=True,
        logger=messages.append,
    )

    assert resolved is None
    assert not (checkpoint_dir / "epoch_1.pt").exists()
    assert not (checkpoint_dir / "epoch_2.pt").exists()
    assert marker.exists()
    assert any("Rerun requested" in message for message in messages)


def test_resolve_resume_checkpoint_auto_picks_latest(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch_2.pt").touch()
    (checkpoint_dir / "epoch_5.pt").touch()

    messages = []
    resolved = resolve_resume_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        logger=messages.append,
    )

    assert resolved == str(checkpoint_dir / "epoch_5.pt")
    assert any("Auto-resume" in message for message in messages)


def test_resume_curriculum_uses_raised_elo_floor(tmp_path):
    data_path = tmp_path / "dummy.jsonl"
    checkpoint_path = tmp_path / "epoch_2.pt"

    _create_dummy_dataset(data_path)
    _write_checkpoint(checkpoint_path, epoch=2)

    output = io.StringIO()
    with redirect_stdout(output):
        train(
            data_path=str(data_path),
            config_name="mac_mini",
            epochs=4,
            batch_size=2,
            lr=1e-4,
            min_depth=0,
            min_elo=1800,
            checkpoint_dir=None,
            resume_from=str(checkpoint_path),
            device_str="cpu",
            use_wandb=False,
            accum_steps=1,
            curriculum=True,
            warmup_steps=10,
            num_workers=0,
            log_every_steps=1,
        )

    logs = output.getvalue()
    assert "Curriculum: resuming with min_elo floor 2100" in logs
    assert "Epoch 3/4 start | min_elo=2100" in logs
    assert "Curriculum: Increasing min_elo floor to 2250" in logs
    assert "Epoch 4/4 start | min_elo=2250" in logs
