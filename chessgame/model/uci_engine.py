"""
UCI (Universal Chess Interface) wrapper for HRM-GAB Chess.
Allows playing against the model in standard chess GUIs.

Usage:
    python -m chessgame.model.uci_engine --checkpoint checkpoints/supervised/epoch_5.pt
"""

import argparse
import sys
import chess
import torch

from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.train.mcts import MCTS
from chessgame.train.runtime import log_runtime, resolve_training_runtime


class UCIEngine:
    def __init__(
        self,
        checkpoint_path: str,
        config_name: str = "mac_mini",
        device: str = "auto",
        forward_dtype: str = "auto",
    ):
        # Load model
        config = (
            HRMChessConfig.full()
            if config_name == "full"
            else HRMChessConfig.mac_mini()
        )
        runtime = resolve_training_runtime(
            config=config,
            device_str=device,
            forward_dtype_str=forward_dtype,
        )
        log_runtime(runtime, lambda message: sys.stderr.write(message + "\n"))

        self.device = runtime.device
        self.model = HRMChess(config).to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

        self.board = chess.Board()
        self.mcts = MCTS(
            self.model, num_simulations=400, batch_size=16, device=str(self.device)
        )

    def main_loop(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                command = line.strip()
                if command == "uci":
                    print("id name HRM-GAB Chess")
                    print("id author Sapient/Tung Nguyen")
                    print("uciok")
                    sys.stdout.flush()
                elif command == "isready":
                    print("readyok")
                    sys.stdout.flush()
                elif command == "ucinewgame":
                    self.board = chess.Board()
                elif command.startswith("position"):
                    self._handle_position(command)
                elif command.startswith("go"):
                    self._handle_go(command)
                elif command == "quit":
                    break
                elif command == "stop":
                    # MCTS isn't easily stoppable in this sync implementation yet
                    pass
            except EOFError:
                break
            except Exception as e:
                # Log to stderr to avoid confusing GUI
                sys.stderr.write(f"Error: {str(e)}\n")
                sys.stderr.flush()

    def _handle_position(self, command):
        # position [fen <fen> | startpos] moves <move1> ...
        parts = command.split()
        if len(parts) < 2:
            return

        idx = 1
        if parts[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif parts[idx] == "fen":
            # Fen is multiple parts
            fen_parts = []
            idx += 1
            while idx < len(parts) and parts[idx] != "moves":
                fen_parts.append(parts[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_parts))

        if idx < len(parts) and parts[idx] == "moves":
            idx += 1
            while idx < len(parts):
                move = chess.Move.from_uci(parts[idx])
                self.board.push(move)
                idx += 1

    def _handle_go(self, command):
        # Very basic go implementation: just use fixed simulations for now
        # GUI might send wtime, btime, etc.
        probs = self.mcts.search(self.board)

        # Select move with highest visit count
        move_idx = probs.argmax().item()
        from chessgame.encoding.move_encoder import decode_move

        move = decode_move(move_idx, self.board)

        print(f"bestmove {move.uci()}")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="mac_mini")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--forward_dtype", default="auto")
    args = parser.parse_args()

    engine = UCIEngine(args.checkpoint, args.config, args.device, args.forward_dtype)
    engine.main_loop()
