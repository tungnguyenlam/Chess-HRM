"""
AlphaZero 8x8x119 board encoding.

Plane layout (119 total):
  [ t*14 :  t*14+6 ]  white pieces for history step t  (pawn,knight,bishop,rook,queen,king)
  [ t*14+6: t*14+12]  black pieces for history step t
  [ t*14+12]          repetition: position seen >= 1 times
  [ t*14+13]          repetition: position seen >= 2 times
  (t = 0..7, most-recent first)

  [112]  side to move  (1 = white)
  [113]  total move count / 500
  [114]  white kingside castling
  [115]  white queenside castling
  [116]  black kingside castling
  [117]  black queenside castling
  [118]  halfmove clock / 100
"""
from typing import List

import chess
import torch

_PIECE_TO_IDX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def encode_board(board: chess.Board, history: List[chess.Board]) -> torch.Tensor:
    """
    Returns a float32 tensor of shape [8, 8, 119].

    Args:
        board:   current position
        history: list of previous positions (most-recent first), up to 7 used
    """
    planes = torch.zeros(8, 8, 119, dtype=torch.float32)

    boards = [board] + list(history[:7])

    for t, b in enumerate(boards):
        offset = t * 14
        for sq, piece in b.piece_map().items():
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            ch = _PIECE_TO_IDX[piece.piece_type]
            if piece.color == chess.BLACK:
                ch += 6
            planes[rank, file, offset + ch] = 1.0

        # Repetition planes (only meaningful for current board; approximate 0 for history)
        if t == 0:
            if b.is_repetition(2):
                planes[:, :, offset + 12] = 1.0
            if b.is_repetition(3):
                planes[:, :, offset + 13] = 1.0

    # Auxiliary planes
    planes[:, :, 112] = float(board.turn == chess.WHITE)
    planes[:, :, 113] = min(board.fullmove_number / 500.0, 1.0)
    planes[:, :, 114] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[:, :, 115] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[:, :, 116] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[:, :, 117] = float(board.has_queenside_castling_rights(chess.BLACK))
    planes[:, :, 118] = min(board.halfmove_clock / 100.0, 1.0)

    return planes
