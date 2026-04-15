"""
AlphaZero 4672-move encoding.

Index = from_square * 73 + move_type

move_type breakdown (73 total per square):
  0–55   queen-type moves:  direction(0–7) * 7 + distance(0–6)
           directions: N NE E SE S SW W NW
           distance 0 = 1 square, distance 6 = 7 squares
  56–63  knight moves: 8 (dr,df) deltas in fixed order
  64–72  underpromotions: piece(0–2) * 3 + dir(0–2)
           pieces: rook bishop knight
           dirs: left-capture push right-capture
           (queen promotions are encoded as queen-type moves)
"""
from typing import Optional, Dict, Tuple

import chess
import torch

# --- Direction tables ---

_QUEEN_DIRS: list[Tuple[int, int]] = [
    (1, 0),    # 0 N
    (1, 1),    # 1 NE
    (0, 1),    # 2 E
    (-1, 1),   # 3 SE
    (-1, 0),   # 4 S
    (-1, -1),  # 5 SW
    (0, -1),   # 6 W
    (1, -1),   # 7 NW
]
_DIR_TO_IDX: Dict[Tuple[int, int], int] = {d: i for i, d in enumerate(_QUEEN_DIRS)}

_KNIGHT_DELTAS: list[Tuple[int, int]] = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]
_KNIGHT_TO_IDX: Dict[Tuple[int, int], int] = {d: i for i, d in enumerate(_KNIGHT_DELTAS)}

_PROMO_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

NUM_MOVES = 4672  # 64 * 73


def encode_move(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)

    dr = to_rank - from_rank
    df = to_file - from_file

    abs_dr, abs_df = abs(dr), abs(df)

    # Knight move
    if (abs_dr, abs_df) in {(2, 1), (1, 2)}:
        move_type = 56 + _KNIGHT_TO_IDX[(dr, df)]

    # Underpromotion (queen promotion falls through to queen-type below)
    elif move.promotion and move.promotion != chess.QUEEN:
        piece_idx = _PROMO_PIECES.index(move.promotion)
        dir_idx = df + 1  # -1,0,+1 → 0,1,2
        move_type = 64 + piece_idx * 3 + dir_idx

    # Queen-type (sliding + queen promotions)
    else:
        max_dist = max(abs_dr, abs_df)
        sign_dr = (1 if dr > 0 else -1 if dr < 0 else 0)
        sign_df = (1 if df > 0 else -1 if df < 0 else 0)
        direction = _DIR_TO_IDX[(sign_dr, sign_df)]
        distance = max_dist - 1  # 0-indexed
        move_type = direction * 7 + distance

    return from_sq * 73 + move_type


def decode_move(idx: int, board: Optional[chess.Board] = None) -> Optional[chess.Move]:
    """Decode a move index.  Returns None if the index maps off the board."""
    if not (0 <= idx < NUM_MOVES):
        return None

    from_sq = idx // 73
    move_type = idx % 73

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)

    if move_type < 56:  # Queen-type
        direction = move_type // 7
        distance = move_type % 7 + 1
        dr, df = _QUEEN_DIRS[direction]
        to_rank = from_rank + dr * distance
        to_file = from_file + df * distance
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            return None
        to_sq = chess.square(to_file, to_rank)
        # Auto-promote to queen when pawn reaches back rank
        promotion = None
        if board is not None:
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and to_rank == 7) or \
                   (piece.color == chess.BLACK and to_rank == 0):
                    promotion = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promotion)

    elif move_type < 64:  # Knight
        dr, df = _KNIGHT_DELTAS[move_type - 56]
        to_rank = from_rank + dr
        to_file = from_file + df
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            return None
        return chess.Move(from_sq, chess.square(to_file, to_rank))

    else:  # Underpromotion (64–72)
        sub = move_type - 64
        piece_idx = sub // 3
        dir_idx = sub % 3
        df = dir_idx - 1  # -1, 0, +1
        # Determine push direction from piece color
        if board is not None:
            piece = board.piece_at(from_sq)
            dr = 1 if (piece and piece.color == chess.WHITE) else -1
        else:
            dr = 1 if from_rank < 4 else -1  # heuristic
        to_rank = from_rank + dr
        to_file = from_file + df
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            return None
        return chess.Move(from_sq, chess.square(to_file, to_rank),
                          promotion=_PROMO_PIECES[piece_idx])


def legal_mask(board: chess.Board) -> torch.BoolTensor:
    """Boolean tensor of shape [4672] — True where the move is legal."""
    mask = torch.zeros(NUM_MOVES, dtype=torch.bool)
    for move in board.legal_moves:
        idx = encode_move(move)
        if 0 <= idx < NUM_MOVES:
            mask[idx] = True
    return mask
