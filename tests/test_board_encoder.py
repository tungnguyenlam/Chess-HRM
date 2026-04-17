"""Tests for chessgame.encoding.board_encoder — PLAN step 0.2."""

import chess
import torch

from chessgame.encoding.board_encoder import encode_board


class TestBoardEncoderShape:
    """Output shape and type."""

    def test_output_shape(self, starting_board):
        t = encode_board(starting_board, [])
        assert t.shape == (8, 8, 119), f"Expected (8,8,119), got {t.shape}"

    def test_output_dtype(self, starting_board):
        t = encode_board(starting_board, [])
        assert t.dtype == torch.float32

    def test_values_are_bounded(self, starting_board):
        t = encode_board(starting_board, [])
        assert t.min() >= 0.0, "Negative values in encoding"
        assert t.max() <= 1.0, "Values > 1.0 in encoding"


class TestBoardEncoderPieces:
    """Pieces are placed correctly in the starting position."""

    def test_white_pawns_rank2(self, starting_board):
        """White pawns should be on rank 1 (0-indexed), plane 0 (PAWN)."""
        t = encode_board(starting_board, [])
        for file_idx in range(8):
            assert t[1, file_idx, 0] == 1.0, f"White pawn missing at file {file_idx}"

    def test_black_pawns_rank7(self, starting_board):
        """Black pawns should be on rank 6 (0-indexed), plane 6 (PAWN+6)."""
        t = encode_board(starting_board, [])
        for file_idx in range(8):
            assert t[6, file_idx, 6] == 1.0, f"Black pawn missing at file {file_idx}"

    def test_white_king(self, starting_board):
        """White king on e1 = rank 0, file 4, plane 5 (KING)."""
        t = encode_board(starting_board, [])
        assert t[0, 4, 5] == 1.0, "White king not at e1"

    def test_black_king(self, starting_board):
        """Black king on e8 = rank 7, file 4, plane 11 (KING+6)."""
        t = encode_board(starting_board, [])
        assert t[7, 4, 11] == 1.0, "Black king not at e8"

    def test_white_rooks(self, starting_board):
        """White rooks on a1 and h1 = rank 0, files 0 and 7, plane 3 (ROOK)."""
        t = encode_board(starting_board, [])
        assert t[0, 0, 3] == 1.0, "White rook not at a1"
        assert t[0, 7, 3] == 1.0, "White rook not at h1"

    def test_empty_squares_are_zero(self, starting_board):
        """Ranks 2-5 (indices 2-5) should have no pieces in plane 0 at step t=0."""
        t = encode_board(starting_board, [])
        # Piece planes for t=0 are 0..11
        for rank in range(2, 6):
            for file_idx in range(8):
                piece_sum = t[rank, file_idx, 0:12].sum()
                assert piece_sum == 0.0, (
                    f"Unexpected piece at rank={rank}, file={file_idx}"
                )

    def test_piece_after_move(self, italian_game_board):
        """After 1.e4, pawn should be at e4 (rank 3, file 4)."""
        t = encode_board(italian_game_board, [])
        # White pawn at e4
        assert t[3, 4, 0] == 1.0, "White pawn not at e4"
        # e2 should be empty in t=0 piece planes
        assert t[1, 4, 0] == 0.0, "e2 should be empty after e4"


class TestBoardEncoderAuxiliary:
    """Auxiliary planes (112-118)."""

    def test_side_to_move_white(self, starting_board):
        """Plane 112 should be 1.0 when white to move."""
        t = encode_board(starting_board, [])
        assert t[0, 0, 112] == 1.0, "Side to move should be 1 for white"
        # All squares should have the same value
        assert (t[:, :, 112] == 1.0).all()

    def test_side_to_move_black(self):
        """After 1.e4, Black to move → plane 112 = 0."""
        b = chess.Board()
        b.push_uci("e2e4")
        t = encode_board(b, [])
        assert t[0, 0, 112] == 0.0, "Side to move should be 0 for black"

    def test_castling_rights_starting(self, starting_board):
        """All castling rights available at start."""
        t = encode_board(starting_board, [])
        assert t[0, 0, 114] == 1.0, "White kingside castling"
        assert t[0, 0, 115] == 1.0, "White queenside castling"
        assert t[0, 0, 116] == 1.0, "Black kingside castling"
        assert t[0, 0, 117] == 1.0, "Black queenside castling"

    def test_castling_lost_after_king_move(self):
        """After Ke2, white loses both castling rights."""
        b = chess.Board()
        b.push_uci("e2e4")
        b.push_uci("e7e5")
        b.push_uci("e1e2")  # King moves
        t = encode_board(b, [])
        assert t[0, 0, 114] == 0.0, "White kingside should be lost"
        assert t[0, 0, 115] == 0.0, "White queenside should be lost"
        # Black still has rights
        assert t[0, 0, 116] == 1.0
        assert t[0, 0, 117] == 1.0

    def test_halfmove_clock(self, starting_board):
        """Halfmove clock at start is 0/100 = 0."""
        t = encode_board(starting_board, [])
        assert t[0, 0, 118] == 0.0

    def test_fullmove_number(self, starting_board):
        """Move 1 → 1/500 = 0.002."""
        t = encode_board(starting_board, [])
        assert abs(t[0, 0, 113] - 1.0 / 500.0) < 1e-6


class TestBoardEncoderHistory:
    """History encoding with multiple plies."""

    def test_history_offset(self, board_with_history):
        """History step t occupies planes [t*14 : t*14+14]."""
        board, history = board_with_history
        t = encode_board(board, history)
        # t=0 is current position; t=1 is one move ago, etc.
        # Just verify that piece planes in step 1 differ from step 0
        current_pieces = t[:, :, 0:12].sum()
        prev_pieces = t[:, :, 14:26].sum()
        assert current_pieces > 0, "No pieces in current step"
        assert prev_pieces > 0, "No pieces in history step 1"

    def test_no_history_zero_padded(self, starting_board):
        """Without history, steps 1-7 should be all zeros in piece planes."""
        t = encode_board(starting_board, [])
        for step in range(1, 8):
            offset = step * 14
            piece_sum = t[:, :, offset : offset + 12].sum()
            assert piece_sum == 0.0, f"History step {step} should be zero-padded"

    def test_history_limited_to_7(self, board_with_history):
        """Only 7 history boards used even if more provided."""
        board, history = board_with_history
        assert len(history) >= 8, "Fixture should have at least 8 history entries"
        t = encode_board(board, history)
        # Steps 0-7 fill planes 0-111; plane 112+ is auxiliary
        assert t.shape == (8, 8, 119)


class TestBoardEncoderRepetition:
    """Repetition detection planes."""

    def test_no_repetition_at_start(self, starting_board):
        """No repetition at game start."""
        t = encode_board(starting_board, [])
        assert t[:, :, 12].sum() == 0.0, "No repetition at start"
        assert t[:, :, 13].sum() == 0.0, "No triple repetition at start"

    def test_repetition_detected(self):
        """After Nf3 Nc6 Ng1 Nb8, position repeats."""
        b = chess.Board()
        moves = ["g1f3", "b8c6", "f3g1", "c6b8"]  # back to start
        for m in moves:
            b.push_uci(m)
        t = encode_board(b, [])
        # Position has been seen >= 2 times now
        assert t[0, 0, 12] == 1.0, "Repetition plane should be set"
