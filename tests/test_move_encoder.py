"""Tests for chessgame.encoding.move_encoder — PLAN step 0.3."""
import chess
import torch

from chessgame.encoding.move_encoder import (
    encode_move, decode_move, legal_mask, NUM_MOVES,
)


class TestMoveEncoderConstants:
    """Basic constants."""

    def test_num_moves(self):
        assert NUM_MOVES == 4672, f"Expected 4672, got {NUM_MOVES}"


class TestMoveEncoderRoundTrip:
    """Encode then decode should produce the original move."""

    def test_roundtrip_e2e4(self, starting_board):
        move = chess.Move.from_uci("e2e4")
        idx = encode_move(move)
        decoded = decode_move(idx, starting_board)
        assert decoded == move, f"Roundtrip failed: {move} → {idx} → {decoded}"

    def test_roundtrip_knight_move(self, starting_board):
        move = chess.Move.from_uci("g1f3")
        idx = encode_move(move)
        decoded = decode_move(idx, starting_board)
        assert decoded == move

    def test_roundtrip_all_legal_starting(self, starting_board):
        """Every legal move in starting position should round-trip."""
        for move in starting_board.legal_moves:
            idx = encode_move(move)
            assert 0 <= idx < NUM_MOVES, f"Index out of range for {move}: {idx}"
            decoded = decode_move(idx, starting_board)
            assert decoded == move, f"Roundtrip failed for {move}: idx={idx}, decoded={decoded}"

    def test_roundtrip_all_legal_italian(self, italian_game_board):
        """Every legal move in Italian Game should round-trip."""
        for move in italian_game_board.legal_moves:
            idx = encode_move(move)
            assert 0 <= idx < NUM_MOVES
            decoded = decode_move(idx, italian_game_board)
            assert decoded == move, f"Roundtrip failed for {move}"


class TestMoveEncoderPromotions:
    """Promotion handling."""

    def test_queen_promotion(self, promotion_board):
        """e7e8q should encode as queen-type move."""
        move = chess.Move.from_uci("e7e8q")
        idx = encode_move(move)
        assert 0 <= idx < NUM_MOVES
        decoded = decode_move(idx, promotion_board)
        assert decoded is not None
        assert decoded.promotion == chess.QUEEN

    def test_knight_underpromotion(self, promotion_board):
        """e7e8n is an underpromotion."""
        move = chess.Move.from_uci("e7e8n")
        idx = encode_move(move)
        assert 0 <= idx < NUM_MOVES
        decoded = decode_move(idx, promotion_board)
        assert decoded is not None
        assert decoded.promotion == chess.KNIGHT

    def test_rook_underpromotion(self, promotion_board):
        move = chess.Move.from_uci("e7e8r")
        idx = encode_move(move)
        decoded = decode_move(idx, promotion_board)
        assert decoded is not None
        assert decoded.promotion == chess.ROOK

    def test_bishop_underpromotion(self, promotion_board):
        move = chess.Move.from_uci("e7e8b")
        idx = encode_move(move)
        decoded = decode_move(idx, promotion_board)
        assert decoded is not None
        assert decoded.promotion == chess.BISHOP


class TestMoveEncoderCastling:
    """Castling is encoded as a king move."""

    def test_white_kingside_castle(self):
        b = chess.Board()
        # Clear pieces between king and rook
        b.remove_piece_at(chess.F1)
        b.remove_piece_at(chess.G1)
        move = chess.Move.from_uci("e1g1")  # O-O
        idx = encode_move(move)
        assert 0 <= idx < NUM_MOVES
        decoded = decode_move(idx, b)
        assert decoded is not None
        assert decoded.from_square == chess.E1
        assert decoded.to_square == chess.G1

    def test_white_queenside_castle(self):
        b = chess.Board()
        b.remove_piece_at(chess.D1)
        b.remove_piece_at(chess.C1)
        b.remove_piece_at(chess.B1)
        move = chess.Move.from_uci("e1c1")  # O-O-O
        idx = encode_move(move)
        assert 0 <= idx < NUM_MOVES
        decoded = decode_move(idx, b)
        assert decoded is not None
        assert decoded.from_square == chess.E1
        assert decoded.to_square == chess.C1


class TestMoveEncoderDirections:
    """Queen-type directional encoding."""

    def test_north_move(self):
        """Rook moving north: a1a4 → direction 0 (N), distance 2 (3 squares = index 2)."""
        move = chess.Move.from_uci("a1a4")
        idx = encode_move(move)
        from_sq = chess.A1
        move_type = idx % 73
        assert move_type // 7 == 0, f"Direction should be 0 (N), got {move_type // 7}"
        assert move_type % 7 == 2, f"Distance should be 2 (3 squares), got {move_type % 7}"

    def test_diagonal_move(self):
        """Bishop moving NE: a1d4 → direction 1 (NE), distance 2."""
        move = chess.Move.from_uci("a1d4")
        idx = encode_move(move)
        move_type = idx % 73
        assert move_type // 7 == 1, f"Direction should be 1 (NE), got {move_type // 7}"
        assert move_type % 7 == 2, f"Distance should be 2, got {move_type % 7}"


class TestMoveEncoderKnightDeltas:
    """Knight move encoding uses indices 56-63."""

    def test_knight_all_8_directions(self):
        """From d4, a knight has 8 possible destinations."""
        from_sq = chess.D4
        deltas = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1),
        ]
        for i, (dr, df) in enumerate(deltas):
            to_rank = chess.square_rank(from_sq) + dr
            to_file = chess.square_file(from_sq) + df
            if 0 <= to_rank <= 7 and 0 <= to_file <= 7:
                to_sq = chess.square(to_file, to_rank)
                move = chess.Move(from_sq, to_sq)
                idx = encode_move(move)
                move_type = idx % 73
                assert 56 <= move_type <= 63, f"Knight move should be 56-63, got {move_type}"
                assert move_type - 56 == i, f"Knight index should be {i}, got {move_type - 56}"


class TestLegalMask:
    """legal_mask() function."""

    def test_starting_position_20_moves(self, starting_board):
        """Starting position has exactly 20 legal moves."""
        mask = legal_mask(starting_board)
        assert mask.shape == (NUM_MOVES,)
        assert mask.dtype == torch.bool
        assert mask.sum().item() == 20, f"Expected 20 legal moves, got {mask.sum()}"

    def test_mask_covers_all_legal(self, italian_game_board):
        """Every legal move should have mask[idx] == True."""
        mask = legal_mask(italian_game_board)
        for move in italian_game_board.legal_moves:
            idx = encode_move(move)
            assert mask[idx], f"Legal move {move} not in mask (idx={idx})"

    def test_checkmate_no_legal_moves(self):
        """Scholar's mate position — no legal moves for Black."""
        b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
        # This is actually Black just delivered checkmate; let's use fool's mate
        b = chess.Board()
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            b.push_uci(uci)
        # White is in checkmate
        mask = legal_mask(b)
        assert mask.sum().item() == 0, "Checkmate should have 0 legal moves"
