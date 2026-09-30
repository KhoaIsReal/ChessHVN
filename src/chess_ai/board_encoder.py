"""Deterministic chess board encoder converting python-chess Board to PyTorch tensors."""

from __future__ import annotations

from typing import List, Sequence
import chess
import torch


class BoardEncoder:
    """Encodes a chess.Board state into a deterministic PyTorch tensor.

    Encoding scheme:
      Planes 0-5:   White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
      Planes 6-11:  Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
      Plane 12:     Side to move (1.0 if White, 0.0 if Black)
      Plane 13:     White Kingside castling right (1.0 if available, 0.0 otherwise)
      Plane 14:     White Queenside castling right (1.0 if available, 0.0 otherwise)
      Plane 15:     Black Kingside castling right (1.0 if available, 0.0 otherwise)
      Plane 16:     Black Queenside castling right (1.0 if available, 0.0 otherwise)
      Plane 17:     En passant target square (1.0 at target square, 0.0 elsewhere)
      Plane 18:     Halfmove clock (normalized min(1.0, halfmove_clock / 100.0))

    Total channels: 19 planes of 8x8 squares.
    Flattened input_dim = 19 * 8 * 8 = 1216.
    """

    NUM_CHANNELS: int = 19
    BOARD_SIZE: int = 8
    SQUARES: int = 64
    INPUT_DIM: int = NUM_CHANNELS * SQUARES  # 1216

    _PIECE_MAP = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    @property
    def input_dim(self) -> int:
        """Dynamic input dimension derived from channel count and board geometry."""
        return self.INPUT_DIM

    @property
    def num_channels(self) -> int:
        return self.NUM_CHANNELS

    def encode_planes(self, board: chess.Board) -> torch.Tensor:
        """Encode a single board into a [19, 8, 8] float32 tensor."""
        planes = torch.zeros((self.NUM_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE), dtype=torch.float32)

        # 1. Piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                offset = 0 if piece.color == chess.WHITE else 6
                plane_idx = offset + self._PIECE_MAP[piece.piece_type]
                planes[plane_idx, rank, file] = 1.0

        # 2. Side to move
        if board.turn == chess.WHITE:
            planes[12].fill_(1.0)

        # 3. Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[13].fill_(1.0)
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[14].fill_(1.0)
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[15].fill_(1.0)
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[16].fill_(1.0)

        # 4. En passant square
        if board.ep_square is not None:
            ep_rank = chess.square_rank(board.ep_square)
            ep_file = chess.square_file(board.ep_square)
            planes[17, ep_rank, ep_file] = 1.0

        # 5. Halfmove clock
        planes[18].fill_(min(1.0, float(board.halfmove_clock) / 100.0))

        return planes

    def encode(self, board: chess.Board) -> torch.Tensor:
        """Encode a single board into a flattened [input_dim] float32 tensor."""
        planes = self.encode_planes(board)
        return planes.view(self.INPUT_DIM)

    def encode_batch(self, boards: Sequence[chess.Board]) -> torch.Tensor:
        """Encode a sequence of boards into a batch tensor of shape [B, input_dim]."""
        if not boards:
            return torch.empty((0, self.INPUT_DIM), dtype=torch.float32)
        encoded_list = [self.encode(b) for b in boards]
        return torch.stack(encoded_list, dim=0)
