"""Unit tests for BoardEncoder."""

import chess
import pytest
import torch

from chess_ai.board_encoder import BoardEncoder


def test_encoder_deterministic():
    encoder = BoardEncoder()
    board1 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    board2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    tensor1 = encoder.encode(board1)
    tensor2 = encoder.encode(board2)

    assert tensor1.shape == (encoder.input_dim,)
    assert torch.all(tensor1 == tensor2), "Encoder must be deterministic for identical positions"


def test_encoder_different_positions():
    encoder = BoardEncoder()
    board_start = chess.Board()
    board_moved = chess.Board()
    board_moved.push_san("e4")

    t1 = encoder.encode(board_start)
    t2 = encoder.encode(board_moved)

    assert not torch.all(t1 == t2), "Moved board must produce different tensor encoding"


def test_encoder_batch():
    encoder = BoardEncoder()
    boards = [chess.Board(), chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")]
    batch = encoder.encode_batch(boards)

    assert batch.shape == (2, encoder.input_dim)
    assert batch.dtype == torch.float32


def test_encoder_planes_metadata():
    encoder = BoardEncoder()
    board = chess.Board()

    planes = encoder.encode_planes(board)
    assert planes.shape == (encoder.num_channels, 8, 8)

    # White to move: plane 12 should be all 1.0
    assert torch.all(planes[12] == 1.0)
    # White castling rights: planes 13, 14 should be all 1.0
    assert torch.all(planes[13] == 1.0)
    assert torch.all(planes[14] == 1.0)
    # Black castling rights: planes 15, 16 should be all 1.0
    assert torch.all(planes[15] == 1.0)
    assert torch.all(planes[16] == 1.0)
    # No en passant initially: plane 17 should be all 0.0
    assert torch.all(planes[17] == 0.0)

    # After e4, en passant square is e3
    board.push_san("e4")
    planes_after = encoder.encode_planes(board)
    # Black to move: plane 12 should be all 0.0
    assert torch.all(planes_after[12] == 0.0)
    # En passant square e3 (rank 2, file 4 in 0-indexed)
    assert planes_after[17, 2, 4] == 1.0
