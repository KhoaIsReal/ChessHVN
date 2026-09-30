"""Unit tests for AlphaBetaSearch vs unpruned Negamax and terminal states."""

import chess
import pytest
import torch

from chess_ai.board_encoder import BoardEncoder
from chess_ai.config import SearchConfig
from chess_ai.model import ChessValueNet
from chess_ai.search import AlphaBetaSearch


def test_alphabeta_vs_unpruned_material():
    """Verify Alpha-Beta pruning returns identical best score as unpruned Negamax."""
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    searcher = AlphaBetaSearch(config=SearchConfig(depth=2, evaluation_mode="material"))

    # 1. Search with Alpha-Beta pruning enabled
    res_pruned = searcher.search(board, depth=2, use_pruning=True)

    # 2. Search unpruned (full minimax tree)
    res_unpruned = searcher.search(board, depth=2, use_pruning=False)

    assert abs(res_pruned.best_score - res_unpruned.best_score) < 1e-5, (
        f"Pruned score {res_pruned.best_score} != Unpruned score {res_unpruned.best_score}"
    )
    # Pruned search must visit fewer or equal nodes
    assert res_pruned.nodes_visited <= res_unpruned.nodes_visited
    assert res_pruned.cutoffs > 0


def test_alphabeta_vs_unpruned_neural():
    """Verify equivalence between pruned and unpruned search using neural evaluator."""
    encoder = BoardEncoder()
    model = ChessValueNet(input_dim=encoder.input_dim, hidden_dims=[32, 32])
    searcher = AlphaBetaSearch(model=model, encoder=encoder, config=SearchConfig(depth=2, evaluation_mode="neural"))

    # Small endgame board for quick evaluation
    board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")

    res_pruned = searcher.search(board, depth=2, use_pruning=True)
    res_unpruned = searcher.search(board, depth=2, use_pruning=False)

    assert abs(res_pruned.best_score - res_unpruned.best_score) < 1e-4
    assert res_pruned.nodes_visited <= res_unpruned.nodes_visited


def test_search_terminal_checkmate():
    """Fool's Mate position: Black has just checkmated White."""
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()

    searcher = AlphaBetaSearch(config=SearchConfig(depth=2, evaluation_mode="material"))
    res = searcher.search(board)

    # Side to move (White) is checkmated, score must be -1.0 (loss)
    assert res.best_score == -1.0
    assert res.best_move is None
