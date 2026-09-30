"""Negamax search with Alpha-Beta pruning and move ordering for Chess AI."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple
import chess
import torch

from chess_ai.board_encoder import BoardEncoder
from chess_ai.config import SearchConfig
from chess_ai.evaluation import evaluate_leaf_stm


@dataclass
class NegamaxSearchResult:
    """Result container for search."""
    best_move: Optional[chess.Move]
    best_score: float
    nodes_visited: int
    cutoffs: int
    depth: int
    elapsed_seconds: float
    nodes_per_second: float


class AlphaBetaSearch:
    """Negamax search with Alpha-Beta pruning.

    Evaluates leaf positions from Side-To-Move (STM) perspective using either
    neural value network or material baseline.
    """

    INF: float = 1e6

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        config: Optional[SearchConfig] = None,
        device: Optional[torch.device] = None,
        encoder: Optional[BoardEncoder] = None,
    ):
        self.model = model
        self.config = config or SearchConfig()
        self.device = device
        self.encoder = encoder or BoardEncoder()

        self.nodes_visited: int = 0
        self.cutoffs: int = 0

    def order_moves(self, board: chess.Board) -> List[chess.Move]:
        """Order moves to maximize Alpha-Beta pruning efficiency.

        Heuristic:
          1. Captures ordered by victim value - attacker value (MVV-LVA)
          2. Pawn promotions
          3. Checks
          4. Other quiet moves
        """
        if not self.config.move_ordering:
            return list(board.legal_moves)

        def move_score(move: chess.Move) -> int:
            score = 0
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                cap_val = captured.piece_type if captured else 1
                att_val = attacker.piece_type if attacker else 1
                score += 100 * cap_val - att_val
            if move.promotion:
                score += 80
            if board.gives_check(move):
                score += 40
            return score

        moves = list(board.legal_moves)
        moves.sort(key=move_score, reverse=True)
        return moves

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        use_pruning: bool = True,
    ) -> float:
        self.nodes_visited += 1

        # Check terminal or leaf depth
        if board.is_game_over() or depth <= 0:
            return evaluate_leaf_stm(
                board=board,
                model=self.model,
                device=self.device,
                encoder=self.encoder,
                evaluation_mode=self.config.evaluation_mode,
            )

        best_score = -self.INF
        moves = self.order_moves(board)

        for move in moves:
            board.push(move)
            score = -self._negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                use_pruning=use_pruning,
            )
            board.pop()

            if score > best_score:
                best_score = score

            if score > alpha:
                alpha = score

            if use_pruning and alpha >= beta:
                self.cutoffs += 1
                break

        return best_score

    def search(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
        use_pruning: bool = True,
    ) -> NegamaxSearchResult:
        """Find the best move from the root position.

        Returns:
          NegamaxSearchResult containing best move, best score, and search stats.
        """
        search_depth = depth if depth is not None else self.config.depth
        self.nodes_visited = 0
        self.cutoffs = 0

        legal_moves = self.order_moves(board)
        if not legal_moves:
            # Game is over
            score = evaluate_leaf_stm(
                board,
                model=self.model,
                device=self.device,
                encoder=self.encoder,
                evaluation_mode=self.config.evaluation_mode,
            )
            return NegamaxSearchResult(
                best_move=None,
                best_score=score,
                nodes_visited=1,
                cutoffs=0,
                depth=0,
                elapsed_seconds=0.0,
                nodes_per_second=0.0,
            )

        start_time = time.perf_counter()
        best_move: Optional[chess.Move] = legal_moves[0]
        best_score = -self.INF
        alpha = -self.INF
        beta = self.INF

        for move in legal_moves:
            board.push(move)
            score = -self._negamax(
                board,
                search_depth - 1,
                -beta,
                -alpha,
                use_pruning=use_pruning,
            )
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if use_pruning and alpha >= beta:
                self.cutoffs += 1
                break

        elapsed = max(1e-6, time.perf_counter() - start_time)
        nps = self.nodes_visited / elapsed

        return NegamaxSearchResult(
            best_move=best_move,
            best_score=best_score,
            nodes_visited=self.nodes_visited,
            cutoffs=self.cutoffs,
            depth=search_depth,
            elapsed_seconds=elapsed,
            nodes_per_second=nps,
        )
