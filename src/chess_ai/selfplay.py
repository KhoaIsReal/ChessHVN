"""Self-play game generation and trajectory collection for Chess AI."""

from __future__ import annotations

import random
from typing import Optional
import chess
import torch

from chess_ai.board_encoder import BoardEncoder
from chess_ai.config import SearchConfig, SelfPlayConfig
from chess_ai.replay import Episode, Transition
from chess_ai.search import AlphaBetaSearch


class SelfPlayEngine:
    """Generates games via model self-play with Alpha-Beta search."""

    def __init__(
        self,
        model: torch.nn.Module,
        encoder: Optional[BoardEncoder] = None,
        search_config: Optional[SearchConfig] = None,
        selfplay_config: Optional[SelfPlayConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.encoder = encoder or BoardEncoder()
        self.search_config = search_config or SearchConfig()
        self.selfplay_config = selfplay_config or SelfPlayConfig()
        self.device = device or next(model.parameters()).device

        self.searcher = AlphaBetaSearch(
            model=self.model,
            config=self.search_config,
            device=self.device,
            encoder=self.encoder,
        )

    def play_game(self) -> Episode:
        """Play a single game of self-play and return recorded Episode."""
        board = chess.Board()
        transitions = []
        move_count = 0

        while not board.is_game_over() and move_count < self.selfplay_config.max_moves_per_game:
            current_fen = board.fen()
            state_tensor = self.encoder.encode(board).cpu()
            turn = board.turn

            # Select move
            if move_count < self.selfplay_config.exploration_moves:
                # Opening exploration
                move = random.choice(list(board.legal_moves))
            else:
                result = self.searcher.search(board, depth=self.search_config.depth)
                move = result.best_move or random.choice(list(board.legal_moves))

            board.push(move)
            move_count += 1

            next_state_tensor = self.encoder.encode(board).cpu()
            done = board.is_game_over() or (move_count >= self.selfplay_config.max_moves_per_game)

            transitions.append(
                Transition(
                    state=state_tensor,
                    action=move.uci(),
                    reward=0.0,  # Will be updated at game termination
                    next_state=next_state_tensor,
                    done=done,
                    turn=turn,
                    fen=current_fen,
                )
            )

        # Determine terminal result and reward (from White's perspective)
        if board.is_checkmate():
            # If side to move was mated, the OTHER side won
            if board.turn == chess.WHITE:
                result_str = "0-1"
                winner = "black"
                white_reward = -1.0
            else:
                result_str = "1-0"
                winner = "white"
                white_reward = 1.0
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw() or move_count >= self.selfplay_config.max_moves_per_game:
            result_str = "1/2-1/2"
            winner = "draw"
            white_reward = 0.0
        else:
            result_str = "*"
            winner = "draw"
            white_reward = 0.0

        episode = Episode(
            transitions=transitions,
            result=result_str,
            winner=winner,
        )
        episode.assign_terminal_rewards(white_reward)

        return episode
