"""Evaluation functions for chess positions supporting neural batch inference and material baseline."""

from __future__ import annotations

from typing import List, Optional, Sequence
import chess
import torch

from chess_ai.board_encoder import BoardEncoder


_MATERIAL_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def evaluate_material(board: chess.Board) -> float:
    """Evaluate position based on material balance from side-to-move perspective.

    Returns value in [-1.0, 1.0].
    """
    if board.is_checkmate():
        return -1.0  # Side to move is checkmated

    white_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in _MATERIAL_VALUES.items())
    black_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in _MATERIAL_VALUES.items())

    # Raw material difference from White perspective:
    diff = white_mat - black_mat
    # Normalized by roughly maximum material advantage (~30 points)
    normalized = max(-1.0, min(1.0, diff / 30.0))

    # Convert to side-to-move perspective
    return normalized if board.turn == chess.WHITE else -normalized


def evaluate_positions(
    model: torch.nn.Module,
    positions: Sequence[chess.Board],
    device: Optional[torch.device] = None,
    encoder: Optional[BoardEncoder] = None,
) -> List[float]:
    """Batch inference evaluating positions from White perspective.

    Args:
      model: ChessValueNet or equivalent
      positions: Sequence of chess.Board instances
      device: torch.device (defaults to model's parameter device)
      encoder: BoardEncoder instance

    Returns:
      List of float values in [-1.0, 1.0] from White's perspective.
    """
    if not positions:
        return []

    if encoder is None:
        encoder = BoardEncoder()

    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            batch_tensor = encoder.encode_batch(positions).to(device)
            values_tensor = model(batch_tensor)  # [B, 1]
            values = values_tensor.squeeze(-1).tolist()
            if isinstance(values, float):
                values = [values]
            return values
    finally:
        if was_training:
            model.train()


def evaluate_leaf_stm(
    board: chess.Board,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    encoder: Optional[BoardEncoder] = None,
    evaluation_mode: str = "neural",
) -> float:
    """Evaluate leaf position from the Side-To-Move (STM) perspective for Negamax.

    Terminal handling:
      Checkmate: -1.0 (loss for side to move)
      Draw (stalemate / 50-move / 3-fold): 0.0

    Perspective mapping:
      Model returns V_white(s) in [-1, +1].
      If board.turn == WHITE: return +V_white(s)
      If board.turn == BLACK: return -V_white(s)
    """
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0

    if evaluation_mode == "material" or model is None:
        return evaluate_material(board)

    # Neural evaluation:
    white_vals = evaluate_positions(model, [board], device=device, encoder=encoder)
    white_v = white_vals[0] if white_vals else 0.0

    # Convert to STM perspective
    return white_v if board.turn == chess.WHITE else -white_v
