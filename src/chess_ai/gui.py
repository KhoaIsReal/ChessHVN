"""Raylib Graphical User Interface for playing chess against the AI engine.

Features:
  - Responsive 60 FPS Raylib GUI with smooth mouse drag-and-click.
  - Interactive board with legal move hints (dots & capture rings).
  - Dynamic piece rendering using Unicode chess glyphs (with vector badge fallback).
  - Real-time Neural Value Network evaluation bar (shows win probability / score).
  - Asynchronous non-blocking bot thinking (UI never freezes during search).
  - Multiple bot engines: Huge Layer Neural Network, Material Search, Stockfish.
  - Promotion picker modal, Undo move, Flip board, New Game, Depth selector.

Launch:
  uv run python -m chess_ai.gui
  uv run python -m chess_ai.main play
  uv run python scripts/play_gui.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine
import raylib
import torch

from chess_ai.board_encoder import BoardEncoder
from chess_ai.checkpoint import CheckpointManager, find_latest_checkpoint
from chess_ai.config import SearchConfig
from chess_ai.model import ChessValueNet
from chess_ai.search import AlphaBetaSearch
from chess_ai.sources import MockStockfishEngine, find_stockfish_binary


# =============================================================================
# UI Constants & Color Palette
# =============================================================================

WINDOW_WIDTH = 1040
WINDOW_HEIGHT = 700
BOARD_SIZE = 640
SQUARE_SIZE = 80
BOARD_OFFSET_X = 52
BOARD_OFFSET_Y = 30

EVAL_BAR_X = 18
EVAL_BAR_Y = 30
EVAL_BAR_WIDTH = 20
EVAL_BAR_HEIGHT = 640

PANEL_X = 712
PANEL_Y = 30
PANEL_WIDTH = 308
PANEL_HEIGHT = 640

# Colors (RGBA tuples)
BG_DARK = (20, 22, 28, 255)
PANEL_BG = (28, 32, 42, 255)
PANEL_BORDER = (48, 54, 70, 255)
CARD_BG = (36, 42, 56, 255)
CARD_BORDER = (58, 66, 86, 255)

BOARD_LIGHT = (238, 238, 210, 255)  # Classic Cream
BOARD_DARK = (118, 150, 86, 255)    # Tournament Olive Green
BOARD_SELECT = (246, 236, 105, 180)  # Amber Glow
BOARD_LAST_MOVE = (205, 210, 106, 140)
BOARD_CHECK = (235, 64, 52, 190)

HINT_DOT = (35, 40, 50, 110)
HINT_CAPTURE = (225, 60, 50, 160)

TEXT_WHITE = (240, 242, 245, 255)
TEXT_MUTED = (140, 148, 165, 255)
TEXT_GOLD = (245, 197, 66, 255)
TEXT_GREEN = (90, 200, 100, 255)
TEXT_RED = (240, 80, 80, 255)

BTN_NORMAL = (46, 54, 72, 255)
BTN_HOVER = (64, 74, 98, 255)
BTN_ACTIVE = (76, 175, 80, 255)
BTN_TEXT = (230, 235, 245, 255)

# Unicode Chess Symbols
UNICODE_PIECES: Dict[str, str] = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656", "B": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265A", "q": "\u265B", "r": "\u265C", "b": "\u265D", "n": "\u265E", "p": "\u265F",
}


def make_vec2(x: float, y: float) -> Any:
    return raylib.ffi.new("Vector2 *", [x, y])[0]


def make_rect(x: float, y: float, w: float, h: float) -> Any:
    return raylib.ffi.new("Rectangle *", [x, y, w, h])[0]


# =============================================================================
# Raylib Chess GUI Application
# =============================================================================

class ChessGUI:
    """Complete Raylib interactive Chess interface with AI Bot."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        stockfish_path: Optional[str] = None,
        device_name: str = "auto",
    ):
        self.device_name = device_name
        self.device = torch.device("cuda" if (device_name in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
        self.encoder = BoardEncoder()

        # Game State
        self.board = chess.Board()
        self.selected_square: Optional[chess.Square] = None
        self.legal_moves_for_selected: List[chess.Move] = []
        self.last_move: Optional[chess.Move] = None
        self.flip_board: bool = False  # False: White at bottom, True: Black at bottom

        # Player Modes
        self.human_is_white: bool = True
        self.human_is_black: bool = False
        self.bot_engine_type: str = "hugenet"  # 'hugenet' | 'stockfish' | 'material'
        self.bot_depth: int = 2

        # Promotion Handling
        self.pending_promotion_move: Optional[Tuple[chess.Square, chess.Square]] = None
        self.show_promotion_modal: bool = False

        # Evaluation State
        self.current_eval: float = 0.0  # In [-1.0, 1.0] from White perspective
        self.eval_str: str = "0.00"

        # Background Bot Search Threading
        self.bot_thinking: bool = False
        self.bot_start_time: float = 0.0
        self.bot_result_queue: queue.Queue = queue.Queue()

        # Load AI Model & Engines
        self.checkpoint_path = checkpoint_path or find_latest_checkpoint()
        self.stockfish_path = stockfish_path
        self._init_models()

        # Font handles
        self.font_chess: Optional[Any] = None
        self.font_loaded: bool = False

    def _init_models(self) -> None:
        """Initialize Neural Network, AlphaBeta Searcher, and Stockfish."""
        # 1. Huge Layer Neural Network
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            try:
                p = Path(self.checkpoint_path)
                exp_dir = p.parent.parent if p.parent.name in ("checkpoints", "latest") else p.parent
                mgr = CheckpointManager(exp_dir)
                self.model, _ = mgr.load_checkpoint(str(self.checkpoint_path), device=self.device)
                print(f"[GUI] Loaded Neural Model from {self.checkpoint_path}")
            except Exception as e:
                print(f"[GUI] Could not load checkpoint ({e}). Using fresh prototype model.")
                self.model = ChessValueNet(input_dim=self.encoder.input_dim, hidden_dims=[512, 512]).to(self.device)
        else:
            self.model = ChessValueNet(input_dim=self.encoder.input_dim, hidden_dims=[512, 512]).to(self.device)

        self.model.eval()

        # Neural Searcher
        self.neural_searcher = AlphaBetaSearch(
            model=self.model,
            config=SearchConfig(depth=self.bot_depth, evaluation_mode="neural"),
            device=self.device,
            encoder=self.encoder,
        )

        # Material Baseline Searcher
        self.material_searcher = AlphaBetaSearch(
            model=None,
            config=SearchConfig(depth=self.bot_depth, evaluation_mode="material"),
            device=self.device,
            encoder=self.encoder,
        )

        # Stockfish UCI
        sf_bin = find_stockfish_binary(self.stockfish_path)
        if sf_bin:
            try:
                self.stockfish_engine = chess.engine.SimpleEngine.popen_uci(sf_bin)
                self.has_real_stockfish = True
            except Exception:
                self.stockfish_engine = MockStockfishEngine()
                self.has_real_stockfish = False
        else:
            self.stockfish_engine = MockStockfishEngine()
            self.has_real_stockfish = False

        self._update_eval()

    def _update_eval(self) -> None:
        """Compute real-time board evaluation using the neural value net."""
        try:
            with torch.no_grad():
                tensor = self.encoder.encode(self.board).unsqueeze(0).to(self.device)
                val = float(self.model(tensor).item())
                self.current_eval = max(-1.0, min(1.0, val))

                # Display string
                if abs(self.current_eval) < 0.05:
                    self.eval_str = "0.00 (Equal)"
                elif self.current_eval > 0:
                    self.eval_str = f"+{self.current_eval:.2f} (White)"
                else:
                    self.eval_str = f"{self.current_eval:.2f} (Black)"
        except Exception:
            self.current_eval = 0.0
            self.eval_str = "0.00"

    def run(self) -> None:
        """Main application lifecycle loop."""
        raylib.SetConfigFlags(raylib.FLAG_MSAA_4X_HINT | raylib.FLAG_WINDOW_HIGHDPI)
        raylib.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, b"Chess AI - Research Prototype [Raylib GUI]")
        raylib.SetTargetFPS(60)

        self._load_fonts()

        try:
            while not raylib.WindowShouldClose():
                self._handle_input()
                self._update()

                raylib.BeginDrawing()
                raylib.ClearBackground(BG_DARK)

                self._draw_board()
                self._draw_eval_bar()
                self._draw_pieces()
                self._draw_side_panel()

                if self.show_promotion_modal:
                    self._draw_promotion_modal()

                if self.board.is_game_over():
                    self._draw_game_over_banner()

                raylib.EndDrawing()
        finally:
            self._cleanup()

    def _load_fonts(self) -> None:
        """Load Segoe UI Symbol font for chess pieces if available on Windows."""
        font_path = "C:/Windows/Fonts/seguisym.ttf"
        if os.path.exists(font_path):
            try:
                codepoints = list(range(32, 128)) + list(range(0x2654, 0x2660))
                c_arr = raylib.ffi.new("int[]", codepoints)
                self.font_chess = raylib.LoadFontEx(font_path.encode("utf-8"), 56, c_arr, len(codepoints))
                self.font_loaded = True
            except Exception:
                self.font_loaded = False
        else:
            self.font_loaded = False

    def _cleanup(self) -> None:
        """Release textures, fonts, and UCI processes."""
        if self.font_loaded and self.font_chess is not None:
            try:
                raylib.UnloadFont(self.font_chess)
            except Exception:
                pass

        if self.stockfish_engine is not None and not isinstance(self.stockfish_engine, MockStockfishEngine):
            try:
                self.stockfish_engine.quit()
            except Exception:
                pass

        raylib.CloseWindow()

    # =========================================================================
    # Input Handling
    # =========================================================================

    def _handle_input(self) -> None:
        """Handle mouse clicks, board selections, and buttons."""
        if self.bot_thinking:
            return  # Ignore player clicks while bot is thinking

        mouse_pos = raylib.GetMousePosition()
        left_click = raylib.IsMouseButtonPressed(0)

        # 1. Handle Promotion Modal clicks
        if self.show_promotion_modal and self.pending_promotion_move:
            if left_click:
                self._handle_promotion_click(mouse_pos)
            return

        if not left_click:
            return

        # 2. Check Board Square Click
        board_rect = make_rect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_SIZE, BOARD_SIZE)
        if raylib.CheckCollisionPointRec(mouse_pos, board_rect):
            sq = self._get_square_from_coords(mouse_pos.x, mouse_pos.y)
            if sq is not None:
                self._handle_square_click(sq)
            return

        # 3. Check Side Panel Buttons
        self._handle_panel_buttons(mouse_pos)

    def _get_square_from_coords(self, x: float, y: float) -> Optional[chess.Square]:
        """Convert screen pixel coordinates to chess.Square index."""
        col = int((x - BOARD_OFFSET_X) // SQUARE_SIZE)
        row = int((y - BOARD_OFFSET_Y) // SQUARE_SIZE)

        if not (0 <= col < 8 and 0 <= row < 8):
            return None

        if self.flip_board:
            file_idx = 7 - col
            rank_idx = row
        else:
            file_idx = col
            rank_idx = 7 - row

        return chess.square(file_idx, rank_idx)

    def _get_square_screen_rect(self, sq: chess.Square) -> Tuple[int, int, int, int]:
        """Convert chess.Square to screen pixel rectangle (x, y, w, h)."""
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)

        if self.flip_board:
            col = 7 - file_idx
            row = rank_idx
        else:
            col = file_idx
            row = 7 - rank_idx

        x = BOARD_OFFSET_X + col * SQUARE_SIZE
        y = BOARD_OFFSET_Y + row * SQUARE_SIZE
        return (x, y, SQUARE_SIZE, SQUARE_SIZE)

    def _handle_square_click(self, sq: chess.Square) -> None:
        """Handle player selecting and moving pieces on the board."""
        if self.board.is_game_over():
            return

        # Check turn authorization
        is_human_turn = (self.board.turn == chess.WHITE and self.human_is_white) or (
            self.board.turn == chess.BLACK and self.human_is_black
        )
        if not is_human_turn:
            return

        piece = self.board.piece_at(sq)

        # If a square was already selected, try to execute move to clicked square
        if self.selected_square is not None:
            # Check for pawn promotion
            is_promo = False
            moving_piece = self.board.piece_at(self.selected_square)
            if moving_piece and moving_piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(sq)
                if (moving_piece.color == chess.WHITE and to_rank == 7) or (
                    moving_piece.color == chess.BLACK and to_rank == 0
                ):
                    promo_move = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)
                    if promo_move in self.board.legal_moves:
                        is_promo = True

            if is_promo:
                self.pending_promotion_move = (self.selected_square, sq)
                self.show_promotion_modal = True
                self.selected_square = None
                self.legal_moves_for_selected.clear()
                return

            candidate_move = chess.Move(self.selected_square, sq)
            if candidate_move in self.board.legal_moves:
                self._execute_move(candidate_move)
                self.selected_square = None
                self.legal_moves_for_selected.clear()
                return

        # Otherwise, select piece if it belongs to current player
        if piece is not None and piece.color == self.board.turn:
            self.selected_square = sq
            self.legal_moves_for_selected = [
                m for m in self.board.legal_moves if m.from_square == sq
            ]
        else:
            self.selected_square = None
            self.legal_moves_for_selected.clear()

    def _execute_move(self, move: chess.Move) -> None:
        """Execute a move on the board and update evaluation and history."""
        self.board.push(move)
        self.last_move = move
        self._update_eval()

        # Trigger Bot move if it's bot's turn now
        if not self.board.is_game_over():
            is_bot_turn = (self.board.turn == chess.WHITE and not self.human_is_white) or (
                self.board.turn == chess.BLACK and not self.human_is_black
            )
            if is_bot_turn:
                self._trigger_bot_move()

    def _handle_promotion_click(self, mouse_pos: Any) -> None:
        """Handle promotion piece selection."""
        from_sq, to_sq = self.pending_promotion_move  # type: ignore
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

        cx = BOARD_OFFSET_X + BOARD_SIZE // 2
        cy = BOARD_OFFSET_Y + BOARD_SIZE // 2
        modal_w, modal_h = 300, 90
        start_x = cx - modal_w // 2 + 15
        btn_y = cy - 10
        btn_w, btn_h = 60, 50

        for i, p_type in enumerate(pieces):
            bx = start_x + i * 70
            rec = make_rect(bx, btn_y, btn_w, btn_h)
            if raylib.CheckCollisionPointRec(mouse_pos, rec):
                move = chess.Move(from_sq, to_sq, promotion=p_type)
                if move in self.board.legal_moves:
                    self._execute_move(move)
                self.show_promotion_modal = False
                self.pending_promotion_move = None
                return

    def _handle_panel_buttons(self, mouse_pos: Any) -> None:
        """Handle clicks on side panel control buttons."""
        bx = PANEL_X + 16
        bw = PANEL_WIDTH - 32

        # 1. New Game Button
        if raylib.CheckCollisionPointRec(mouse_pos, make_rect(bx, 480, bw, 36)):
            self.board.reset()
            self.selected_square = None
            self.legal_moves_for_selected.clear()
            self.last_move = None
            self._update_eval()
            if not self.human_is_white:
                self._trigger_bot_move()
            return

        # 2. Undo Move Button (Undoes Human + Bot move)
        if raylib.CheckCollisionPointRec(mouse_pos, make_rect(bx, 524, bw, 36)):
            if len(self.board.move_stack) >= 2:
                self.board.pop()
                self.board.pop()
            elif len(self.board.move_stack) == 1:
                self.board.pop()
            self.selected_square = None
            self.legal_moves_for_selected.clear()
            self.last_move = self.board.peek() if self.board.move_stack else None
            self._update_eval()
            return

        # 3. Flip Board Button
        if raylib.CheckCollisionPointRec(mouse_pos, make_rect(bx, 568, (bw - 8) // 2, 36)):
            self.flip_board = not self.flip_board
            return

        # 4. Bot Move Now Button
        if raylib.CheckCollisionPointRec(mouse_pos, make_rect(bx + (bw - 8) // 2 + 8, 568, (bw - 8) // 2, 36)):
            if not self.board.is_game_over() and not self.bot_thinking:
                self._trigger_bot_move()
            return

        # 5. Engine Selector Buttons (HugeNet, Stockfish, Material)
        btn_y = 380
        engines = ["hugenet", "stockfish", "material"]
        col_w = (bw - 8) // 3
        for i, eng in enumerate(engines):
            rec = make_rect(bx + i * (col_w + 4), btn_y, col_w, 30)
            if raylib.CheckCollisionPointRec(mouse_pos, rec):
                self.bot_engine_type = eng
                return

        # 6. Search Depth Buttons (1, 2, 3, 4)
        depth_y = 422
        depth_col = (bw - 12) // 4
        for d in [1, 2, 3, 4]:
            rec = make_rect(bx + (d - 1) * (depth_col + 4), depth_y, depth_col, 28)
            if raylib.CheckCollisionPointRec(mouse_pos, rec):
                self.bot_depth = d
                self.neural_searcher.config.depth = d
                self.material_searcher.config.depth = d
                return

    # =========================================================================
    # Asynchronous Bot Move Computation
    # =========================================================================

    def _trigger_bot_move(self) -> None:
        """Launch bot move calculation on a background worker thread."""
        if self.bot_thinking or self.board.is_game_over():
            return

        self.bot_thinking = True
        self.bot_start_time = time.time()
        board_copy = self.board.copy()

        thread = threading.Thread(
            target=self._bot_worker,
            args=(board_copy, self.bot_engine_type, self.bot_depth),
            daemon=True,
        )
        thread.start()

    def _bot_worker(self, board_state: chess.Board, engine_type: str, depth: int) -> None:
        """Worker thread executing search without freezing GUI."""
        best_move = None
        eval_score = 0.0

        try:
            if engine_type == "stockfish":
                limit = chess.engine.Limit(depth=depth, time=0.03)
                res = self.stockfish_engine.play(board_state, limit)
                best_move = res.move if res else None
            elif engine_type == "material":
                self.material_searcher.config.depth = depth
                res = self.material_searcher.search(board_state)
                best_move = res.best_move
            else:  # 'hugenet' Neural Network
                self.neural_searcher.config.depth = depth
                res = self.neural_searcher.search(board_state)
                best_move = res.best_move

            if best_move is None or best_move not in board_state.legal_moves:
                moves = list(board_state.legal_moves)
                best_move = moves[0] if moves else None

        except Exception as e:
            print(f"[Bot Worker Error]: {e}")
            moves = list(board_state.legal_moves)
            best_move = moves[0] if moves else None

        self.bot_result_queue.put(best_move)

    def _update(self) -> None:
        """Poll background bot thread results."""
        if self.bot_thinking and not self.bot_result_queue.empty():
            bot_move = self.bot_result_queue.get()
            self.bot_thinking = False
            if bot_move and bot_move in self.board.legal_moves:
                self._execute_move(bot_move)

    # =========================================================================
    # Rendering Methods
    # =========================================================================

    def _draw_board(self) -> None:
        """Render board squares, rank/file coordinates, and highlights."""
        # Board shadow & outline
        raylib.DrawRectangle(BOARD_OFFSET_X - 4, BOARD_OFFSET_Y - 4, BOARD_SIZE + 8, BOARD_SIZE + 8, (12, 14, 18, 255))
        raylib.DrawRectangle(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_SIZE, BOARD_SIZE, (40, 44, 56, 255))

        for row in range(8):
            for col in range(8):
                x = BOARD_OFFSET_X + col * SQUARE_SIZE
                y = BOARD_OFFSET_Y + row * SQUARE_SIZE

                is_light = (row + col) % 2 == 0
                sq_color = BOARD_LIGHT if is_light else BOARD_DARK
                raylib.DrawRectangle(x, y, SQUARE_SIZE, SQUARE_SIZE, sq_color)

        # Highlight Last Move
        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                rx, ry, rw, rh = self._get_square_screen_rect(sq)
                raylib.DrawRectangle(rx, ry, rw, rh, BOARD_LAST_MOVE)

        # Highlight Selected Square
        if self.selected_square is not None:
            rx, ry, rw, rh = self._get_square_screen_rect(self.selected_square)
            raylib.DrawRectangle(rx, ry, rw, rh, BOARD_SELECT)

        # Highlight King in Check
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                kx, ky, kw, kh = self._get_square_screen_rect(king_sq)
                raylib.DrawRectangle(kx, ky, kw, kh, BOARD_CHECK)

        # Draw Legal Move Destination Hints
        for move in self.legal_moves_for_selected:
            to_sq = move.to_square
            tx, ty, tw, th = self._get_square_screen_rect(to_sq)
            cx = tx + tw // 2
            cy = ty + th // 2

            if self.board.piece_at(to_sq) is not None:
                # Capture ring
                raylib.DrawCircleLines(cx, cy, 32, HINT_CAPTURE)
                raylib.DrawCircleLines(cx, cy, 31, HINT_CAPTURE)
            else:
                # Center hint dot
                raylib.DrawCircle(cx, cy, 11, HINT_DOT)

        # Draw File & Rank coordinate labels
        for i in range(8):
            file_char = chr(ord('h' if self.flip_board else 'a') + (-i if self.flip_board else i))
            rank_char = str((i + 1) if self.flip_board else (8 - i))

            # File label (bottom)
            fx = BOARD_OFFSET_X + i * SQUARE_SIZE + SQUARE_SIZE - 14
            fy = BOARD_OFFSET_Y + BOARD_SIZE - 16
            col = BOARD_DARK if (7 + i) % 2 == 0 else BOARD_LIGHT
            raylib.DrawText(file_char.encode("utf-8"), fx, fy, 13, col)

            # Rank label (top-left)
            rx = BOARD_OFFSET_X + 4
            ry = BOARD_OFFSET_Y + i * SQUARE_SIZE + 4
            col = BOARD_LIGHT if (i % 2 == 0) else BOARD_DARK
            raylib.DrawText(rank_char.encode("utf-8"), rx, ry, 13, col)

    def _draw_pieces(self) -> None:
        """Render all active chess pieces on the board."""
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece is None:
                continue

            x, y, w, h = self._get_square_screen_rect(sq)
            cx = x + w // 2
            cy = y + h // 2

            symbol = piece.symbol()
            is_white = piece.color == chess.WHITE

            # 1. Elegant circular token badge
            token_radius = int(SQUARE_SIZE * 0.40)
            shadow_color = (15, 18, 22, 120)
            raylib.DrawCircle(cx + 2, cy + 3, token_radius, shadow_color)

            if is_white:
                raylib.DrawCircle(cx, cy, token_radius, (250, 248, 242, 255))
                raylib.DrawCircleLines(cx, cy, token_radius, (50, 52, 60, 255))
                raylib.DrawCircleLines(cx, cy, token_radius - 1, (215, 210, 200, 255))
                fg_color = (25, 28, 36, 255)
            else:
                raylib.DrawCircle(cx, cy, token_radius, (38, 40, 48, 255))
                raylib.DrawCircleLines(cx, cy, token_radius, (215, 220, 230, 255))
                raylib.DrawCircleLines(cx, cy, token_radius - 1, (75, 80, 95, 255))
                fg_color = (245, 248, 252, 255)

            # 2. Render piece symbol / glyph
            if self.font_loaded and self.font_chess is not None:
                glyph = UNICODE_PIECES.get(symbol, symbol.upper())
                glyph_bytes = glyph.encode("utf-8")
                # Measure glyph offset
                vec = raylib.MeasureTextEx(self.font_chess, glyph_bytes, 46, 0.0)
                pos = make_vec2(cx - vec.x / 2.0, cy - vec.y / 2.0 - 2)
                raylib.DrawTextEx(self.font_chess, glyph_bytes, pos, 46, 0.0, fg_color)
            else:
                # Clean vector letter fallback
                char_str = symbol.upper().encode("utf-8")
                tw = raylib.MeasureText(char_str, 26)
                raylib.DrawText(char_str, int(cx - tw / 2), int(cy - 13), 26, fg_color)

    def _draw_eval_bar(self) -> None:
        """Render the vertical winning evaluation bar on the left."""
        raylib.DrawRectangle(EVAL_BAR_X - 1, EVAL_BAR_Y - 1, EVAL_BAR_WIDTH + 2, EVAL_BAR_HEIGHT + 2, (15, 18, 24, 255))

        # Clamp eval between -1.0 and 1.0 (tanh value net output)
        val = self.current_eval
        if self.flip_board:
            val = -val

        # Map to white height ratio [0.0, 1.0]
        white_ratio = (val + 1.0) / 2.0
        white_h = int(EVAL_BAR_HEIGHT * white_ratio)
        black_h = EVAL_BAR_HEIGHT - white_h

        # Top is Black, Bottom is White
        raylib.DrawRectangle(EVAL_BAR_X, EVAL_BAR_Y, EVAL_BAR_WIDTH, black_h, (45, 48, 56, 255))
        raylib.DrawRectangle(EVAL_BAR_X, EVAL_BAR_Y + black_h, EVAL_BAR_WIDTH, white_h, (245, 245, 240, 255))

        # Divider line
        raylib.DrawLine(EVAL_BAR_X, EVAL_BAR_Y + black_h, EVAL_BAR_X + EVAL_BAR_WIDTH, EVAL_BAR_Y + black_h, (230, 70, 70, 255))

    def _draw_side_panel(self) -> None:
        """Render the dashboard containing game stats, move history, and control buttons."""
        # Panel Background
        raylib.DrawRectangle(PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_HEIGHT, PANEL_BG)
        raylib.DrawRectangleLines(PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_HEIGHT, PANEL_BORDER)

        px = PANEL_X + 16
        py = PANEL_Y + 16
        pw = PANEL_WIDTH - 32

        # 1. Header Card
        raylib.DrawRectangle(px, py, pw, 70, CARD_BG)
        raylib.DrawRectangleLines(px, py, pw, 70, CARD_BORDER)
        raylib.DrawText(b"CHESS AI - PROTOTYPE", px + 12, py + 12, 16, TEXT_GOLD)
        raylib.DrawText(b"Huge Layer Value Net + AlphaBeta", px + 12, py + 34, 12, TEXT_MUTED)

        # Compute device indicator
        dev_text = f"Device: {self.device}".encode("utf-8")
        raylib.DrawText(dev_text, px + 12, py + 50, 11, TEXT_GREEN if "cuda" in str(self.device) else TEXT_MUTED)

        # 2. Status & Evaluation Card
        status_y = py + 82
        raylib.DrawRectangle(px, status_y, pw, 96, CARD_BG)
        raylib.DrawRectangleLines(px, status_y, pw, 96, CARD_BORDER)

        turn_color = TEXT_WHITE if self.board.turn == chess.WHITE else (180, 185, 200, 255)
        turn_str = "White's Turn" if self.board.turn == chess.WHITE else "Black's Turn"
        if self.board.turn == chess.WHITE and self.human_is_white:
            turn_str += " (You)"
        elif self.board.turn == chess.BLACK and self.human_is_black:
            turn_str += " (You)"
        else:
            turn_str += " (Bot)"

        raylib.DrawText(turn_str.encode("utf-8"), px + 12, status_y + 12, 16, turn_color)

        # Neural Eval Score
        eval_label = f"Neural Eval: {self.eval_str}".encode("utf-8")
        raylib.DrawText(eval_label, px + 12, status_y + 36, 13, TEXT_GOLD)

        # Bot Thinking Indicator
        if self.bot_thinking:
            elapsed = time.time() - self.bot_start_time
            dots = "." * (int(elapsed * 3) % 4)
            think_str = f"Bot is thinking{dots} [{elapsed:.1f}s]".encode("utf-8")
            raylib.DrawText(think_str, px + 12, status_y + 64, 13, (245, 180, 60, 255))
        else:
            if self.board.is_checkmate():
                msg = b"CHECKMATE! Game Over."
                raylib.DrawText(msg, px + 12, status_y + 64, 13, TEXT_RED)
            elif self.board.is_check():
                msg = b"Check!"
                raylib.DrawText(msg, px + 12, status_y + 64, 13, TEXT_RED)
            else:
                raylib.DrawText(b"Ready for move", px + 12, status_y + 64, 13, TEXT_MUTED)

        # 3. Move History Log Card
        hist_y = status_y + 108
        raylib.DrawRectangle(px, hist_y, pw, 130, CARD_BG)
        raylib.DrawRectangleLines(px, hist_y, pw, 130, CARD_BORDER)
        raylib.DrawText(b"MOVE HISTORY", px + 12, hist_y + 10, 12, TEXT_MUTED)

        moves = list(self.board.move_stack)
        recent_moves = moves[-10:] if len(moves) > 10 else moves

        hist_lines: List[str] = []
        for i in range(0, len(recent_moves), 2):
            w_move = recent_moves[i].uci()
            b_move = recent_moves[i + 1].uci() if (i + 1 < len(recent_moves)) else ""
            hist_lines.append(f"{w_move:<7} {b_move}")

        for idx, line in enumerate(hist_lines[-4:]):
            raylib.DrawText(line.encode("utf-8"), px + 14, hist_y + 32 + idx * 22, 13, TEXT_WHITE)

        # 4. Bot Engine & Search Depth Selectors
        cfg_y = hist_y + 142
        raylib.DrawText(b"BOT ENGINE:", px, cfg_y, 11, TEXT_MUTED)

        engines = [("hugenet", "HugeNet"), ("stockfish", "Stockfish"), ("material", "Material")]
        col_w = (pw - 8) // 3
        mouse_pos = raylib.GetMousePosition()

        for i, (key, label) in enumerate(engines):
            bx = px + i * (col_w + 4)
            by = cfg_y + 18
            rec = make_rect(bx, by, col_w, 28)
            is_active = (self.bot_engine_type == key)
            is_hover = raylib.CheckCollisionPointRec(mouse_pos, rec)

            btn_col = BTN_ACTIVE if is_active else (BTN_HOVER if is_hover else BTN_NORMAL)
            raylib.DrawRectangle(bx, by, col_w, 28, btn_col)
            raylib.DrawRectangleLines(bx, by, col_w, 28, PANEL_BORDER)

            tw = raylib.MeasureText(label.encode("utf-8"), 11)
            raylib.DrawText(label.encode("utf-8"), bx + (col_w - tw) // 2, by + 8, 11, BTN_TEXT)

        # Search Depth Selector
        depth_y = cfg_y + 54
        raylib.DrawText(b"SEARCH DEPTH:", px, depth_y, 11, TEXT_MUTED)
        depth_col = (pw - 12) // 4
        for d in [1, 2, 3, 4]:
            dx = px + (d - 1) * (depth_col + 4)
            dy = depth_y + 18
            rec = make_rect(dx, dy, depth_col, 26)
            is_active = (self.bot_depth == d)
            is_hover = raylib.CheckCollisionPointRec(mouse_pos, rec)

            btn_col = BTN_ACTIVE if is_active else (BTN_HOVER if is_hover else BTN_NORMAL)
            raylib.DrawRectangle(dx, dy, depth_col, 26, btn_col)
            raylib.DrawRectangleLines(dx, dy, depth_col, 26, PANEL_BORDER)
            raylib.DrawText(str(d).encode("utf-8"), dx + depth_col // 2 - 4, dy + 6, 12, BTN_TEXT)

        # 5. Action Buttons (New Game, Undo, Flip, Bot Move)
        act_y = depth_y + 54
        self._draw_action_button(px, act_y, pw, 34, b"NEW GAME", (70, 80, 105, 255))
        self._draw_action_button(px, act_y + 42, pw, 34, b"UNDO MOVE", BTN_NORMAL)

        half_w = (pw - 8) // 2
        self._draw_action_button(px, act_y + 84, half_w, 34, b"FLIP BOARD", BTN_NORMAL)
        self._draw_action_button(px + half_w + 8, act_y + 84, half_w, 34, b"BOT MOVE", (50, 110, 80, 255))

    def _draw_action_button(self, x: int, y: int, w: int, h: int, text: bytes, base_color: tuple) -> None:
        """Render a rounded action button with hover effects."""
        mouse_pos = raylib.GetMousePosition()
        rec = make_rect(x, y, w, h)
        is_hover = raylib.CheckCollisionPointRec(mouse_pos, rec)

        col = BTN_HOVER if is_hover else base_color
        raylib.DrawRectangle(x, y, w, h, col)
        raylib.DrawRectangleLines(x, y, w, h, PANEL_BORDER)

        tw = raylib.MeasureText(text, 13)
        raylib.DrawText(text, x + (w - tw) // 2, y + (h - 13) // 2, 13, BTN_TEXT)

    def _draw_promotion_modal(self) -> None:
        """Render the modal dialog allowing the player to pick Queen, Rook, Bishop, or Knight."""
        # Dark overlay
        raylib.DrawRectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, (0, 0, 0, 180))

        cx = BOARD_OFFSET_X + BOARD_SIZE // 2
        cy = BOARD_OFFSET_Y + BOARD_SIZE // 2
        modal_w, modal_h = 320, 110
        mx = cx - modal_w // 2
        my = cy - modal_h // 2

        raylib.DrawRectangle(mx, my, modal_w, modal_h, CARD_BG)
        raylib.DrawRectangleLines(mx, my, modal_w, modal_h, TEXT_GOLD)

        raylib.DrawText(b"CHOOSE PROMOTION PIECE:", mx + 20, my + 14, 14, TEXT_GOLD)

        pieces = [("Queen", "Q"), ("Rook", "R"), ("Bishop", "B"), ("Knight", "N")]
        btn_w, btn_h = 62, 50
        start_x = mx + 20
        btn_y = my + 44
        mouse_pos = raylib.GetMousePosition()

        for i, (label, symbol) in enumerate(pieces):
            bx = start_x + i * 72
            rec = make_rect(bx, btn_y, btn_w, btn_h)
            is_hover = raylib.CheckCollisionPointRec(mouse_pos, rec)
            col = BTN_HOVER if is_hover else BTN_NORMAL

            raylib.DrawRectangle(bx, btn_y, btn_w, btn_h, col)
            raylib.DrawRectangleLines(bx, btn_y, btn_w, btn_h, PANEL_BORDER)

            tw = raylib.MeasureText(symbol.encode("utf-8"), 22)
            raylib.DrawText(symbol.encode("utf-8"), bx + (btn_w - tw) // 2, btn_y + 14, 22, TEXT_WHITE)

    def _draw_game_over_banner(self) -> None:
        """Render a banner when game concludes."""
        bx = BOARD_OFFSET_X + 40
        by = BOARD_OFFSET_Y + BOARD_SIZE // 2 - 40
        bw = BOARD_SIZE - 80
        bh = 80

        raylib.DrawRectangle(bx - 2, by - 2, bw + 4, bh + 4, (10, 12, 16, 220))
        raylib.DrawRectangle(bx, by, bw, bh, (30, 36, 50, 240))
        raylib.DrawRectangleLines(bx, by, bw, bh, TEXT_GOLD)

        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            text = f"CHECKMATE! {winner.upper()} WINS!".encode("utf-8")
            color = TEXT_GREEN if (winner == "White" and self.human_is_white) else TEXT_GOLD
        else:
            text = b"GAME DRAWN (Stalemate / Insufficient Material)"
            color = TEXT_MUTED

        tw = raylib.MeasureText(text, 18)
        raylib.DrawText(text, bx + (bw - tw) // 2, by + 28, 18, color)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Play Chess against AI Bot with Raylib GUI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish binary")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")

    args = parser.parse_args()

    gui = ChessGUI(
        checkpoint_path=args.checkpoint,
        stockfish_path=args.stockfish_path,
        device_name=args.device,
    )
    gui.run()


if __name__ == "__main__":
    main()
