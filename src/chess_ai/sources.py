"""Multi-source concurrent data streaming for training Chess Value Networks.

Supports training simultaneously from multiple data streams:
  1. SelfPlaySource: Online / concurrent multi-worker self-play games with search.
  2. PGNSource: Real human / Grandmaster games parsed from PGN files.
  3. FENSource: Static position datasets / tactical puzzle benchmarks with labels.
  4. MultiSourceDataManager: Manages concurrent ingestion, source weighting, and mixed batch sampling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import io
import math
from pathlib import Path
import queue
import random
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import chess
import chess.engine
import chess.pgn
import torch

from chess_ai.board_encoder import BoardEncoder
from chess_ai.config import MultiSourceConfig, SearchConfig, SelfPlayConfig
from chess_ai.replay import Episode, ReplayBuffer, Transition
from chess_ai.search import AlphaBetaSearch
from chess_ai.selfplay import SelfPlayEngine


class BaseDataSource(ABC):
    """Abstract base class for a chess training data source."""

    @abstractmethod
    def name(self) -> str:
        """Name identifier of the data source."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the source has enough transitions available to sample."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of transitions currently buffered."""
        pass

    @abstractmethod
    def sample(self, n: int) -> List[Transition]:
        """Sample n transitions from this data source."""
        pass

    def start(self) -> None:
        """Start background workers if applicable."""
        pass

    def stop(self) -> None:
        """Stop background workers if applicable."""
        pass


# =============================================================================
# 1. Self-Play Data Source (with optional concurrent multi-workers)
# =============================================================================

class SelfPlaySource(BaseDataSource):
    """Generates games online via model self-play search.

    Supports concurrent background worker threads to play games simultaneously
    without blocking training execution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        encoder: Optional[BoardEncoder] = None,
        search_config: Optional[SearchConfig] = None,
        selfplay_config: Optional[SelfPlayConfig] = None,
        device: Optional[torch.device] = None,
        capacity: int = 20000,
        num_workers: int = 1,
    ):
        self.model = model
        self.encoder = encoder or BoardEncoder()
        self.search_config = search_config or SearchConfig()
        self.selfplay_config = selfplay_config or SelfPlayConfig()
        self.device = device or next(model.parameters()).device
        self.buffer = ReplayBuffer(capacity=capacity)
        self.num_workers = num_workers

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

    def name(self) -> str:
        return "selfplay"

    def is_ready(self) -> bool:
        with self._lock:
            return len(self.buffer) > 0

    def size(self) -> int:
        with self._lock:
            return len(self.buffer)

    def sample(self, n: int) -> List[Transition]:
        with self._lock:
            if not self.buffer.buffer:
                return []
            return random.sample(self.buffer.buffer, min(n, len(self.buffer)))

    def generate_game_sync(self) -> Episode:
        """Generate a single game synchronously and push to buffer."""
        engine = SelfPlayEngine(
            model=self.model,
            encoder=self.encoder,
            search_config=self.search_config,
            selfplay_config=self.selfplay_config,
            device=self.device,
        )
        ep = engine.play_game()
        with self._lock:
            self.buffer.push_episode(ep)
        return ep

    def _worker_loop(self, worker_id: int) -> None:
        engine = SelfPlayEngine(
            model=self.model,
            encoder=self.encoder,
            search_config=self.search_config,
            selfplay_config=self.selfplay_config,
            device=self.device,
        )
        while not self._stop_event.is_set():
            try:
                ep = engine.play_game()
                with self._lock:
                    self.buffer.push_episode(ep)
            except Exception as e:
                time.sleep(0.1)

    def start(self) -> None:
        if self.num_workers > 1 and not self._threads:
            self._stop_event.clear()
            for wid in range(self.num_workers):
                t = threading.Thread(target=self._worker_loop, args=(wid,), daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()


# =============================================================================
# 2. PGN Data Source (Human / Master games from PGN files)
# =============================================================================

class PGNSource(BaseDataSource):
    """Parses real human / grandmaster games from PGN files or strings.

    Extracts transitions (s_t, a_t, r_{t+1}, s_{t+1}, done) with final rewards:
      '1-0'   -> +1.0 (White wins)
      '0-1'   -> -1.0 (Black wins)
      '1/2-1/2' -> 0.0  (Draw)
    """

    def __init__(
        self,
        pgn_paths: Optional[Sequence[str | Path]] = None,
        encoder: Optional[BoardEncoder] = None,
        capacity: int = 20000,
        max_games_to_load: int = 2000,
    ):
        self.pgn_paths = [Path(p) for p in (pgn_paths or [])]
        self.encoder = encoder or BoardEncoder()
        self.buffer = ReplayBuffer(capacity=capacity)
        self.max_games_to_load = max_games_to_load

        # Auto-load PGN files if provided
        if self.pgn_paths:
            self.load_from_files(self.pgn_paths)

    def name(self) -> str:
        return "pgn"

    def is_ready(self) -> bool:
        return len(self.buffer) > 0

    def size(self) -> int:
        return len(self.buffer)

    def sample(self, n: int) -> List[Transition]:
        if not self.buffer.buffer:
            return []
        return random.sample(self.buffer.buffer, min(n, len(self.buffer)))

    def load_from_text(self, pgn_text: str, max_games: Optional[int] = None) -> int:
        """Parse PGN text stream into transitions."""
        handle = io.StringIO(pgn_text)
        return self._parse_from_handle(handle, max_games=max_games or self.max_games_to_load)

    def load_from_files(self, paths: Sequence[str | Path]) -> int:
        """Parse one or multiple PGN files."""
        total_loaded = 0
        for p in paths:
            path = Path(p)
            if path.is_file():
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    loaded = self._parse_from_handle(f, max_games=self.max_games_to_load - total_loaded)
                    total_loaded += loaded
                    if total_loaded >= self.max_games_to_load:
                        break
            elif path.is_dir():
                for pgn_file in path.glob("*.pgn"):
                    with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
                        loaded = self._parse_from_handle(f, max_games=self.max_games_to_load - total_loaded)
                        total_loaded += loaded
                        if total_loaded >= self.max_games_to_load:
                            break
        return total_loaded

    def _parse_from_handle(self, handle, max_games: int) -> int:
        games_loaded = 0
        while games_loaded < max_games and len(self.buffer) < self.buffer.capacity:
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result == "1-0":
                white_reward = 1.0
            elif result == "0-1":
                white_reward = -1.0
            elif result == "1/2-1/2":
                white_reward = 0.0
            else:
                continue  # Skip uncompleted games

            board = game.board()
            moves = list(game.mainline_moves())
            if not moves:
                continue

            transitions = []
            for i, move in enumerate(moves):
                current_fen = board.fen()
                s_t = self.encoder.encode(board).cpu()
                turn = board.turn

                board.push(move)
                next_fen = board.fen()
                s_next = self.encoder.encode(board).cpu()
                is_last_move = (i == len(moves) - 1)

                r = white_reward if is_last_move else 0.0
                transitions.append(
                    Transition(
                        state=s_t,
                        action=move.uci(),
                        reward=r,
                        next_state=s_next,
                        done=is_last_move,
                        turn=turn,
                        fen=current_fen,
                    )
                )

            for t in transitions:
                self.buffer.push(t)
            games_loaded += 1

        return games_loaded


# =============================================================================
# 3. FEN Data Source (Static positions / annotated evaluations)
# =============================================================================

class FENSource(BaseDataSource):
    """Loads static FEN positions and evaluations from files or lists.

    Format per line:
      <FEN> [score]
    where score can be:
      - float in [-1.0, 1.0] (White perspective)
      - result: '1-0', '0-1', '1/2-1/2'
      - or omitted (defaults to 0.0 draw or material heuristic)
    """

    def __init__(
        self,
        fen_paths: Optional[Sequence[str | Path]] = None,
        encoder: Optional[BoardEncoder] = None,
        capacity: int = 20000,
    ):
        self.fen_paths = [Path(p) for p in (fen_paths or [])]
        self.encoder = encoder or BoardEncoder()
        self.buffer = ReplayBuffer(capacity=capacity)

        if self.fen_paths:
            self.load_from_files(self.fen_paths)

    def name(self) -> str:
        return "fen"

    def is_ready(self) -> bool:
        return len(self.buffer) > 0

    def size(self) -> int:
        return len(self.buffer)

    def sample(self, n: int) -> List[Transition]:
        if not self.buffer.buffer:
            return []
        return random.sample(self.buffer.buffer, min(n, len(self.buffer)))

    def load_from_text(self, lines: Sequence[str]) -> int:
        """Parse list of FEN lines."""
        count = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(";")
            fen_part = parts[0].strip()

            reward = 0.0
            if len(parts) > 1:
                eval_str = parts[1].strip()
                try:
                    reward = float(eval_str)
                except ValueError:
                    if eval_str == "1-0":
                        reward = 1.0
                    elif eval_str == "0-1":
                        reward = -1.0
                    else:
                        reward = 0.0

            try:
                board = chess.Board(fen_part)
                state = self.encoder.encode(board).cpu()
                # Self-loop transition marked done for direct target regression:
                # y = r (done=True, no bootstrap) -> Loss = 0.5 * (V(s) - r)^2
                t = Transition(
                    state=state,
                    action="none",
                    reward=reward,
                    next_state=state,
                    done=True,
                    turn=board.turn,
                    fen=fen_part,
                )
                self.buffer.push(t)
                count += 1
            except Exception:
                continue

        return count

    def load_from_files(self, paths: Sequence[str | Path]) -> int:
        total = 0
        for p in paths:
            path = Path(p)
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    total += self.load_from_text(lines)
        return total


# =============================================================================
# 4. Stockfish Data Source (Oracle Distillation & Engine Play)
# =============================================================================

class MockScore:
    """Mock score object mimicking chess.engine.Score."""

    def __init__(self, cp: int = 0, is_mate: bool = False, mate_score: Optional[int] = None):
        self._cp = cp
        self._is_mate = is_mate
        self._mate = mate_score

    def score(self, mate_score: int = 10000) -> Optional[int]:
        if self._is_mate:
            return mate_score if (self._mate and self._mate > 0) else -mate_score
        return self._cp

    def is_mate(self) -> bool:
        return self._is_mate

    def mate(self) -> Optional[int]:
        return self._mate


class MockPovScore:
    """Mock PovScore mimicking chess.engine.PovScore."""

    def __init__(self, white_score: MockScore):
        self._white_score = white_score

    def white(self) -> MockScore:
        return self._white_score


class MockStockfishEngine:
    """Safe fallback engine when Stockfish binary is not installed on the system.

    Implements material + center control heuristic evaluation and 1-ply search.
    """

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }
    CENTER_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}

    def evaluate_heuristic(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for sq, piece in board.piece_map().items():
            val = self.PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += val
                if sq in self.CENTER_SQUARES:
                    score += 15
            else:
                score -= val
                if sq in self.CENTER_SQUARES:
                    score -= 15

        # Mobility bonus
        mobility_white = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
        board.turn = not board.turn
        mobility_black = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
        board.turn = not board.turn
        score += (mobility_white - mobility_black) * 2

        return score

    def analyse(self, board: chess.Board, limit: Any = None) -> Dict[str, Any]:
        if board.is_checkmate():
            mate_plies = -1 if board.turn == chess.WHITE else 1
            return {"score": MockPovScore(MockScore(cp=0, is_mate=True, mate_score=mate_plies))}
        cp = self.evaluate_heuristic(board)
        return {"score": MockPovScore(MockScore(cp=cp, is_mate=False))}

    def play(self, board: chess.Board, limit: Any = None) -> Any:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # 1-ply greedy search
        best_move = legal_moves[0]
        best_val = -999999 if board.turn == chess.WHITE else 999999
        for m in legal_moves:
            board.push(m)
            v = self.evaluate_heuristic(board)
            board.pop()
            if board.turn == chess.WHITE:
                if v > best_val:
                    best_val = v
                    best_move = m
            else:
                if v < best_val:
                    best_val = v
                    best_move = m

        class PlayResult:
            def __init__(self, move: chess.Move):
                self.move = move
                self.ponder = None

        return PlayResult(best_move)

    def configure(self, options: Dict[str, Any]) -> None:
        pass

    def quit(self) -> None:
        pass

    def close(self) -> None:
        pass


def find_stockfish_binary(custom_path: Optional[str | Path] = None) -> Optional[str]:
    """Locate Stockfish executable from custom path, system PATH, or common locations."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return str(p.resolve())
        which_custom = shutil.which(str(custom_path))
        if which_custom:
            return which_custom

    for name in ["stockfish", "stockfish.exe", "stockfish-windows-x86-64-avx2.exe"]:
        found = shutil.which(name)
        if found:
            return found

    candidate_dirs = [
        Path.cwd(),
        Path.cwd() / "bin",
        Path.home() / "stockfish",
        Path("C:/Program Files/Stockfish"),
        Path("C:/stockfish"),
    ]
    for d in candidate_dirs:
        if d.is_dir():
            for f in d.glob("*stockfish*.exe"):
                if f.is_file():
                    return str(f.resolve())

    return None


class StockfishSource(BaseDataSource):
    """Generates training data using the Stockfish chess engine.

    Modes:
      1. 'eval' (Distillation / Value Oracle):
         Evaluates positions (opening, midgame, endgame) with Stockfish depth/time limits.
         Converts centipawns and mate scores to [-1.0, 1.0] winning probability:
           V = tanh(0.00184104 * cp)
         Creates supervised regression transitions (done=True, reward=V) for TD(0).
      2. 'play' (Superhuman / Engine Games):
         Generates full game trajectories where Stockfish plays moves.
         Creates game episodes (s_t, a_t, r_{t+1}, s_{t+1}, done) with terminal rewards.

    Supports:
      - Auto-discovery of Stockfish in PATH or custom executable path.
      - Graceful heuristic fallback (MockStockfishEngine) if binary is not installed.
      - Concurrent background worker threads for non-blocking generation.
    """

    def __init__(
        self,
        stockfish_path: Optional[str | Path] = None,
        mode: str = "eval",
        depth: int = 6,
        time_limit: float = 0.02,
        elo: Optional[int] = None,
        encoder: Optional[BoardEncoder] = None,
        capacity: int = 20000,
        num_workers: int = 0,
        prefill_count: int = 10,
    ):
        self.mode = mode
        self.depth = depth
        self.time_limit = time_limit
        self.elo = elo
        self.encoder = encoder or BoardEncoder()
        self.buffer = ReplayBuffer(capacity=capacity)
        self.num_workers = num_workers

        self._lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

        self.stockfish_path = find_stockfish_binary(stockfish_path)
        self._engine = None
        self.is_mock = False

        self._init_engine()

        if prefill_count > 0:
            if self.mode == "eval":
                self.generate_eval_batch(n_positions=prefill_count)
            elif self.mode == "play":
                self.generate_game_sync(max_moves=30)

    def _init_engine(self) -> None:
        if self.stockfish_path:
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                if self.elo is not None:
                    try:
                        self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
                    except Exception:
                        pass
                self.is_mock = False
            except Exception as e:
                print(f"[StockfishSource] Could not launch Stockfish at '{self.stockfish_path}': {e}. Using MockStockfish fallback.")
                self._engine = MockStockfishEngine()
                self.is_mock = True
        else:
            self._engine = MockStockfishEngine()
            self.is_mock = True

    def name(self) -> str:
        return "stockfish"

    def is_ready(self) -> bool:
        with self._lock:
            return len(self.buffer) > 0

    def size(self) -> int:
        with self._lock:
            return len(self.buffer)

    def sample(self, n: int) -> List[Transition]:
        with self._lock:
            if not self.buffer.buffer:
                return []
            return random.sample(self.buffer.buffer, min(n, len(self.buffer)))

    def _score_to_value(self, score_obj: Any) -> float:
        """Convert score object to [-1.0, 1.0] from White perspective."""
        white_score = score_obj.white()
        if white_score.is_mate():
            m = white_score.mate()
            if m is not None and m > 0:
                return max(0.8, 1.0 - 0.01 * min(abs(m), 20))
            elif m is not None and m < 0:
                return min(-0.8, -1.0 + 0.01 * min(abs(m), 20))
            return 1.0 if (white_score.score() or 0) > 0 else -1.0
        else:
            cp = white_score.score()
            if cp is None:
                return 0.0
            # Centipawn to winning score in [-1.0, 1.0]
            # Standard tanh scaling: cp=400 (one pawn advantage) ~ +0.627, cp=1000 ~ +0.951
            return math.tanh(0.00184104 * cp)

    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate a chess board, returning winning probability in [-1.0, 1.0] for White."""
        with self._engine_lock:
            if self._engine is None:
                return 0.0
            limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
            try:
                info = self._engine.analyse(board, limit)
                return self._score_to_value(info["score"])
            except Exception:
                return 0.0

    def evaluate_fen(self, fen: str) -> float:
        """Evaluate a position from a FEN string."""
        board = chess.Board(fen)
        return self.evaluate_board(board)

    def generate_eval_batch(self, n_positions: int = 10, max_random_moves: int = 30) -> int:
        """Generate n diverse positions and evaluate them with Stockfish."""
        count = 0
        for _ in range(n_positions):
            board = chess.Board()
            plies = random.randint(1, max_random_moves)
            for _ in range(plies):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))
                if board.is_game_over():
                    break

            val = self.evaluate_board(board)
            s_t = self.encoder.encode(board).cpu()
            t = Transition(
                state=s_t,
                action="stockfish_eval",
                reward=val,
                next_state=s_t,
                done=True,
                turn=board.turn,
                fen=board.fen(),
            )
            with self._lock:
                self.buffer.push(t)
            count += 1
        return count

    def load_fens(self, fens: Sequence[str]) -> int:
        """Evaluate a list of FEN positions with Stockfish and push to buffer."""
        count = 0
        for fen in fens:
            fen = fen.strip()
            if not fen or fen.startswith("#"):
                continue
            fen_part = fen.split(";")[0].strip()
            try:
                board = chess.Board(fen_part)
                val = self.evaluate_board(board)
                s_t = self.encoder.encode(board).cpu()
                t = Transition(
                    state=s_t,
                    action="stockfish_eval",
                    reward=val,
                    next_state=s_t,
                    done=True,
                    turn=board.turn,
                    fen=fen_part,
                )
                with self._lock:
                    self.buffer.push(t)
                count += 1
            except Exception:
                continue
        return count

    def generate_game_sync(self, max_moves: int = 150) -> Episode:
        """Play a complete game using Stockfish to choose moves."""
        board = chess.Board()
        transitions: List[Transition] = []
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            current_fen = board.fen()
            turn = board.turn
            state_tensor = self.encoder.encode(board).cpu()

            move = None
            with self._engine_lock:
                if self._engine is not None:
                    limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
                    try:
                        res = self._engine.play(board, limit)
                        if res and res.move:
                            move = res.move
                    except Exception:
                        move = None

            if move is None or move not in board.legal_moves:
                moves = list(board.legal_moves)
                if not moves:
                    break
                move = random.choice(moves)

            board.push(move)
            move_count += 1
            next_state_tensor = self.encoder.encode(board).cpu()
            done = board.is_game_over() or (move_count >= max_moves)

            transitions.append(
                Transition(
                    state=state_tensor,
                    action=move.uci(),
                    reward=0.0,
                    next_state=next_state_tensor,
                    done=done,
                    turn=turn,
                    fen=current_fen,
                )
            )

        # Determine terminal result and reward (from White perspective)
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                res_str = "0-1"
                winner = "black"
                white_reward = -1.0
            else:
                res_str = "1-0"
                winner = "white"
                white_reward = 1.0
        elif (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
            or move_count >= max_moves
        ):
            res_str = "1/2-1/2"
            winner = "draw"
            white_reward = 0.0
        else:
            res_str = "*"
            winner = "draw"
            white_reward = 0.0

        episode = Episode(transitions=transitions, result=res_str, winner=winner)
        episode.assign_terminal_rewards(white_reward)

        with self._lock:
            self.buffer.push_episode(episode)

        return episode

    def _worker_loop(self, worker_id: int) -> None:
        while not self._stop_event.is_set():
            try:
                if self.mode == "eval":
                    self.generate_eval_batch(n_positions=5)
                else:
                    self.generate_game_sync(max_moves=80)
            except Exception:
                pass
            time.sleep(0.05)

    def start(self) -> None:
        if self.num_workers > 0 and not self._threads:
            self._stop_event.clear()
            for wid in range(self.num_workers):
                t = threading.Thread(target=self._worker_loop, args=(wid,), daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()

    def close(self) -> None:
        self.stop()
        with self._engine_lock:
            if self._engine is not None and not self.is_mock:
                try:
                    self._engine.quit()
                except Exception:
                    pass
                self._engine = None


# =============================================================================
# 5. Multi-Source Data Manager (Concurrent Mixed Stream Training)
# =============================================================================

class MultiSourceDataManager:
    """Orchestrates simultaneous training from multiple concurrent sources.

    Features:
      - Weighted mixed batch sampling across active sources (e.g. 50% Self-Play, 30% PGN, 20% FEN).
      - Per-source loss and TD error tracking.
      - Concurrent asynchronous worker threads for non-blocking self-play generation.
    """

    def __init__(
        self,
        config: MultiSourceConfig,
        model: torch.nn.Module,
        encoder: Optional[BoardEncoder] = None,
        search_config: Optional[SearchConfig] = None,
        selfplay_config: Optional[SelfPlayConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.model = model
        self.encoder = encoder or BoardEncoder()
        self.device = device or next(model.parameters()).device

        self.sources: Dict[str, BaseDataSource] = {}

        # 1. Initialize Self-Play source if enabled
        if "selfplay" in config.enabled_sources:
            self.sources["selfplay"] = SelfPlaySource(
                model=self.model,
                encoder=self.encoder,
                search_config=search_config,
                selfplay_config=selfplay_config,
                device=self.device,
                capacity=config.buffer_capacity_per_source,
                num_workers=config.num_selfplay_workers,
            )

        # 2. Initialize PGN source if enabled or pgn_path provided
        if "pgn" in config.enabled_sources or config.pgn_path:
            paths = [config.pgn_path] if config.pgn_path else []
            self.sources["pgn"] = PGNSource(
                pgn_paths=paths,
                encoder=self.encoder,
                capacity=config.buffer_capacity_per_source,
            )

        # 3. Initialize FEN source if enabled or fen_path provided
        if "fen" in config.enabled_sources or config.fen_path:
            paths = [config.fen_path] if config.fen_path else []
            self.sources["fen"] = FENSource(
                fen_paths=paths,
                encoder=self.encoder,
                capacity=config.buffer_capacity_per_source,
            )

        # 4. Initialize Stockfish source if enabled or stockfish_path provided
        if "stockfish" in config.enabled_sources or config.stockfish_path:
            if "stockfish" not in config.enabled_sources:
                config.enabled_sources.append("stockfish")
            self.sources["stockfish"] = StockfishSource(
                stockfish_path=config.stockfish_path,
                mode=config.stockfish_mode,
                depth=config.stockfish_depth,
                time_limit=config.stockfish_time_limit,
                elo=config.stockfish_elo,
                encoder=self.encoder,
                capacity=config.buffer_capacity_per_source,
                num_workers=config.num_stockfish_workers,
            )

        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize sampling weights across currently available and ready sources."""
        raw_weights = {k: self.config.source_weights.get(k, 1.0) for k in self.sources}
        total = sum(raw_weights.values()) or 1.0
        self.normalized_weights = {k: v / total for k, v in raw_weights.items()}

    def add_source(self, source: BaseDataSource, weight: float = 1.0) -> None:
        """Dynamically add or register a custom data source."""
        self.sources[source.name()] = source
        self.config.source_weights[source.name()] = weight
        self._normalize_weights()

    def start(self) -> None:
        """Start all active background workers."""
        for s in self.sources.values():
            s.start()

    def stop(self) -> None:
        """Stop all background workers and release engine resources."""
        for s in self.sources.values():
            s.stop()
            if hasattr(s, "close"):
                s.close()

    def is_ready(self) -> bool:
        """Check if at least one active source has transitions."""
        return any(s.is_ready() for s in self.sources.values())

    def sample_mixed_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Sample a blended batch from all active sources simultaneously.

        Returns:
          states: [B, input_dim]
          rewards: [B, 1]
          next_states: [B, input_dim]
          dones: [B, 1]
          source_names: List[str] of length B indicating the origin source for each sample
        """
        ready_sources = {k: s for k, s in self.sources.items() if s.is_ready()}
        if not ready_sources:
            raise RuntimeError("No data sources are ready to sample from!")

        # Re-normalize weights among ready sources
        sub_weights = {k: self.normalized_weights.get(k, 1.0) for k in ready_sources}
        w_sum = sum(sub_weights.values()) or 1.0
        norm_sub = {k: v / w_sum for k, v in sub_weights.items()}

        collected_transitions: List[Transition] = []
        source_labels: List[str] = []

        # Allocate counts per source
        remaining = batch_size
        items = list(norm_sub.items())
        for i, (name, weight) in enumerate(items):
            src = ready_sources[name]
            if i == len(items) - 1:
                n_count = remaining
            else:
                n_count = int(round(batch_size * weight))
                n_count = min(remaining, n_count)

            if n_count > 0:
                sampled = src.sample(n_count)
                collected_transitions.extend(sampled)
                source_labels.extend([name] * len(sampled))
                remaining -= len(sampled)

        # If any slots remain unfilled due to small buffers, fill with any ready source
        if remaining > 0 and collected_transitions:
            any_src = list(ready_sources.values())[0]
            extra = any_src.sample(remaining)
            collected_transitions.extend(extra)
            source_labels.extend([any_src.name()] * len(extra))

        # Stack into PyTorch tensors
        states = torch.stack([t.state for t in collected_transitions], dim=0)
        rewards = torch.tensor([t.reward for t in collected_transitions], dtype=torch.float32).unsqueeze(-1)
        next_states = torch.stack([t.next_state for t in collected_transitions], dim=0)
        dones = torch.tensor([t.done for t in collected_transitions], dtype=torch.float32).unsqueeze(-1)

        dev = device or self.device
        if dev is not None:
            states = states.to(dev)
            rewards = rewards.to(dev)
            next_states = next_states.to(dev)
            dones = dones.to(dev)

        return states, rewards, next_states, dones, source_labels

    def get_source_stats(self) -> Dict[str, Any]:
        """Return status and sizes of all sources."""
        return {
            k: {
                "size": s.size(),
                "ready": s.is_ready(),
                "weight": self.normalized_weights.get(k, 0.0),
            }
            for k, s in self.sources.items()
        }
