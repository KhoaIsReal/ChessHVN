"""Unit tests for Multi-Source concurrent data streaming (SelfPlay, PGN, FEN, MultiSourceDataManager)."""

import io
import time
import pytest
import torch

from chess_ai.board_encoder import BoardEncoder
from chess_ai.config import MultiSourceConfig, SearchConfig, SelfPlayConfig
from chess_ai.model import ChessValueNet
from chess_ai.sources import FENSource, MultiSourceDataManager, PGNSource, SelfPlaySource, StockfishSource


SAMPLE_PGN = """
[Event "F/S Return Match"]
[Site "Belgrade, Serbia JUG"]
[Date "1992.11.04"]
[Round "29"]
[White "Fischer, Robert J."]
[Black "Spassky, Boris V."]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 1/2-1/2

[Event "World Championship 35th"]
[Site "Lyon / New York"]
[Date "1990.12.15"]
[Round "20"]
[White "Kasparov, Garry"]
[Black "Karpov, Anatoly"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. d4 Bg4 10. d5 1-0
"""

SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1; 0.0",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4; 0.25",
    "8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1; 1.0",
    "8/8/8/4k3/8/8/4K3/8 b - - 0 1; 0.0",
]


def test_pgn_source_parsing():
    encoder = BoardEncoder()
    source = PGNSource(encoder=encoder)
    loaded = source.load_from_text(SAMPLE_PGN)

    assert loaded == 2, "Should parse 2 completed games from PGN"
    assert source.size() > 0
    assert source.is_ready()

    sample_transitions = source.sample(10)
    assert len(sample_transitions) == 10
    # Verify transitions have valid shapes and contents
    for t in sample_transitions:
        assert t.state.shape == (encoder.input_dim,)
        assert t.next_state.shape == (encoder.input_dim,)
        assert isinstance(t.reward, float)


def test_fen_source_parsing():
    encoder = BoardEncoder()
    source = FENSource(encoder=encoder)
    loaded = source.load_from_text(SAMPLE_FENS)

    assert loaded == 4
    assert source.size() == 4
    assert source.is_ready()

    batch = source.sample(4)
    assert len(batch) == 4
    # Check that winning position has reward 1.0
    queen_endgame = [t for t in batch if "4Q3" in t.fen][0]
    assert queen_endgame.reward == 1.0
    assert queen_endgame.done is True


def test_multi_source_mixed_batch():
    encoder = BoardEncoder()
    model = ChessValueNet(input_dim=encoder.input_dim, hidden_dims=[32, 32])

    cfg = MultiSourceConfig(
        enabled_sources=["pgn", "fen"],
        source_weights={"pgn": 0.5, "fen": 0.5},
    )

    manager = MultiSourceDataManager(config=cfg, model=model, encoder=encoder)

    # Populate PGN and FEN
    pgn_src: PGNSource = manager.sources["pgn"]  # type: ignore
    pgn_src.load_from_text(SAMPLE_PGN)

    fen_src: FENSource = manager.sources["fen"]  # type: ignore
    fen_src.load_from_text(SAMPLE_FENS)

    assert manager.is_ready()
    stats = manager.get_source_stats()
    assert "pgn" in stats and "fen" in stats

    # Sample mixed batch of 16
    batch_size = 16
    states, rewards, next_states, dones, labels = manager.sample_mixed_batch(batch_size)

    assert states.shape == (batch_size, encoder.input_dim)
    assert rewards.shape == (batch_size, 1)
    assert next_states.shape == (batch_size, encoder.input_dim)
    assert dones.shape == (batch_size, 1)

    # Verify both sources contributed to the mixed batch
    assert "pgn" in labels
    assert "fen" in labels


def test_concurrent_selfplay_workers():
    """Verify concurrent multi-worker self-play runs asynchronously and buffers games."""
    encoder = BoardEncoder()
    model = ChessValueNet(input_dim=encoder.input_dim, hidden_dims=[32, 32])

    source = SelfPlaySource(
        model=model,
        encoder=encoder,
        search_config=SearchConfig(depth=1),
        selfplay_config=SelfPlayConfig(max_moves_per_game=10),
        num_workers=2,
    )

    assert source.size() == 0

    # Start 2 concurrent workers
    source.start()
    # Let workers generate at least 1 game in the background
    time.sleep(1.2)
    source.stop()

    assert source.size() > 0, "Background workers should have generated games into the buffer"
    assert source.is_ready()


def test_stockfish_source_mock_eval():
    """Verify StockfishSource evaluation mode with mock fallback works end-to-end."""
    encoder = BoardEncoder()
    source = StockfishSource(encoder=encoder, mode="eval", prefill_count=5)

    assert source.name() == "stockfish"
    assert source.is_ready()
    assert source.size() >= 5
    assert source.is_mock is True

    # 1. Starting position evaluation should be near 0
    start_val = source.evaluate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert abs(start_val) < 0.3

    # 2. Queen advantage for White should be strongly positive
    white_queen_up = source.evaluate_fen("8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1")
    assert white_queen_up > 0.7

    # 3. Queen advantage for Black should be strongly negative
    black_queen_up = source.evaluate_fen("8/8/8/4k3/8/8/4K3/4q3 w - - 0 1")
    assert black_queen_up < -0.7

    # 4. Generate batch of diverse evaluations
    initial_size = source.size()
    added = source.generate_eval_batch(n_positions=4)
    assert added == 4
    assert source.size() == initial_size + 4

    # 5. Check sampled transition properties
    sampled = source.sample(5)
    assert len(sampled) == 5
    for t in sampled:
        assert t.state.shape == (encoder.input_dim,)
        assert t.done is True
        assert -1.0 <= t.reward <= 1.0


def test_stockfish_source_play_mode():
    """Verify StockfishSource game generation mode creates full game trajectories."""
    encoder = BoardEncoder()
    source = StockfishSource(encoder=encoder, mode="play", prefill_count=0)

    assert source.name() == "stockfish"
    assert source.size() == 0

    episode = source.generate_game_sync(max_moves=20)
    assert episode.length > 0
    assert episode.winner in ["white", "black", "draw"]
    assert episode.result in ["1-0", "0-1", "1/2-1/2", "*"]
    assert source.size() == episode.length

    # Verify terminal transition has done=True
    assert episode.transitions[-1].done is True


def test_stockfish_score_to_value_formula():
    """Verify centipawn and mate score conversions to [-1.0, 1.0]."""
    source = StockfishSource(prefill_count=0)

    from chess_ai.sources import MockPovScore, MockScore

    # 0 centipawns -> 0.0
    v_even = source._score_to_value(MockPovScore(MockScore(cp=0)))
    assert abs(v_even - 0.0) < 1e-4

    # +400 cp (~one pawn) -> ~0.627
    v_pawn = source._score_to_value(MockPovScore(MockScore(cp=400)))
    assert 0.60 < v_pawn < 0.65

    # -400 cp -> -0.627
    v_pawn_down = source._score_to_value(MockPovScore(MockScore(cp=-400)))
    assert -0.65 < v_pawn_down < -0.60
    assert abs(v_pawn + v_pawn_down) < 1e-4  # Antisymmetry

    # +1000 cp -> ~0.95
    v_huge = source._score_to_value(MockPovScore(MockScore(cp=1000)))
    assert 0.90 < v_huge < 1.0

    # Mate in 1 for White -> ~0.99
    v_mate_w = source._score_to_value(MockPovScore(MockScore(is_mate=True, mate_score=1)))
    assert 0.95 <= v_mate_w <= 1.0

    # Mate in 1 for Black -> ~-0.99
    v_mate_b = source._score_to_value(MockPovScore(MockScore(is_mate=True, mate_score=-1)))
    assert -1.0 <= v_mate_b <= -0.95


def test_multi_source_with_stockfish():
    """Verify MultiSourceDataManager seamlessly integrates StockfishSource with SelfPlay."""
    encoder = BoardEncoder()
    model = ChessValueNet(input_dim=encoder.input_dim, hidden_dims=[32, 32])

    cfg = MultiSourceConfig(
        enabled_sources=["selfplay", "stockfish"],
        source_weights={"selfplay": 0.5, "stockfish": 0.5},
        stockfish_mode="eval",
    )

    manager = MultiSourceDataManager(
        config=cfg,
        model=model,
        encoder=encoder,
        search_config=SearchConfig(depth=1),
        selfplay_config=SelfPlayConfig(max_moves_per_game=8),
    )

    # Generate 1 self-play game
    sp_src: SelfPlaySource = manager.sources["selfplay"]  # type: ignore
    sp_src.generate_game_sync()

    assert manager.is_ready()
    stats = manager.get_source_stats()
    assert "selfplay" in stats and "stockfish" in stats
    assert stats["stockfish"]["ready"] is True

    # Sample mixed batch of 16
    batch_size = 16
    states, rewards, next_states, dones, labels = manager.sample_mixed_batch(batch_size)

    assert states.shape == (batch_size, encoder.input_dim)
    assert rewards.shape == (batch_size, 1)
    assert next_states.shape == (batch_size, encoder.input_dim)
    assert dones.shape == (batch_size, 1)

    assert "selfplay" in labels
    assert "stockfish" in labels

    manager.stop()
