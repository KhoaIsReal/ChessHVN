"""Main CLI and entrypoints for Chess AI research prototype."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.optim as optim

from chess_ai.board_encoder import BoardEncoder
from chess_ai.checkpoint import CheckpointManager, find_latest_checkpoint
from chess_ai.config import ExperimentConfig, GrowthConfig, ModelConfig, SearchConfig, SelfPlayConfig, TDConfig, MultiSourceConfig
from chess_ai.growth import DynamicGrowthManager, GrowthController
from chess_ai.model import ChessValueNet
from chess_ai.replay import ReplayBuffer
from chess_ai.search import AlphaBetaSearch
from chess_ai.selfplay import SelfPlayEngine
from chess_ai.sources import MultiSourceDataManager, SelfPlaySource
from chess_ai.td_learning import TDLearner


def seed_everything(seed: int) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device_str: str) -> torch.device:
    """Select compute device with robust CPU fallback."""
    if device_str.lower() in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def get_gpu_memory_str(device: torch.device) -> str:
    """Get current GPU memory allocation string if running on CUDA."""
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        return f"{allocated:.1f}MB / {reserved:.1f}MB"
    return "N/A (CPU)"


# =============================================================================
# CLI Subcommand: TRAIN
# =============================================================================

def run_train(args: argparse.Namespace) -> None:
    # Build configuration with auto-named experiment if omitted
    exp_name = args.name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    cfg = ExperimentConfig(
        name=exp_name,
        device=args.device,
        seed=args.seed,
        episodes=args.episodes,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
    )

    # Hidden dims override
    if args.hidden_dim is not None:
        cfg.model.hidden_dims = [args.hidden_dim, args.hidden_dim]
    elif args.hidden_dims is not None:
        cfg.model.hidden_dims = args.hidden_dims

    # Sparsity / Neuron masking
    if args.sparse_residual_enabled or args.sparse_residual_density is not None:
        cfg.model.sparse_residual_enabled = True
        if args.sparse_residual_density is not None:
            cfg.model.sparse_residual_density = args.sparse_residual_density

    if args.genius_neuron_dropout_enabled or args.genius_neuron_dropout_rate is not None:
        cfg.model.genius_neuron_dropout_enabled = True
        if args.genius_neuron_dropout_rate is not None:
            cfg.model.genius_neuron_dropout_rate = args.genius_neuron_dropout_rate

    # TD config
    if args.td_method:
        cfg.td.td_method = args.td_method
    if args.gamma is not None:
        cfg.td.gamma = args.gamma
    if args.lambda_ is not None:
        cfg.td.lambda_ = args.lambda_
    if args.lr is not None:
        cfg.td.learning_rate = args.lr

    # Search config
    if args.search_depth is not None:
        cfg.search.depth = args.search_depth

    # Growth config
    if args.growth_mode:
        cfg.growth.growth_mode = args.growth_mode
    if args.growth_increment is not None:
        cfg.growth.growth_increment = args.growth_increment
    if args.patience is not None:
        cfg.growth.patience = args.patience

    # Multi-Source configuration
    if getattr(args, "sources", None):
        cfg.sources.enabled_sources = args.sources
    if getattr(args, "pgn_path", None):
        cfg.sources.pgn_path = args.pgn_path
        if "pgn" not in cfg.sources.enabled_sources:
            cfg.sources.enabled_sources.append("pgn")
    if getattr(args, "fen_path", None):
        cfg.sources.fen_path = args.fen_path
        if "fen" not in cfg.sources.enabled_sources:
            cfg.sources.enabled_sources.append("fen")
    if getattr(args, "stockfish_path", None):
        cfg.sources.stockfish_path = args.stockfish_path
        if "stockfish" not in cfg.sources.enabled_sources:
            cfg.sources.enabled_sources.append("stockfish")
    if getattr(args, "stockfish_mode", None):
        cfg.sources.stockfish_mode = args.stockfish_mode
    if getattr(args, "stockfish_depth", None) is not None:
        cfg.sources.stockfish_depth = args.stockfish_depth
    if getattr(args, "stockfish_time_limit", None) is not None:
        cfg.sources.stockfish_time_limit = args.stockfish_time_limit
    if getattr(args, "stockfish_elo", None) is not None:
        cfg.sources.stockfish_elo = args.stockfish_elo
    if getattr(args, "num_selfplay_workers", None) is not None:
        cfg.sources.num_selfplay_workers = args.num_selfplay_workers
    if getattr(args, "num_stockfish_workers", None) is not None:
        cfg.sources.num_stockfish_workers = args.num_stockfish_workers
    if getattr(args, "source_weights", None) and len(args.source_weights) == len(cfg.sources.enabled_sources):
        cfg.sources.source_weights = {s: w for s, w in zip(cfg.sources.enabled_sources, args.source_weights)}

    # Seed
    seed_everything(cfg.seed)
    device = select_device(cfg.device)

    # Prepare experiment output directory
    exp_dir = Path(cfg.output_dir) / cfg.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(exp_dir / "config.json")

    # Encoder & Model
    encoder = BoardEncoder()
    model = ChessValueNet(
        input_dim=encoder.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        sparse_residual_enabled=cfg.model.sparse_residual_enabled,
        sparse_residual_density=cfg.model.sparse_residual_density,
        genius_neuron_dropout_enabled=cfg.model.genius_neuron_dropout_enabled,
        genius_neuron_dropout_rate=cfg.model.genius_neuron_dropout_rate,
        activation=cfg.model.activation,
        use_layer_norm=cfg.model.use_layer_norm,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.td.learning_rate,
        weight_decay=cfg.td.weight_decay,
    )

    td_learner = TDLearner(model=model, config=cfg.td, optimizer=optimizer)
    growth_manager = DynamicGrowthManager(model=model, optimizer=optimizer)
    growth_controller = GrowthController(growth_manager, config=cfg.growth)
    checkpoint_format = getattr(args, "checkpoint_format", "huggingface")
    checkpoint_mgr = CheckpointManager(experiment_dir=exp_dir, default_format=checkpoint_format)

    # Initialize Multi-Source Data Manager
    multi_source_mgr = MultiSourceDataManager(
        config=cfg.sources,
        model=model,
        encoder=encoder,
        search_config=cfg.search,
        selfplay_config=cfg.selfplay,
        device=device,
    )
    multi_source_mgr.start()

    # Print setup summary
    stats = model.get_parameter_stats()
    print("=" * 70)
    print(f"Starting Experiment: {cfg.name}")
    print(f"Device: {device} | Seed: {cfg.seed}")
    print(f"Architecture: input={encoder.input_dim} -> hidden={model.hidden_dims} -> 1")
    print(f"Total Parameters: {stats['total_params']:,} ({stats['fp32_str']} FP32)")
    print(f"Sparse Residual: {cfg.model.sparse_residual_enabled} (density={cfg.model.sparse_residual_density})")
    print(f"Neuron Masking: {cfg.model.genius_neuron_dropout_enabled} (p={cfg.model.genius_neuron_dropout_rate})")
    print(f"TD Method: {cfg.td.td_method} (gamma={cfg.td.gamma}, lambda={cfg.td.lambda_})")
    print(f"Search Depth: {cfg.search.depth} | Growth Mode: {cfg.growth.growth_mode}")

    active_sources = list(multi_source_mgr.sources.keys())
    weights_str = ", ".join(f"{k}: {w:.1%}" for k, w in multi_source_mgr.normalized_weights.items())
    print(f"Data Sources: {active_sources} (Sampling Weights: {weights_str})")
    for src_name, src_obj in multi_source_mgr.sources.items():
        is_mock_str = " [Mock Heuristic Fallback]" if getattr(src_obj, "is_mock", False) else ""
        print(f"  - Source [{src_name}]: {src_obj.size():,} pre-buffered transitions{is_mock_str}")
    print("=" * 70)

    # Metrics CSV setup
    metrics_path = exp_dir / "metrics.csv"
    csv_file = open(metrics_path, "w", newline="", encoding="utf-8")
    csv_fields = [
        "step", "episode", "loss", "mean_td_error", "mean_abs_td_error",
        "win_rate", "draw_rate", "loss_rate", "total_params", "hidden_dims",
        "active_neuron_ratio", "sparse_density", "nodes_per_sec", "gpu_memory",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    game_outcomes: List[str] = []
    total_steps = 0

    start_train_time = time.time()

    for ep in range(1, cfg.episodes + 1):
        # 1. Step self-play game if selfplay is an active source
        ep_start = time.perf_counter()
        episode = None
        if "selfplay" in multi_source_mgr.sources:
            sp_src = multi_source_mgr.sources["selfplay"]
            if isinstance(sp_src, SelfPlaySource):
                episode = sp_src.generate_game_sync()
                game_outcomes.append(episode.winner or "draw")

        # Step Stockfish generation if active and synchronous (num_workers == 0)
        if "stockfish" in multi_source_mgr.sources:
            sf_src = multi_source_mgr.sources["stockfish"]
            if getattr(sf_src, "num_workers", 0) == 0:
                if getattr(sf_src, "mode", "eval") == "eval":
                    sf_src.generate_eval_batch(n_positions=5)
                elif getattr(sf_src, "mode", "eval") == "play":
                    sf_ep = sf_src.generate_game_sync(max_moves=60)
                    if episode is None:
                        episode = sf_ep
        ep_elapsed = time.perf_counter() - ep_start

        # 2. TD Learning updates from multi-source stream simultaneously
        td_metrics = {"loss": 0.0, "mean_td_error": 0.0, "mean_abs_td_error": 0.0}

        if cfg.td.td_method == "td_lambda":
            # Forward-view trajectory update on episode
            if episode and episode.transitions:
                traj_states = torch.stack([t.state for t in episode.transitions], dim=0).to(device)
                traj_rewards = torch.tensor([t.reward for t in episode.transitions], dtype=torch.float32, device=device)
                traj_dones = torch.tensor([t.done for t in episode.transitions], dtype=torch.float32, device=device)
                td_metrics = td_learner.train_step_td_lambda_forward(traj_states, traj_rewards, traj_dones)
                total_steps += 1
        else:
            # TD(0) batched update from simultaneous multi-sources
            if multi_source_mgr.is_ready():
                states, rewards, next_states, dones, source_labels = multi_source_mgr.sample_mixed_batch(
                    cfg.batch_size, device=device
                )
                td_metrics = td_learner.train_step_td0(states, rewards, next_states, dones)
                total_steps += 1

        # 3. Dynamic Growth check
        growth_events = growth_controller.step(
            current_metric=td_metrics["mean_abs_td_error"],
            step_idx=total_steps,
        )
        if growth_events:
            for ev in growth_events:
                print(f"[GROWTH EVENT] Layer {ev['layer_idx']} grew {ev['old_dim']} -> {ev['new_dim']} "
                      f"({ev['old_params']:,} -> {ev['new_params']:,} params). Reason: {ev['reason']}")

        # 4. Compute running game statistics
        recent = game_outcomes[-50:]
        win_rate = (recent.count("white") / len(recent)) if recent else 0.0
        loss_rate = (recent.count("black") / len(recent)) if recent else 0.0
        draw_rate = (recent.count("draw") / len(recent)) if recent else 0.0

        # Sparsity & neuron ratios
        active_ratio = model.blocks[0].active_neuron_ratio if model.blocks else 1.0
        sparse_density = model.blocks[0].actual_sparse_density if model.blocks else 0.0
        gpu_mem = get_gpu_memory_str(device)

        # Search NPS estimate
        nodes_per_sec = (len(episode.transitions) / max(1e-4, ep_elapsed)) if episode else 0.0
        ep_res = episode.result if episode else "N/A"
        ep_win = episode.winner if episode else "N/A"

        # Log to CSV
        log_entry = {
            "step": total_steps,
            "episode": ep,
            "loss": round(td_metrics["loss"], 6),
            "mean_td_error": round(td_metrics["mean_td_error"], 6),
            "mean_abs_td_error": round(td_metrics["mean_abs_td_error"], 6),
            "win_rate": round(win_rate, 4),
            "draw_rate": round(draw_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "total_params": model.get_parameter_stats()["total_params"],
            "hidden_dims": str(model.hidden_dims),
            "active_neuron_ratio": round(active_ratio, 4),
            "sparse_density": round(sparse_density, 6),
            "nodes_per_sec": round(nodes_per_sec, 2),
            "gpu_memory": gpu_mem,
        }
        csv_writer.writerow(log_entry)
        csv_file.flush()

        if ep % cfg.log_interval == 0:
            print(f"Ep {ep:3d}/{cfg.episodes} | Steps: {total_steps:3d} | "
                  f"Loss: {td_metrics['loss']:.5f} | "
                  f"|TD|: {td_metrics['mean_abs_td_error']:.4f} | "
                  f"Result: {ep_res:5s} ({ep_win:5s}) | "
                  f"Dims: {model.hidden_dims} | "
                  f"GPU: {gpu_mem}")

        # 5. Checkpoint
        if ep % cfg.checkpoint_interval == 0 or ep == cfg.episodes:
            ckpt_path = checkpoint_mgr.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=total_steps,
                episode=ep,
                config=cfg,
            )
            print(f"[CHECKPOINT] Saved checkpoint to {ckpt_path.name} (format: {checkpoint_format})")

    multi_source_mgr.stop()
    csv_file.close()

    # Save summary
    summary = {
        "experiment_name": cfg.name,
        "total_episodes": cfg.episodes,
        "total_training_steps": total_steps,
        "final_parameters": model.get_parameter_stats(),
        "final_hidden_dims": model.hidden_dims,
        "growth_history": model.growth_history,
        "elapsed_seconds": round(time.time() - start_train_time, 2),
        "outcomes": {
            "white_wins": game_outcomes.count("white"),
            "black_wins": game_outcomes.count("black"),
            "draws": game_outcomes.count("draw"),
        },
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"Training completed successfully. Summary saved to {exp_dir / 'summary.json'}")
    print("=" * 70)


# =============================================================================
# CLI Subcommand: BENCHMARK
# =============================================================================

def run_benchmark(args: argparse.Namespace) -> None:
    """Benchmark forward throughput and memory across batch sizes and Huge Layer widths."""
    device = select_device(args.device)
    hidden_dims = args.hidden_dims if args.hidden_dims else ([args.hidden_dim, args.hidden_dim] if args.hidden_dim else [4096, 4096])
    batch_sizes = args.batch_sizes or [1, 16, 64, 256, 1024]

    encoder = BoardEncoder()
    model = ChessValueNet(
        input_dim=encoder.input_dim,
        hidden_dims=hidden_dims,
        sparse_residual_enabled=args.sparse_residual_enabled,
        sparse_residual_density=args.sparse_residual_density or 0.01,
        genius_neuron_dropout_enabled=args.genius_neuron_dropout_enabled,
        genius_neuron_dropout_rate=args.genius_neuron_dropout_rate or 0.05,
    ).to(device)
    model.eval()

    param_stats = model.get_parameter_stats()

    print("=" * 75)
    print("CHESS AI HUGE LAYER BENCHMARK (INFERENCE ONLY)")
    print("=" * 75)
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"Architecture: input={encoder.input_dim} -> hidden={hidden_dims} -> 1")
    print(f"Total Parameters:      {param_stats['total_params']:,}")
    print(f"Memory (FP32):         {param_stats['fp32_str']}")
    print(f"Memory (FP16/BF16):    {param_stats['fp16_str']}")
    print(f"Memory (INT16 est.):   {param_stats['int16_str']}")
    print(f"Sparse Residual:       {args.sparse_residual_enabled} (density={args.sparse_residual_density})")
    print(f"Neuron Masking:        {args.genius_neuron_dropout_enabled} (rate={args.genius_neuron_dropout_rate})")
    print("-" * 75)
    print(f"{'Batch Size':>10} | {'Latency (ms)':>14} | {'Throughput (pos/s)':>20} | {'GPU Mem':>16}")
    print("-" * 75)

    num_warmup = 10
    num_runs = 50

    with torch.no_grad():
        for b in batch_sizes:
            # Create dummy input batch
            x = torch.randn(b, encoder.input_dim, device=device)

            # Warmup
            for _ in range(num_warmup):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Timed runs
            t0 = time.perf_counter()
            for _ in range(num_runs):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_elapsed = t1 - t0
            avg_batch_latency_ms = (total_elapsed / num_runs) * 1000.0
            throughput = (b * num_runs) / total_elapsed
            gpu_mem = get_gpu_memory_str(device)

            print(f"{b:>10d} | {avg_batch_latency_ms:>14.3f} | {throughput:>20.1f} | {gpu_mem:>16}")

    print("=" * 75)


# =============================================================================
# CLI Subcommand: SELFPLAY
# =============================================================================

def run_selfplay(args: argparse.Namespace) -> None:
    """Generate self-play games and inspect move trajectories."""
    device = select_device(args.device)
    encoder = BoardEncoder()

    ckpt_path = args.checkpoint
    if not ckpt_path:
        discovered = find_latest_checkpoint()
        if discovered:
            ckpt_path = str(discovered)
            print(f"[INFO] Auto-discovered latest checkpoint: {ckpt_path}")
        else:
            print("[INFO] No --checkpoint specified and no saved checkpoint found. Using freshly initialized model.")

    if ckpt_path:
        p = Path(ckpt_path)
        exp_dir = p.parent.parent if p.parent.name in ("checkpoints", "latest") else p.parent
        mgr = CheckpointManager(exp_dir)
        model, meta = mgr.load_checkpoint(ckpt_path, model=None, device=device)
        print(f"Loaded checkpoint from {ckpt_path} (format: {meta.get('format', 'unknown')}, hidden dims: {model.hidden_dims})")
    else:
        model = ChessValueNet(input_dim=encoder.input_dim, hidden_dims=[1024, 1024]).to(device)

    search_cfg = SearchConfig(depth=args.search_depth or 2)
    selfplay_cfg = SelfPlayConfig(max_moves_per_game=args.max_moves or 100)
    engine = SelfPlayEngine(
        model=model,
        encoder=encoder,
        search_config=search_cfg,
        selfplay_config=selfplay_cfg,
        device=device,
    )

    print(f"Running {args.num_games} self-play games at search depth {search_cfg.depth}...")
    for i in range(1, args.num_games + 1):
        t0 = time.perf_counter()
        ep = engine.play_game()
        dur = time.perf_counter() - t0
        moves_str = " ".join(t.action for t in ep.transitions[:10]) + ("..." if len(ep.transitions) > 10 else "")
        print(f"Game {i:2d} | Result: {ep.result:7s} | Winner: {ep.winner:6s} | Plies: {len(ep.transitions):3d} | Time: {dur:.2f}s")
        print(f"       Moves sample: {moves_str}")


# =============================================================================
# CLI Subcommand: EVALUATE
# =============================================================================

def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate model against material baseline."""
    import chess
    device = select_device(args.device)
    encoder = BoardEncoder()

    ckpt_path = args.checkpoint
    if not ckpt_path:
        discovered = find_latest_checkpoint()
        if discovered:
            ckpt_path = str(discovered)
            print(f"[INFO] Auto-discovered latest checkpoint: {ckpt_path}")
        else:
            print("[INFO] No --checkpoint specified and no saved checkpoint found. Using freshly initialized model.")

    if ckpt_path:
        p = Path(ckpt_path)
        exp_dir = p.parent.parent if p.parent.name in ("checkpoints", "latest") else p.parent
        mgr = CheckpointManager(exp_dir)
        model, meta = mgr.load_checkpoint(ckpt_path, model=None, device=device)
        print(f"Loaded model from {ckpt_path} (format: {meta.get('format', 'unknown')}, hidden dims: {model.hidden_dims})")
    else:
        model = ChessValueNet(input_dim=encoder.input_dim, hidden_dims=[1024, 1024]).to(device)

    neural_searcher = AlphaBetaSearch(
        model=model,
        config=SearchConfig(depth=args.search_depth or 2, evaluation_mode="neural"),
        device=device,
        encoder=encoder,
    )

    material_searcher = AlphaBetaSearch(
        model=None,
        config=SearchConfig(depth=args.search_depth or 2, evaluation_mode="material"),
        device=device,
        encoder=encoder,
    )

    num_games = args.num_games or 4
    print(f"Evaluating Neural Model vs Material Search ({num_games} games)...")

    results = {"model_wins": 0, "material_wins": 0, "draws": 0}

    for g in range(1, num_games + 1):
        # Alternate colors: Odd games Model is White, Even games Material is White
        model_is_white = (g % 2 != 0)
        board = chess.Board()
        move_count = 0

        while not board.is_game_over() and move_count < 150:
            is_model_turn = (board.turn == chess.WHITE) if model_is_white else (board.turn == chess.BLACK)
            searcher = neural_searcher if is_model_turn else material_searcher
            res = searcher.search(board)
            move = res.best_move or random.choice(list(board.legal_moves))
            board.push(move)
            move_count += 1

        if board.is_checkmate():
            if (board.turn == chess.WHITE and model_is_white) or (board.turn == chess.BLACK and not model_is_white):
                winner = "material"
                results["material_wins"] += 1
            else:
                winner = "neural_model"
                results["model_wins"] += 1
        else:
            winner = "draw"
            results["draws"] += 1

        print(f"Match {g:2d} | Model is {'White' if model_is_white else 'Black'} | Winner: {winner} | Plies: {move_count}")

    print("-" * 50)
    print(f"Results: {results}")


# =============================================================================
# Argument Parser & Entrypoint
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chess_ai",
        description="Chess AI: Research Prototype for Huge Layer Value Networks",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. train
    p_train = subparsers.add_parser("train", help="Train value network with self-play and TD learning")
    p_train.add_argument("--name", type=str, default=None, help="Experiment name (auto-generated if omitted)")
    p_train.add_argument("--checkpoint-format", type=str, choices=["huggingface", "pt"], default="huggingface", help="Checkpoint format (default: huggingface)")
    p_train.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument("--episodes", type=int, default=10, help="Number of self-play episodes")
    p_train.add_argument("--batch-size", type=int, default=64, help="Batch size for TD learning")
    p_train.add_argument("--replay-buffer-size", type=int, default=10000, help="Replay buffer capacity")
    p_train.add_argument("--checkpoint-interval", type=int, default=5, help="Episodes between checkpoints")
    p_train.add_argument("--output-dir", type=str, default="runs", help="Output directory")

    p_train.add_argument("--hidden-dim", type=int, default=None, help="Uniform hidden dimension for all layers")
    p_train.add_argument("--hidden-dims", type=int, nargs="+", default=None, help="Specific hidden dimensions")
    p_train.add_argument("--sparse-residual-enabled", action="store_true", help="Enable sparse residual mechanism")
    p_train.add_argument("--sparse-residual-density", type=float, default=None, help="Sparse residual density (e.g. 0.01)")
    p_train.add_argument("--genius-neuron-dropout-enabled", action="store_true", help="Enable neuron masking")
    p_train.add_argument("--genius-neuron-dropout-rate", type=float, default=None, help="Neuron dropout rate (e.g. 0.05)")

    p_train.add_argument("--td-method", type=str, choices=["td0", "td_lambda"], default="td0", help="TD learning algorithm")
    p_train.add_argument("--lambda", dest="lambda_", type=float, default=0.8, help="TD(lambda) trace decay")
    p_train.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    p_train.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    p_train.add_argument("--search-depth", type=int, default=2, help="Alpha-Beta search depth")
    p_train.add_argument("--growth-mode", type=str, choices=["manual", "plateau", "adaptive"], default="plateau", help="Growth trigger mode (default: plateau)")
    p_train.add_argument("--growth-increment", type=int, default=2048, help="Neurons added on growth")
    p_train.add_argument("--patience", type=int, default=10, help="Patience steps before growth on plateau")

    # Multi-source training arguments
    p_train.add_argument("--sources", type=str, nargs="+", default=None, help="Active training sources ('selfplay', 'pgn', 'fen', 'stockfish')")
    p_train.add_argument("--source-weights", type=float, nargs="+", default=None, help="Sampling weights for active sources")
    p_train.add_argument("--pgn-path", type=str, default=None, help="Path to PGN file or folder of Grandmaster games")
    p_train.add_argument("--fen-path", type=str, default=None, help="Path to FEN/EPD file of position benchmarks")
    p_train.add_argument("--num-selfplay-workers", type=int, default=1, help="Concurrent self-play worker threads")
    p_train.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish executable (auto-discovered if omitted)")
    p_train.add_argument("--stockfish-mode", type=str, choices=["eval", "play"], default="eval", help="Stockfish mode ('eval' for distillation, 'play' for game episodes)")
    p_train.add_argument("--stockfish-depth", type=int, default=6, help="Stockfish search depth (default: 6)")
    p_train.add_argument("--stockfish-time-limit", type=float, default=0.02, help="Stockfish move/eval time limit in seconds (default: 0.02)")
    p_train.add_argument("--stockfish-elo", type=int, default=None, help="Stockfish UCI_Elo rating limit (e.g. 1500, 2000)")
    p_train.add_argument("--num-stockfish-workers", type=int, default=0, help="Concurrent background workers for Stockfish generation (0 = synchronous)")

    # 2. benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark inference latency and GPU throughput")
    p_bench.add_argument("--device", type=str, default="auto", help="Compute device")
    p_bench.add_argument("--hidden-dim", type=int, default=None, help="Uniform hidden layer width")
    p_bench.add_argument("--hidden-dims", type=int, nargs="+", default=None, help="Hidden layer dimensions")
    p_bench.add_argument("--batch-sizes", type=int, nargs="+", default=None, help="List of batch sizes to benchmark")
    p_bench.add_argument("--sparse-residual-enabled", action="store_true", help="Enable sparse residual")
    p_bench.add_argument("--sparse-residual-density", type=float, default=0.01, help="Density")
    p_bench.add_argument("--genius-neuron-dropout-enabled", action="store_true", help="Enable neuron masking")
    p_bench.add_argument("--genius-neuron-dropout-rate", type=float, default=0.05, help="Dropout rate")

    # 3. selfplay
    p_sp = subparsers.add_parser("selfplay", help="Generate self-play games")
    p_sp.add_argument("--num-games", type=int, default=3, help="Number of games to play")
    p_sp.add_argument("--search-depth", type=int, default=2, help="Search depth")
    p_sp.add_argument("--max-moves", type=int, default=100, help="Max moves per game")
    p_sp.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory or file (auto-discovers latest if omitted)")
    p_sp.add_argument("--device", type=str, default="auto", help="Compute device")

    # 4. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model against material baseline")
    p_eval.add_argument("--num-games", type=int, default=4, help="Number of evaluation games")
    p_eval.add_argument("--search-depth", type=int, default=2, help="Search depth")
    p_eval.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory or file (auto-discovers latest if omitted)")
    p_eval.add_argument("--device", type=str, default="auto", help="Compute device")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "selfplay":
        run_selfplay(args)
    elif args.command == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
