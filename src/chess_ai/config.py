"""Configuration definitions for Chess AI research engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for ChessValueNet and Huge Layer."""
    hidden_dims: List[int] = field(default_factory=lambda: [4096, 4096])
    sparse_residual_enabled: bool = False
    sparse_residual_density: float = 0.01
    genius_neuron_dropout_enabled: bool = False
    genius_neuron_dropout_rate: float = 0.05
    activation: str = "relu"
    use_layer_norm: bool = False


@dataclass
class TDConfig:
    """Configuration for Temporal-Difference learning."""
    td_method: str = "td0"  # 'td0' | 'td_lambda'
    gamma: float = 1.0
    lambda_: float = 0.8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0


@dataclass
class SearchConfig:
    """Configuration for Negamax with Alpha-Beta pruning."""
    depth: int = 2
    evaluation_mode: str = "neural"  # 'neural' | 'material'
    move_ordering: bool = True


@dataclass
class GrowthConfig:
    """Configuration for Dynamic Parameter Growth."""
    growth_mode: str = "plateau"  # 'plateau' (default) | 'manual' | 'adaptive'
    patience: int = 10
    threshold: float = 0.001
    growth_increment: int = 2048
    max_hidden_dim: int = 32768


@dataclass
class SelfPlayConfig:
    """Configuration for self-play games."""
    max_moves_per_game: int = 200
    temperature: float = 0.0
    exploration_moves: int = 4


@dataclass
class MultiSourceConfig:
    """Configuration for training simultaneously from multiple data streams."""
    enabled_sources: List[str] = field(default_factory=lambda: ["selfplay"])
    source_weights: Dict[str, float] = field(default_factory=lambda: {"selfplay": 1.0})
    pgn_path: Optional[str] = None
    fen_path: Optional[str] = None
    stockfish_path: Optional[str] = None
    stockfish_mode: str = "eval"  # 'eval' | 'play'
    stockfish_depth: int = 6
    stockfish_time_limit: float = 0.02
    stockfish_elo: Optional[int] = None
    num_selfplay_workers: int = 1
    num_stockfish_workers: int = 0
    buffer_capacity_per_source: int = 20000


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "experiment_001"
    device: str = "auto"  # 'auto' | 'cuda' | 'cpu'
    seed: int = 42
    episodes: int = 20
    batch_size: int = 64
    replay_buffer_size: int = 20000
    checkpoint_interval: int = 5
    log_interval: int = 1
    output_dir: str = "runs"

    model: ModelConfig = field(default_factory=ModelConfig)
    td: TDConfig = field(default_factory=TDConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    sources: MultiSourceConfig = field(default_factory=MultiSourceConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        data = data.copy()
        model_data = data.pop("model", {})
        td_data = data.pop("td", {})
        search_data = data.pop("search", {})
        growth_data = data.pop("growth", {})
        selfplay_data = data.pop("selfplay", {})
        sources_data = data.pop("sources", {})

        return cls(
            **data,
            model=ModelConfig(**model_data),
            td=TDConfig(**td_data),
            search=SearchConfig(**search_data),
            growth=GrowthConfig(**growth_data),
            selfplay=SelfPlayConfig(**selfplay_data),
            sources=MultiSourceConfig(**sources_data),
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> ExperimentConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
