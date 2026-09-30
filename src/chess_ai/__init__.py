"""Chess AI: Research Prototype for Huge Layer Chess Engine."""

__version__ = "0.1.0"

from chess_ai.board_encoder import BoardEncoder
from chess_ai.config import ExperimentConfig, ModelConfig, TDConfig, SearchConfig, GrowthConfig
from chess_ai.model import ChessValueNet
from chess_ai.huge_layer import HugeLayerBlock
from chess_ai.sparsity import SparseResidual, RandomNeuronMask, NeuronMaskStrategy
from chess_ai.td_learning import TDLearner
from chess_ai.search import AlphaBetaSearch, NegamaxSearchResult
from chess_ai.selfplay import SelfPlayEngine
from chess_ai.replay import ReplayBuffer, Transition, Episode
from chess_ai.growth import DynamicGrowthManager, GrowthController
from chess_ai.checkpoint import CheckpointManager, find_latest_checkpoint
from chess_ai.config import ExperimentConfig, ModelConfig, TDConfig, SearchConfig, GrowthConfig, MultiSourceConfig
from chess_ai.evaluation import evaluate_positions
from chess_ai.sources import BaseDataSource, SelfPlaySource, PGNSource, FENSource, MultiSourceDataManager

__all__ = [
    "BoardEncoder",
    "ExperimentConfig",
    "ModelConfig",
    "TDConfig",
    "SearchConfig",
    "GrowthConfig",
    "MultiSourceConfig",
    "ChessValueNet",
    "HugeLayerBlock",
    "SparseResidual",
    "RandomNeuronMask",
    "NeuronMaskStrategy",
    "TDLearner",
    "AlphaBetaSearch",
    "NegamaxSearchResult",
    "SelfPlayEngine",
    "ReplayBuffer",
    "Transition",
    "Episode",
    "DynamicGrowthManager",
    "GrowthController",
    "CheckpointManager",
    "find_latest_checkpoint",
    "evaluate_positions",
    "BaseDataSource",
    "SelfPlaySource",
    "PGNSource",
    "FENSource",
    "MultiSourceDataManager",
]
