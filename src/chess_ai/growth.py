"""Dynamic Parameter Growth management and criteria for expanding neural capacity."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from chess_ai.config import GrowthConfig
from chess_ai.model import ChessValueNet


class DynamicGrowthManager:
    """Manages neural network capacity expansion without retraining from scratch."""

    def __init__(self, model: ChessValueNet, optimizer: Optional[optim.Optimizer] = None):
        self.model = model
        self.optimizer = optimizer

    def grow_layer(
        self,
        layer_idx: int,
        new_dim: int,
        reason: str = "manual",
    ) -> Dict[str, Any]:
        """Grow a specific layer and update optimizer state."""
        event = self.model.grow_hidden(new_dim, layer_idx=layer_idx, reason=reason)

        # Update optimizer state for new parameters if optimizer is present
        if self.optimizer is not None:
            self._update_optimizer_state()

        return event

    def grow_all(
        self,
        new_dims: List[int],
        reason: str = "manual",
    ) -> List[Dict[str, Any]]:
        """Grow all hidden layers and update optimizer state."""
        events = self.model.grow_all_hidden(new_dims, reason=reason)
        if self.optimizer is not None:
            self._update_optimizer_state()
        return events

    def _update_optimizer_state(self) -> None:
        """Adapt optimizer internal state (e.g. Adam momentum/variance) to new parameter tensors.

        Preserves existing momentum buffers for old weights and initializes new entries with 0.
        """
        if self.optimizer is None:
            return

        old_state = self.optimizer.state
        param_groups = self.optimizer.param_groups

        # Collect current parameter references
        current_params = list(self.model.parameters())

        # Re-initialize optimizer with new parameters while preserving hyperparams
        new_param_groups = []
        for pg in param_groups:
            new_pg = pg.copy()
            new_pg["params"] = current_params
            new_param_groups.append(new_pg)

        # Reconstruct optimizer
        opt_cls = type(self.optimizer)
        lr = param_groups[0].get("lr", 1e-4)
        weight_decay = param_groups[0].get("weight_decay", 0.0)

        new_optimizer = opt_cls(current_params, lr=lr, weight_decay=weight_decay)

        # Match old parameters to new parameters by index
        old_params = list(old_state.keys())
        for idx, p_new in enumerate(current_params):
            if idx < len(old_params):
                p_old = old_params[idx]
                p_state = old_state.get(p_old, {})
                adapted_state: Dict[str, Any] = {}
                for k, v in p_state.items():
                    if isinstance(v, torch.Tensor):
                        if v.ndim == 0:
                            # Scalar tensor like step count in Adam
                            adapted_state[k] = v.clone()
                        elif v.shape == p_new.shape:
                            adapted_state[k] = v.clone()
                        elif v.ndim == p_new.ndim:
                            new_v = torch.zeros_like(p_new.data)
                            slices = tuple(slice(0, min(s_old, s_new)) for s_old, s_new in zip(v.shape, p_new.shape))
                            new_v[slices] = v[slices]
                            adapted_state[k] = new_v
                        else:
                            adapted_state[k] = v.clone()
                    else:
                        adapted_state[k] = v
                if adapted_state:
                    new_optimizer.state[p_new] = adapted_state

        self.optimizer.state.clear()
        self.optimizer.state.update(new_optimizer.state)
        self.optimizer.param_groups = new_optimizer.param_groups


class GrowthController:
    """Monitors metrics and triggers dynamic capacity expansion based on criteria."""

    def __init__(
        self,
        growth_manager: DynamicGrowthManager,
        config: Optional[GrowthConfig] = None,
    ):
        self.manager = growth_manager
        self.config = config or GrowthConfig()
        self.best_metric: float = float("inf")
        self.steps_without_improvement: int = 0
        self.growth_events: List[Dict[str, Any]] = []

    def step(self, current_metric: float, step_idx: int) -> Optional[List[Dict[str, Any]]]:
        """Check growth condition (e.g. plateau on TD error or loss) and trigger growth if met.

        Args:
          current_metric: float (e.g. mean absolute TD error or loss)
          step_idx: current training step

        Returns:
          List of growth event records if growth occurred, else None.
        """
        if self.config.growth_mode == "manual":
            return None

        # Check for improvement
        if current_metric < (self.best_metric - self.config.threshold):
            self.best_metric = current_metric
            self.steps_without_improvement = 0
            return None

        self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.config.patience:
            # Trigger growth
            current_dims = list(self.manager.model.hidden_dims)
            new_dims = [
                min(self.config.max_hidden_dim, d + self.config.growth_increment)
                for d in current_dims
            ]

            # Only grow if at least one dimension increases
            if any(n > c for n, c in zip(new_dims, current_dims)):
                reason = f"plateau_at_step_{step_idx}_metric_{current_metric:.5f}"
                events = self.manager.grow_all(new_dims, reason=reason)
                self.growth_events.extend(events)
                self.steps_without_improvement = 0
                self.best_metric = current_metric
                return events

        return None
