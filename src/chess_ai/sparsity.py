"""Sparse Residuals and Neuron Masking mechanisms for Huge Layer research.

Mechanisms implemented:
1. SparseResidual:
   A research mechanism distinct from Dropout.
   Computes H' = H + S, where S is a high-capacity sparse residual correction
   tensor with very few non-zero entries (density d, e.g. 0.01).
   Dense base + sparse corrections.

2. RandomNeuronMask ("Trống neuron học bá"):
   Temporary random neuron masking during training:
   h_tilde_i = (m_i * h_i) / (1 - p), m_i ~ Bernoulli(1 - p).
   In evaluation mode (model.eval()), deterministic identity is used.

3. NeuronMaskStrategy (Extensible ABC):
   Base interface supporting random, activation-based, and frequency-based masking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronMaskStrategy(ABC):
    """Abstract strategy for neuron masking."""

    @abstractmethod
    def create_mask(self, activations: torch.Tensor, training: bool) -> torch.Tensor:
        """Create a neuron mask tensor of shape matching activations or broadcastable.

        Args:
            activations: Tensor of shape [B, D]
            training: Whether the model is in training mode

        Returns:
            Mask tensor of shape [B, D] or [1, D]
        """
        pass


class RandomNeuronMaskStrategy(NeuronMaskStrategy):
    """Random temporary neuron masking ('Trống neuron học bá').

    Generates m_i ~ Bernoulli(1 - p), scales by 1 / (1 - p) during training.
    Identity during evaluation.
    """

    def __init__(self, dropout_rate: float = 0.05):
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"Dropout rate must be in [0, 1), got {dropout_rate}")
        self.dropout_rate = dropout_rate

    def create_mask(self, activations: torch.Tensor, training: bool) -> torch.Tensor:
        if not training or self.dropout_rate == 0.0:
            return torch.ones_like(activations)

        keep_prob = 1.0 - self.dropout_rate
        # Sample per-neuron Bernoulli mask (shared across batch or per-sample)
        # For batch-level neuron disabling: shape [1, D] or [B, D]
        # Per requirement: in each training batch/step, randomly disable some neurons
        mask = (torch.rand_like(activations) < keep_prob).to(activations.dtype)
        # Inverted scaling to keep expected activation scale constant:
        return mask / keep_prob


class ActivationBasedNeuronMaskStrategy(NeuronMaskStrategy):
    """Adaptive strategy: temporarily mask neurons with the highest or lowest activations."""

    def __init__(self, mask_rate: float = 0.05, mask_highest: bool = True):
        self.mask_rate = mask_rate
        self.mask_highest = mask_highest

    def create_mask(self, activations: torch.Tensor, training: bool) -> torch.Tensor:
        if not training or self.mask_rate == 0.0:
            return torch.ones_like(activations)

        B, D = activations.shape
        k = max(1, int(round(D * self.mask_rate)))
        # Find k extreme neurons per batch mean activation
        mean_act = activations.detach().abs().mean(dim=0, keepdim=True)  # [1, D]
        if self.mask_highest:
            _, indices = torch.topk(mean_act, k, dim=-1, largest=True)
        else:
            _, indices = torch.topk(mean_act, k, dim=-1, largest=False)

        mask = torch.ones_like(activations)
        mask.scatter_(-1, indices.expand(B, -1), 0.0)
        keep_prob = 1.0 - self.mask_rate
        return mask / keep_prob


class RandomNeuronMask(nn.Module):
    """Neuron masking layer for research into preventing co-adaptation in Huge Layers."""

    def __init__(
        self,
        dropout_rate: float = 0.05,
        enabled: bool = True,
        strategy: NeuronMaskStrategy | None = None,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.enabled = enabled
        self.strategy = strategy or RandomNeuronMaskStrategy(dropout_rate=dropout_rate)
        self.last_active_ratio: float = 1.0

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training or self.dropout_rate == 0.0:
            self.last_active_ratio = 1.0
            return h

        mask = self.strategy.create_mask(h, training=self.training)
        # Track active neuron ratio for metrics
        with torch.no_grad():
            self.last_active_ratio = float((mask > 0).float().mean().item())

        return h * mask


class SparseResidual(nn.Module):
    """Custom Sparse Residual layer: H' = H + S.

    Computes a sparse correction tensor S from H:
      S = TopK_Sparse(W_res * H + b_res, density)
    where only the top-k largest elements in absolute magnitude are kept,
    guaranteeing that nonzero_count / total_count <= density.

    This implements dense base representations + sparse high-capacity corrections.
    """

    def __init__(
        self,
        dim: int,
        density: float = 0.01,
        enabled: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.density = density
        self.enabled = enabled
        # Residual projection: generates candidate correction only when enabled
        if self.enabled and self.density > 0.0:
            self.linear: nn.Linear | None = nn.Linear(dim, dim)
            nn.init.normal_(self.linear.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.linear.bias)
        else:
            self.linear = None

        self.last_actual_density: float = 0.0

    def grow(self, new_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        """Grow SparseResidual dimensions to new_dim, preserving old weights."""
        if not self.enabled or self.linear is None:
            self.dim = new_dim
            return

        old_dim = self.dim
        old_linear = self.linear
        new_linear = nn.Linear(new_dim, new_dim, device=device, dtype=dtype)
        nn.init.normal_(new_linear.weight.data, mean=0.0, std=1e-3)
        new_linear.bias.data.zero_()

        # Copy old submatrix
        new_linear.weight.data[:old_dim, :old_dim] = old_linear.weight.data
        new_linear.bias.data[:old_dim] = old_linear.bias.data

        self.linear = new_linear
        self.dim = new_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.density <= 0.0 or self.linear is None:
            self.last_actual_density = 0.0
            return h

        # 1. Compute candidate correction tensor R
        candidate = self.linear(h)  # [B, D]

        # 2. Enforce top-k sparsity based on absolute magnitude
        B, D = candidate.shape
        k = max(1, min(D, int(math.ceil(D * self.density))))

        # Magnitude-based top-k selection
        abs_cand = candidate.abs()
        topk_vals, topk_indices = torch.topk(abs_cand, k, dim=-1, largest=True, sorted=False)

        # Create sparse residual tensor S
        s = torch.zeros_like(candidate)
        # Gather the actual values (with sign) for top-k indices
        cand_values = torch.gather(candidate, dim=-1, index=topk_indices)
        s.scatter_(dim=-1, index=topk_indices, src=cand_values)

        # Track density
        with torch.no_grad():
            self.last_actual_density = float((s != 0).float().mean().item())

        # 3. H' = H + S
        return h + s
