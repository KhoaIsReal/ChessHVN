"""Huge Layer Block implementation for wide value networks."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

from chess_ai.sparsity import RandomNeuronMask, SparseResidual


class HugeLayerBlock(nn.Module):
    """A building block for wide/huge neural network representations.

    Composition:
      Input x
        ↓
      Linear(in_dim, out_dim)
        ↓
      Activation (ReLU / SiLU / GELU)
        ↓
      Optional LayerNorm
        ↓
      Optional RandomNeuronMask ("Trống neuron học bá")
        ↓
      Optional SparseResidual (H' = H + S)
        ↓
      Output
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "relu",
        use_layer_norm: bool = False,
        sparse_residual_enabled: bool = False,
        sparse_residual_density: float = 0.01,
        genius_neuron_dropout_enabled: bool = False,
        genius_neuron_dropout_rate: float = 0.05,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_name = activation
        self.use_layer_norm = use_layer_norm

        self.linear = nn.Linear(in_dim, out_dim)
        self._init_weights(self.linear)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.layer_norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()

        self.neuron_mask = RandomNeuronMask(
            dropout_rate=genius_neuron_dropout_rate,
            enabled=genius_neuron_dropout_enabled,
        )

        self.sparse_residual = SparseResidual(
            dim=out_dim,
            density=sparse_residual_density,
            enabled=sparse_residual_enabled,
        )

    def _init_weights(self, layer: nn.Linear) -> None:
        """Initialize weights with Kaiming uniform and zeros for biases."""
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.act(h)
        h = self.layer_norm(h)
        h = self.neuron_mask(h)
        h = self.sparse_residual(h)
        return h

    @property
    def active_neuron_ratio(self) -> float:
        return self.neuron_mask.last_active_ratio

    @property
    def actual_sparse_density(self) -> float:
        return self.sparse_residual.last_actual_density
