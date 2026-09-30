"""Unit tests for Sparse Residual and Random Neuron Masking."""

import pytest
import torch

from chess_ai.model import ChessValueNet
from chess_ai.sparsity import RandomNeuronMask, SparseResidual


def test_random_neuron_mask_training_vs_eval():
    """Verify different random masks during training, deterministic in eval."""
    dim = 256
    layer = RandomNeuronMask(dropout_rate=0.2, enabled=True)

    h = torch.ones(8, dim)

    # In eval mode: same batch repeated -> identical result, no masking
    layer.eval()
    out_eval1 = layer(h)
    out_eval2 = layer(h)
    assert torch.equal(out_eval1, out_eval2), "Eval mode must be deterministic"
    assert torch.equal(out_eval1, h), "Eval mode must leave input unchanged"

    # In train mode: same batch repeated -> different masks
    layer.train()
    out_train1 = layer(h)
    out_train2 = layer(h)
    assert not torch.equal(out_train1, out_train2), "Training mode must produce different random masks"


def test_sparse_residual_density_and_addition():
    """Verify H' = H + S and nonzero_count / total_count matches target density."""
    dim = 1000
    density = 0.05
    sparse_res = SparseResidual(dim=dim, density=density, enabled=True)

    h = torch.randn(10, dim)
    h_out = sparse_res(h)

    # 1. Check residual relation H' = H + S -> S = H' - H
    s = h_out - h

    # 2. Check sparsity of S
    total_elements = s.numel()
    nonzero_elements = (s != 0).sum().item()
    actual_density = nonzero_elements / total_elements

    assert abs(actual_density - density) <= 0.01, f"Expected density ~{density}, got {actual_density}"
    assert abs(sparse_res.last_actual_density - actual_density) < 1e-5


def test_sparse_residual_disabled():
    """When disabled, sparse residual should be an identity (H' = H)."""
    dim = 128
    sparse_res = SparseResidual(dim=dim, density=0.05, enabled=False)

    h = torch.randn(4, dim)
    h_out = sparse_res(h)
    assert torch.equal(h, h_out)


def test_independent_features_in_model():
    """Verify that Sparse Residual and Neuron Mask can be enabled independently in ChessValueNet."""
    # 1. Neither enabled
    m0 = ChessValueNet(input_dim=64, hidden_dims=[64, 64], sparse_residual_enabled=False, genius_neuron_dropout_enabled=False)
    # 2. Only Sparse Residual
    m1 = ChessValueNet(input_dim=64, hidden_dims=[64, 64], sparse_residual_enabled=True, genius_neuron_dropout_enabled=False)
    # 3. Only Neuron Mask
    m2 = ChessValueNet(input_dim=64, hidden_dims=[64, 64], sparse_residual_enabled=False, genius_neuron_dropout_enabled=True)
    # 4. Both enabled
    m3 = ChessValueNet(input_dim=64, hidden_dims=[64, 64], sparse_residual_enabled=True, genius_neuron_dropout_enabled=True)

    x = torch.randn(4, 64)
    for m in [m0, m1, m2, m3]:
        out = m(x)
        assert out.shape == (4, 1)
