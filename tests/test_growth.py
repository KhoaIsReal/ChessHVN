"""Unit tests for Dynamic Parameter Growth, weight preservation, and optimizer adaptation."""

import pytest
import torch
import torch.optim as optim

from chess_ai.checkpoint import CheckpointManager
from chess_ai.config import ExperimentConfig, GrowthConfig
from chess_ai.growth import DynamicGrowthManager, GrowthController
from chess_ai.model import ChessValueNet


def test_growth_parameter_increase():
    model = ChessValueNet(input_dim=64, hidden_dims=[128, 64])
    old_stats = model.get_parameter_stats()

    event = model.grow_hidden(new_hidden_dim=256, layer_idx=0, reason="test_growth")
    new_stats = model.get_parameter_stats()

    assert new_stats["total_params"] > old_stats["total_params"]
    assert event["new_dim"] == 256
    assert event["old_dim"] == 128
    assert len(model.growth_history) == 1


def test_growth_weight_and_function_preservation():
    """Verify that existing weights are retained and model output is mathematically identical post-growth."""
    input_dim = 64
    model = ChessValueNet(input_dim=input_dim, hidden_dims=[64, 32])

    old_w0 = model.blocks[0].linear.weight.clone()
    old_b0 = model.blocks[0].linear.bias.clone()

    x = torch.randn(5, input_dim)
    output_before = model(x).detach().clone()

    # Grow layer 0 from 64 to 128
    model.grow_hidden(new_hidden_dim=128, layer_idx=0)

    # 1. Check old slice of weights in layer 0 is preserved
    assert torch.equal(model.blocks[0].linear.weight[:64, :], old_w0)
    assert torch.equal(model.blocks[0].linear.bias[:64], old_b0)

    # 2. Check function preservation (outgoing weights from new neurons are zero)
    output_after = model(x).detach()
    diff = torch.abs(output_before - output_after).max().item()
    assert diff < 1e-5, f"Function output changed upon growth! Max diff: {diff}"


def test_growth_optimizer_adaptation():
    """Verify optimizer state adapts smoothly after layer growth."""
    model = ChessValueNet(input_dim=32, hidden_dims=[32, 16])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    manager = DynamicGrowthManager(model=model, optimizer=optimizer)

    # Run a gradient step so optimizer populates state buffers (exp_avg, exp_avg_sq)
    x = torch.randn(4, 32)
    loss = model(x).mean()
    loss.backward()
    optimizer.step()

    # Grow layer
    manager.grow_layer(layer_idx=0, new_dim=64, reason="test_opt")

    # Run another gradient step post-growth
    optimizer.zero_grad()
    loss_post = model(x).mean()
    loss_post.backward()
    optimizer.step()

    assert model.hidden_dims[0] == 64


def test_checkpoint_after_growth(tmp_path):
    """Verify checkpoint save and reload after dynamic growth."""
    exp_dir = tmp_path / "exp_test"
    mgr = CheckpointManager(exp_dir)
    cfg = ExperimentConfig(name="test")

    model = ChessValueNet(input_dim=32, hidden_dims=[32, 16])
    model.grow_hidden(64, layer_idx=0, reason="test")

    ckpt_path = mgr.save_checkpoint(model, optimizer=None, step=10, episode=2, config=cfg)

    # Load into fresh instance
    loaded_model, data = mgr.load_checkpoint(ckpt_path)
    assert loaded_model.hidden_dims == [64, 16]
    assert data["step"] == 10
    assert len(loaded_model.growth_history) == 1
