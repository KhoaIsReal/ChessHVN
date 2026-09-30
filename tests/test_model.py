"""Unit tests for ChessValueNet architecture, batching, and parameter counting."""

import pytest
import torch

from chess_ai.model import ChessValueNet


def test_model_forward_shape():
    input_dim = 1216
    batch_size = 8
    model = ChessValueNet(input_dim=input_dim, hidden_dims=[64, 64])

    x = torch.randn(batch_size, input_dim)
    y = model(x)

    assert y.shape == (batch_size, 1)
    assert torch.all(y >= -1.0) and torch.all(y <= 1.0), "tanh output must be in [-1, 1]"


def test_model_single_input():
    input_dim = 1216
    model = ChessValueNet(input_dim=input_dim, hidden_dims=[32, 32])

    x = torch.randn(input_dim)
    y = model(x)

    assert y.shape == (1, 1)


def test_model_parameter_stats():
    input_dim = 100
    model = ChessValueNet(input_dim=input_dim, hidden_dims=[200, 100])
    stats = model.get_parameter_stats()

    # Layer 0: 100*200 + 200 = 20200
    # Layer 1: 200*100 + 100 = 20100
    # Head: 100*1 + 1 = 101
    # Total = 40401
    assert stats["total_params"] == 40401
    assert stats["trainable_params"] == 40401
    assert stats["fp32_bytes"] == 40401 * 4
    assert stats["fp16_bytes"] == 40401 * 2
    assert stats["int16_bytes"] == 40401 * 2
    assert "MB" in stats["fp32_str"]


def test_model_int16_quantization_reconstruction():
    input_dim = 128
    model = ChessValueNet(input_dim=input_dim, hidden_dims=[128, 64])

    x = torch.randn(4, input_dim)
    orig_output = model(x).detach()

    # Quantize to INT16 state dict and reload
    q_state = model.to_int16_state_dict()
    model.load_int16_state_dict(q_state)
    q_output = model(x).detach()

    # Reconstructed INT16 output should closely match FP32 (relative error < 1%)
    diff = torch.abs(orig_output - q_output).max().item()
    assert diff < 0.05, f"INT16 reconstruction error too high: {diff}"
