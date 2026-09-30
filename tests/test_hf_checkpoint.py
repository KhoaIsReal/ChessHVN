"""Tests for Hugging Face style saving (safetensors + config.json) and auto-checkpoint discovery."""

from pathlib import Path
import pytest
import torch
import torch.optim as optim

from chess_ai.checkpoint import CheckpointManager, find_latest_checkpoint
from chess_ai.config import ExperimentConfig
from chess_ai.model import ChessValueNet


def test_model_save_and_from_pretrained(tmp_path):
    """Verify Hugging Face save_pretrained and from_pretrained with safetensors."""
    save_dir = tmp_path / "hf_chess_model"
    model = ChessValueNet(input_dim=64, hidden_dims=[128, 64], sparse_residual_enabled=True, genius_neuron_dropout_enabled=True)

    # Grow model so growth_history is populated
    model.grow_hidden(256, layer_idx=0, reason="test_growth")

    # Forward pass before saving
    x = torch.randn(4, 64)
    model.eval()
    y_before = model(x).detach()

    # Save pretrained
    saved_path = model.save_pretrained(save_dir)
    assert saved_path == save_dir
    assert (save_dir / "config.json").is_file()
    assert (save_dir / "model.safetensors").is_file()

    # Load pretrained
    loaded_model = ChessValueNet.from_pretrained(save_dir)
    loaded_model.eval()
    y_after = loaded_model(x).detach()

    assert loaded_model.input_dim == 64
    assert loaded_model.hidden_dims == [256, 64]
    assert loaded_model.sparse_residual_enabled is True
    assert loaded_model.genius_neuron_dropout_enabled is True
    assert len(loaded_model.growth_history) == 1
    assert torch.allclose(y_before, y_after, atol=1e-5), "Loaded weights must produce identical output"


def test_checkpoint_manager_hf_default(tmp_path):
    """Verify CheckpointManager saves in Hugging Face directory format by default."""
    exp_dir = tmp_path / "test_run"
    mgr = CheckpointManager(exp_dir)

    model = ChessValueNet(input_dim=32, hidden_dims=[64, 32])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    cfg = ExperimentConfig(name="test_run")

    ckpt_path = mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=50,
        episode=5,
        config=cfg,
    )

    assert ckpt_path.is_dir(), "Default checkpoint must be a Hugging Face directory"
    assert ckpt_path.name == "checkpoint-step-000050"
    assert (ckpt_path / "model.safetensors").is_file()
    assert (ckpt_path / "config.json").is_file()
    assert (ckpt_path / "trainer_state.json").is_file()
    assert (ckpt_path / "optimizer.pt").is_file()

    # Verify latest directory exists and is also Hugging Face format
    latest_dir = exp_dir / "checkpoints" / "latest"
    assert latest_dir.is_dir()
    assert (latest_dir / "model.safetensors").is_file()
    assert (latest_dir / "config.json").is_file()

    # Load from directory
    loaded_model, meta = mgr.load_checkpoint(ckpt_path)
    assert loaded_model.hidden_dims == [64, 32]
    assert meta["step"] == 50
    assert meta["format"] == "huggingface"


def test_find_latest_checkpoint_auto_discovery(tmp_path):
    """Verify find_latest_checkpoint correctly locates the newest checkpoint."""
    exp1 = tmp_path / "exp_1"
    exp2 = tmp_path / "exp_2"

    mgr1 = CheckpointManager(exp1)
    mgr2 = CheckpointManager(exp2)

    model = ChessValueNet(input_dim=16, hidden_dims=[32, 16])
    cfg = ExperimentConfig()

    # Save older checkpoint in exp1
    mgr1.save_checkpoint(model, optimizer=None, step=10, episode=1, config=cfg)

    # Save newer checkpoint in exp2
    mgr2.save_checkpoint(model, optimizer=None, step=20, episode=2, config=cfg)

    latest_discovered = find_latest_checkpoint(search_roots=[tmp_path])
    assert latest_discovered is not None
    # Latest should be from exp2 or latest directory
    assert "exp_2" in str(latest_discovered)
