"""Checkpoint management with default Hugging Face format (safetensors + config.json) and auto-discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from chess_ai.config import ExperimentConfig
from chess_ai.model import ChessValueNet


def find_latest_checkpoint(search_roots: Optional[List[str | Path]] = None) -> Optional[Path]:
    """Auto-discover the latest checkpoint across experiment directories.

    Searches for:
      1. 'checkpoints/latest' directories (Hugging Face format)
      2. 'checkpoints/checkpoint-step-*' directories
      3. 'checkpoints/latest.pt' or 'checkpoints/step_*.pt' files

    Returns the Path to the most recently modified valid checkpoint, or None.
    """
    if search_roots is None:
        search_roots = ["runs", "checkpoints", "."]

    candidate_checkpoints: List[Tuple[float, Path]] = []

    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue

        # 1. Search for Hugging Face checkpoint directories (contains model.safetensors or config.json)
        for path in root_path.rglob("*"):
            if path.is_dir():
                has_weights = (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
                has_config = (path / "config.json").is_file()
                if has_weights and has_config:
                    mtime = path.stat().st_mtime
                    # Prioritize directories named 'latest' by giving a slight boost if tied
                    candidate_checkpoints.append((mtime, path))

            # 2. Search for legacy .pt checkpoint files
            elif path.is_file() and path.suffix == ".pt":
                mtime = path.stat().st_mtime
                candidate_checkpoints.append((mtime, path))

    if not candidate_checkpoints:
        return None

    # Sort by modification time descending
    candidate_checkpoints.sort(key=lambda x: x[0], reverse=True)
    return candidate_checkpoints[0][1]


class CheckpointManager:
    """Manages Hugging Face style checkpoints (default) and legacy PyTorch checkpoints."""

    def __init__(self, experiment_dir: str | Path, default_format: str = "huggingface"):
        self.experiment_dir = Path(experiment_dir)
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.default_format = default_format

    def save_checkpoint(
        self,
        model: ChessValueNet,
        optimizer: Optional[optim.Optimizer],
        step: int,
        episode: int,
        config: ExperimentConfig,
        dirname: Optional[str] = None,
        is_latest: bool = True,
        save_format: Optional[str] = None,
    ) -> Path:
        """Save a complete checkpoint in Hugging Face format (default) or legacy .pt format.

        In Hugging Face format:
          checkpoints/checkpoint-step-000100/
            ├── config.json          (model architecture)
            ├── model.safetensors    (safe, zero-copy weights)
            ├── trainer_state.json   (step, episode, growth history, metrics)
            └── optimizer.pt         (optimizer state)
          checkpoints/latest/        (copy/mirror of latest checkpoint)
        """
        fmt = save_format or self.default_format

        if fmt == "huggingface":
            # Auto-name checkpoint directory in standard Hugging Face style
            step_dir_name = dirname or f"checkpoint-step-{step:06d}"
            save_path = self.checkpoints_dir / step_dir_name
            save_path.mkdir(parents=True, exist_ok=True)

            # 1. Save model with Hugging Face standard save_pretrained (config.json + model.safetensors)
            model.save_pretrained(save_path)

            # 2. Save trainer state metadata (trainer_state.json)
            param_stats = model.get_parameter_stats()
            trainer_state: Dict[str, Any] = {
                "step": step,
                "episode": episode,
                "parameter_stats": param_stats,
                "growth_history": list(model.growth_history),
                "config": config.to_dict(),
                "seed": config.seed,
            }
            with open(save_path / "trainer_state.json", "w", encoding="utf-8") as f:
                json.dump(trainer_state, f, indent=2)

            # 3. Save optimizer state if present
            if optimizer is not None:
                torch.save(optimizer.state_dict(), save_path / "optimizer.pt")

            # 4. Mirror to 'latest' Hugging Face directory
            if is_latest:
                latest_dir = self.checkpoints_dir / "latest"
                latest_dir.mkdir(parents=True, exist_ok=True)
                for item in save_path.iterdir():
                    dest = latest_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest)

            return save_path

        else:
            # Legacy single .pt file format
            param_stats = model.get_parameter_stats()
            checkpoint_data: Dict[str, Any] = {
                "step": step,
                "episode": episode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "architecture": {
                    "input_dim": model.input_dim,
                    "hidden_dims": list(model.hidden_dims),
                    "sparse_residual_enabled": model.sparse_residual_enabled,
                    "sparse_residual_density": model.sparse_residual_density,
                    "genius_neuron_dropout_enabled": model.genius_neuron_dropout_enabled,
                    "genius_neuron_dropout_rate": model.genius_neuron_dropout_rate,
                    "activation": model.activation,
                    "use_layer_norm": model.use_layer_norm,
                },
                "parameter_stats": param_stats,
                "growth_history": list(model.growth_history),
                "config": config.to_dict(),
                "seed": config.seed,
            }

            save_name = dirname or f"step_{step:06d}.pt"
            save_path = self.checkpoints_dir / save_name
            torch.save(checkpoint_data, save_path)

            if is_latest:
                latest_path = self.checkpoints_dir / "latest.pt"
                torch.save(checkpoint_data, latest_path)

            return save_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: Optional[ChessValueNet] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[ChessValueNet, Dict[str, Any]]:
        """Load checkpoint supporting both Hugging Face format directories and legacy .pt files."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        map_location = device or torch.device("cpu")

        # ---------------------------------------------------------------------
        # Case A: Hugging Face Directory Format
        # ---------------------------------------------------------------------
        if path.is_dir() or (path.is_file() and path.name in ("model.safetensors", "config.json")):
            ckpt_dir = path if path.is_dir() else path.parent
            config_file = ckpt_dir / "config.json"
            if not config_file.is_file():
                raise FileNotFoundError(f"Missing config.json in checkpoint directory: {ckpt_dir}")

            with open(config_file, "r", encoding="utf-8") as f:
                arch = json.load(f)

            saved_hidden_dims = arch["hidden_dims"]

            # Instantiate model if not provided, or sync dimensions
            if model is None:
                model = ChessValueNet.from_pretrained(ckpt_dir, device=map_location)
            else:
                if model.hidden_dims != saved_hidden_dims:
                    model.grow_all_hidden(saved_hidden_dims, reason="checkpoint_load_sync")
                # Load weights into existing model
                loaded_net = ChessValueNet.from_pretrained(ckpt_dir, device=map_location)
                model.load_state_dict(loaded_net.state_dict())
                model.growth_history = list(loaded_net.growth_history)

            # Load trainer_state metadata
            trainer_state: Dict[str, Any] = {}
            trainer_state_file = ckpt_dir / "trainer_state.json"
            if trainer_state_file.is_file():
                with open(trainer_state_file, "r", encoding="utf-8") as f:
                    trainer_state = json.load(f)

            # Load optimizer state if available
            opt_file = ckpt_dir / "optimizer.pt"
            if optimizer is not None and opt_file.is_file():
                try:
                    opt_data = torch.load(opt_file, map_location=map_location)
                    optimizer.load_state_dict(opt_data)
                except Exception:
                    pass

            metadata = {
                "step": trainer_state.get("step", 0),
                "episode": trainer_state.get("episode", 0),
                "growth_history": model.growth_history,
                "config": trainer_state.get("config", {}),
                "parameter_stats": model.get_parameter_stats(),
                "format": "huggingface",
                "checkpoint_path": str(ckpt_dir),
            }
            return model, metadata

        # ---------------------------------------------------------------------
        # Case B: Legacy .pt File Format
        # ---------------------------------------------------------------------
        data = torch.load(path, map_location=map_location)
        arch = data["architecture"]
        saved_hidden_dims = arch["hidden_dims"]

        if model is None:
            model = ChessValueNet(
                input_dim=arch["input_dim"],
                hidden_dims=saved_hidden_dims,
                sparse_residual_enabled=arch.get("sparse_residual_enabled", False),
                sparse_residual_density=arch.get("sparse_residual_density", 0.01),
                genius_neuron_dropout_enabled=arch.get("genius_neuron_dropout_enabled", False),
                genius_neuron_dropout_rate=arch.get("genius_neuron_dropout_rate", 0.05),
                activation=arch.get("activation", "relu"),
                use_layer_norm=arch.get("use_layer_norm", False),
            )
        elif model.hidden_dims != saved_hidden_dims:
            model.grow_all_hidden(saved_hidden_dims, reason="checkpoint_load_sync")

        model.load_state_dict(data["model_state_dict"])
        model.growth_history = list(data.get("growth_history", []))

        if optimizer is not None and data.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(data["optimizer_state_dict"])
            except Exception:
                pass

        if device is not None:
            model.to(device)

        metadata = {
            "step": data.get("step", 0),
            "episode": data.get("episode", 0),
            "growth_history": model.growth_history,
            "config": data.get("config", {}),
            "parameter_stats": model.get_parameter_stats(),
            "format": "legacy_pt",
            "checkpoint_path": str(path),
        }
        return model, metadata
