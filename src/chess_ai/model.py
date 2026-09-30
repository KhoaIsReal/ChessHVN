"""Chess Value Network architecture with Huge Layer support, dynamic growth, and parameter tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import safetensors.torch
import torch
import torch.nn as nn

from chess_ai.huge_layer import HugeLayerBlock


class ChessValueNet(nn.Module):
    """Deep Chess Value Network with scalable Huge Layers.

    Architecture:
      Board Tensor [B, input_dim]
            ↓
      Huge Layer Block 1 [B, hidden_dims[0]]
            ↓
      Huge Layer Block 2 [B, hidden_dims[1]]
            ↓
      ...
            ↓
      Value Head Linear [B, 1]
            ↓
      tanh(z) -> V(s) in [-1.0, +1.0]

    Value Convention:
      +1.0 = White winning
       0.0 = Draw
      -1.0 = Black winning (White losing)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        sparse_residual_enabled: bool = False,
        sparse_residual_density: float = 0.01,
        genius_neuron_dropout_enabled: bool = False,
        genius_neuron_dropout_rate: float = 0.05,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [4096, 4096]
        self.sparse_residual_enabled = sparse_residual_enabled
        self.sparse_residual_density = sparse_residual_density
        self.genius_neuron_dropout_enabled = genius_neuron_dropout_enabled
        self.genius_neuron_dropout_rate = genius_neuron_dropout_rate
        self.activation = activation
        self.use_layer_norm = use_layer_norm

        self.growth_history: List[Dict[str, Any]] = []

        # Construct Huge Layers
        self.blocks = nn.ModuleList()
        current_dim = input_dim
        for h_dim in self.hidden_dims:
            block = HugeLayerBlock(
                in_dim=current_dim,
                out_dim=h_dim,
                activation=activation,
                use_layer_norm=use_layer_norm,
                sparse_residual_enabled=sparse_residual_enabled,
                sparse_residual_density=sparse_residual_density,
                genius_neuron_dropout_enabled=genius_neuron_dropout_enabled,
                genius_neuron_dropout_rate=genius_neuron_dropout_rate,
            )
            self.blocks.append(block)
            current_dim = h_dim

        # Value Head: Linear -> tanh
        self.value_head = nn.Linear(current_dim, 1)
        nn.init.kaiming_uniform_(self.value_head.weight, nonlinearity="linear")
        if self.value_head.bias is not None:
            nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch forward pass: [B, input_dim] -> [B, 1]."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = x
        for block in self.blocks:
            h = block(h)

        z = self.value_head(h)
        v = torch.tanh(z)
        return v

    def grow_hidden(self, new_hidden_dim: int, layer_idx: int = -1, reason: str = "manual") -> Dict[str, Any]:
        """Dynamically grow hidden dimension of a layer, preserving existing weights.

        Uses zero-initialization for new outgoing weights (Net2Net style), ensuring
        that the model's output f_new(x) == f_old(x) at the instant of growth.
        """
        if layer_idx < 0:
            layer_idx = len(self.blocks) + layer_idx
        if not (0 <= layer_idx < len(self.blocks)):
            raise IndexError(f"Layer index {layer_idx} out of range (0..{len(self.blocks)-1})")

        old_dim = self.hidden_dims[layer_idx]
        if new_hidden_dim <= old_dim:
            raise ValueError(f"New hidden dim {new_hidden_dim} must be greater than current dim {old_dim}")

        old_stats = self.get_parameter_stats()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        target_block: HugeLayerBlock = self.blocks[layer_idx]
        in_dim = target_block.in_dim

        # 1. Expand target block linear layer: [old_dim, in_dim] -> [new_hidden_dim, in_dim]
        old_w = target_block.linear.weight.data
        old_b = target_block.linear.bias.data if target_block.linear.bias is not None else None

        new_linear = nn.Linear(in_dim, new_hidden_dim, bias=(old_b is not None), device=device, dtype=dtype)
        # Preserve old weights
        new_linear.weight.data[:old_dim, :] = old_w
        # Initialize new neurons (incoming connections)
        nn.init.kaiming_uniform_(new_linear.weight.data[old_dim:, :], nonlinearity="relu")

        if old_b is not None:
            new_linear.bias.data[:old_dim] = old_b
            new_linear.bias.data[old_dim:].zero_()

        target_block.linear = new_linear
        target_block.out_dim = new_hidden_dim

        # 2. Expand LayerNorm if active
        if target_block.use_layer_norm:
            old_ln = target_block.layer_norm
            new_ln = nn.LayerNorm(new_hidden_dim, device=device, dtype=dtype)
            new_ln.weight.data[:old_dim] = old_ln.weight.data
            new_ln.bias.data[:old_dim] = old_ln.bias.data
            target_block.layer_norm = new_ln

        # 3. Expand SparseResidual if present
        target_block.sparse_residual.grow(new_hidden_dim, device=device, dtype=dtype)

        # 4. Expand next layer's input connections: [next_out, old_dim] -> [next_out, new_hidden_dim]
        # Outgoing connections from new neurons are set to 0 to preserve output identically!
        if layer_idx + 1 < len(self.blocks):
            next_target = self.blocks[layer_idx + 1]
            next_linear = next_target.linear
            next_out = next_linear.out_features
            new_next_linear = nn.Linear(new_hidden_dim, next_out, bias=(next_linear.bias is not None), device=device, dtype=dtype)
            new_next_linear.weight.data[:, :old_dim] = next_linear.weight.data
            new_next_linear.weight.data[:, old_dim:].zero_()  # Zero outgoing weights!
            if next_linear.bias is not None:
                new_next_linear.bias.data = next_linear.bias.data
            next_target.linear = new_next_linear
            next_target.in_dim = new_hidden_dim
        else:
            # Value head connects to this layer
            next_linear = self.value_head
            next_out = next_linear.out_features  # 1
            new_value_head = nn.Linear(new_hidden_dim, next_out, bias=(next_linear.bias is not None), device=device, dtype=dtype)
            new_value_head.weight.data[:, :old_dim] = next_linear.weight.data
            new_value_head.weight.data[:, old_dim:].zero_()  # Zero outgoing weights!
            if next_linear.bias is not None:
                new_value_head.bias.data = next_linear.bias.data
            self.value_head = new_value_head

        self.hidden_dims[layer_idx] = new_hidden_dim
        new_stats = self.get_parameter_stats()

        event = {
            "layer_idx": layer_idx,
            "old_dim": old_dim,
            "new_dim": new_hidden_dim,
            "old_params": old_stats["total_params"],
            "new_params": new_stats["total_params"],
            "reason": reason,
        }
        self.growth_history.append(event)
        return event

    def grow_all_hidden(self, new_dims: List[int], reason: str = "manual") -> List[Dict[str, Any]]:
        """Grow all hidden layers sequentially."""
        if len(new_dims) != len(self.hidden_dims):
            raise ValueError(f"Length of new_dims ({len(new_dims)}) must match hidden_dims ({len(self.hidden_dims)})")

        events = []
        for i, new_dim in enumerate(new_dims):
            if new_dim > self.hidden_dims[i]:
                evt = self.grow_hidden(new_dim, layer_idx=i, reason=reason)
                events.append(evt)
        return events

    def get_parameter_stats(self) -> Dict[str, Any]:
        """Compute exact parameter counts and memory estimates across precisions."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Byte calculations:
        # FP32: 4 bytes
        # FP16: 2 bytes
        # BF16: 2 bytes
        # INT16: 2 bytes
        fp32_bytes = total_params * 4
        fp16_bytes = total_params * 2
        bf16_bytes = total_params * 2
        int16_bytes = total_params * 2

        def _format_mb(b: int) -> str:
            return f"{b / (1024 * 1024):.2f} MB"

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "fp32_bytes": fp32_bytes,
            "fp16_bytes": fp16_bytes,
            "bf16_bytes": bf16_bytes,
            "int16_bytes": int16_bytes,
            "fp32_str": _format_mb(fp32_bytes),
            "fp16_str": _format_mb(fp16_bytes),
            "bf16_str": _format_mb(bf16_bytes),
            "int16_str": _format_mb(int16_bytes),
        }

    def to_int16_state_dict(self) -> Dict[str, Any]:
        """Export model weights quantized to INT16 with per-tensor scale factor.

        q = round(x / scale)
        scale = max(|x|) / 32767.0
        x_approx = q * scale
        """
        int16_state: Dict[str, Any] = {}
        for name, param in self.state_dict().items():
            tensor = param.detach().cpu().float()
            max_val = float(tensor.abs().max().item())
            scale = max_val / 32767.0 if max_val > 0.0 else 1.0
            quantized = torch.clamp(torch.round(tensor / scale), -32768, 32767).to(torch.int16)
            int16_state[name] = {
                "qweight": quantized,
                "scale": scale,
                "shape": list(param.shape),
            }
        return int16_state

    def load_int16_state_dict(self, int16_state: Dict[str, Any]) -> None:
        """Dequantize INT16 weights back to FP32 and load into model."""
        fp32_state: Dict[str, torch.Tensor] = {}
        for name, item in int16_state.items():
            q = item["qweight"].float()
            scale = float(item["scale"])
            dequantized = q * scale
            fp32_state[name] = dequantized
        self.load_state_dict(fp32_state)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save model in standard Hugging Face style directory (config.json + model.safetensors)."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. config.json
        config_dict: Dict[str, Any] = {
            "model_type": "chess_value_net",
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "sparse_residual_enabled": self.sparse_residual_enabled,
            "sparse_residual_density": self.sparse_residual_density,
            "genius_neuron_dropout_enabled": self.genius_neuron_dropout_enabled,
            "genius_neuron_dropout_rate": self.genius_neuron_dropout_rate,
            "activation": self.activation,
            "use_layer_norm": self.use_layer_norm,
            "growth_history": list(self.growth_history),
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        # 2. model.safetensors (zero-copy, safe weights)
        weights_path = save_dir / "model.safetensors"
        # Ensure contiguous cpu/gpu tensors
        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        safetensors.torch.save_file(state_dict, str(weights_path))

        return save_dir

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> ChessValueNet:
        """Load model from standard Hugging Face directory format (config.json + model.safetensors)."""
        model_dir = Path(pretrained_model_path)
        if model_dir.is_file():
            model_dir = model_dir.parent

        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"config.json not found in directory: {model_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = cls(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            sparse_residual_enabled=cfg.get("sparse_residual_enabled", False),
            sparse_residual_density=cfg.get("sparse_residual_density", 0.01),
            genius_neuron_dropout_enabled=cfg.get("genius_neuron_dropout_enabled", False),
            genius_neuron_dropout_rate=cfg.get("genius_neuron_dropout_rate", 0.05),
            activation=cfg.get("activation", "relu"),
            use_layer_norm=cfg.get("use_layer_norm", False),
        )
        model.growth_history = list(cfg.get("growth_history", []))

        safetensors_path = model_dir / "model.safetensors"
        bin_path = model_dir / "pytorch_model.bin"

        if safetensors_path.is_file():
            state_dict = safetensors.torch.load_file(str(safetensors_path))
            model.load_state_dict(state_dict)
        elif bin_path.is_file():
            state_dict = torch.load(bin_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(
                f"No weights file (model.safetensors or pytorch_model.bin) found in: {model_dir}"
            )

        if device is not None:
            model.to(device)

        return model
