"""Temporal-Difference Learning (TD(0) and TD(lambda)) for Chess Value Network."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from chess_ai.config import TDConfig


class TDLearner:
    """Manages Temporal-Difference learning for ChessValueNet.

    Supports:
      1. TD(0): One-step TD update with target y_t = r_{t+1} + (1 - done) * gamma * V(s_{t+1}).
      2. Forward-View TD(lambda): Computes exact multi-step lambda-returns over trajectories:
           G_t^lambda = r_{t+1} + gamma * ((1 - lambda) * V(s_{t+1}) + lambda * G_{t+1}^lambda)
         which is the trajectory-level equivalent of offline TD(lambda).
      3. Exact Online Backward-View TD(lambda): Explicit eligibility trace accumulation:
           e_t = gamma * lambda * e_{t-1} + grad(V(s_t))
           theta <- theta + alpha * delta_t * e_t
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TDConfig] = None,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        self.model = model
        self.config = config or TDConfig()
        self.gamma = self.config.gamma
        self.lambda_ = self.config.lambda_
        self.td_method = self.config.td_method

        self.optimizer = optimizer or optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Online eligibility traces for exact backward-view TD(lambda)
        self.traces: Dict[str, torch.Tensor] = {}

    def reset_traces(self) -> None:
        """Reset eligibility traces to zero at start of an episode."""
        self.traces.clear()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.traces[name] = torch.zeros_like(param.data)

    def train_step_td0(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform a single batched TD(0) update.

        Target:
          y_t = r_{t+1} if done else r_{t+1} + gamma * V(s_{t+1})
          Note: stopgrad (detach) is applied to target.
        Loss:
          L_t = 0.5 * (V(s_t) - y_t)^2
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Compute target with stopgrad (detach)
        with torch.no_grad():
            next_values = self.model(next_states)  # [B, 1]
            # Zero out bootstrap value for terminal states: (1 - dones)
            not_done = (1.0 - dones.view(-1, 1).float())
            targets = rewards.view(-1, 1).float() + not_done * self.gamma * next_values

        # 2. Compute current state value
        values = self.model(states)  # [B, 1]

        # 3. Squared TD error loss
        td_errors = targets - values
        loss = 0.5 * (td_errors.pow(2)).mean()

        loss.backward()

        if self.config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

        self.optimizer.step()

        abs_errors = td_errors.detach().abs()
        return {
            "loss": float(loss.item()),
            "mean_td_error": float(td_errors.detach().mean().item()),
            "mean_abs_td_error": float(abs_errors.mean().item()),
            "max_abs_td_error": float(abs_errors.max().item()),
            "mean_value": float(values.detach().mean().item()),
            "mean_target": float(targets.mean().item()),
        }

    @staticmethod
    def compute_lambda_returns(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 1.0,
        lambda_: float = 0.8,
    ) -> torch.Tensor:
        """Compute forward-view lambda-returns over a sequential trajectory.

        Recurrence from T-1 down to 0:
          G_t^lambda = r_{t+1} + gamma * ((1 - lambda) * V(s_{t+1}) + lambda * G_{t+1}^lambda)
          if s_{t+1} is terminal: G_t^lambda = r_{t+1}

        Args:
          rewards: [T] tensor of transition rewards
          values: [T + 1] tensor of state values V(s_0), ..., V(s_T)
          dones: [T] tensor of done flags
          gamma: discount factor
          lambda_: lambda parameter in [0, 1]

        Returns:
          [T] tensor of lambda-return targets
        """
        T = rewards.size(0)
        returns = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
        running_return = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)

        for t in reversed(range(T)):
            r = rewards[t]
            done = dones[t]
            next_v = values[t + 1]

            if done:
                running_return = r
            else:
                running_return = r + gamma * ((1.0 - lambda_) * next_v + lambda_ * running_return)

            returns[t] = running_return

        return returns

    def train_step_td_lambda_forward(
        self,
        trajectory_states: torch.Tensor,
        trajectory_rewards: torch.Tensor,
        trajectory_dones: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform trajectory update using forward-view TD(lambda) targets.

        Note on Implementation:
          This implements the Forward-View lambda-return TD(lambda) over complete trajectories.
          In RL theory (Sutton & Barto Ch. 12), the forward-view lambda-return is mathematically
          equivalent to the offline backward-view eligibility trace TD(lambda).
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Compute values for all states in trajectory
        with torch.no_grad():
            all_values = self.model(trajectory_states).squeeze(-1)  # [T]
            # Append 0 for terminal bootstrap reference
            values_with_terminal = torch.cat([all_values, torch.tensor([0.0], device=trajectory_states.device)])
            targets = self.compute_lambda_returns(
                rewards=trajectory_rewards,
                values=values_with_terminal,
                dones=trajectory_dones,
                gamma=self.gamma,
                lambda_=self.lambda_,
            ).unsqueeze(-1)

        # Forward pass on states
        current_values = self.model(trajectory_states)  # [T, 1]
        td_errors = targets - current_values
        loss = 0.5 * (td_errors.pow(2)).mean()

        loss.backward()

        if self.config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

        self.optimizer.step()

        abs_errors = td_errors.detach().abs()
        return {
            "loss": float(loss.item()),
            "mean_td_error": float(td_errors.detach().mean().item()),
            "mean_abs_td_error": float(abs_errors.mean().item()),
            "max_abs_td_error": float(abs_errors.max().item()),
            "mean_value": float(current_values.detach().mean().item()),
            "mean_target": float(targets.mean().item()),
        }

    def train_step_td_lambda_exact_online(
        self,
        state: torch.Tensor,
        reward: float,
        next_state: Optional[torch.Tensor],
        done: bool,
    ) -> Dict[str, float]:
        """Exact online backward-view TD(lambda) with eligibility traces.

        Updates parameters according to:
          delta_t = r_{t+1} + gamma * V(s_{t+1}) - V(s_t) (non-terminal)
          delta_t = r_{t+1} - V(s_t)                       (terminal)
          e_t = gamma * lambda * e_{t-1} + grad_theta(V(s_t))
          theta <- theta + alpha * delta_t * e_t
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Forward pass on s_t
        value_t = self.model(state.unsqueeze(0) if state.dim() == 1 else state)
        value_t_val = value_t.item()

        # 2. Compute TD error delta_t
        if done or next_state is None:
            target = float(reward)
        else:
            with torch.no_grad():
                next_val = self.model(next_state.unsqueeze(0) if next_state.dim() == 1 else next_state).item()
            target = float(reward) + self.gamma * next_val

        delta = target - value_t_val

        # 3. Compute gradient grad_theta(V(s_t))
        value_t.backward()

        # 4. Update traces and apply parameter update theta <- theta + alpha * delta * e
        lr = self.optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in self.traces:
                        self.traces[name] = torch.zeros_like(param.data)

                    # e_t = gamma * lambda * e_{t-1} + grad(V)
                    self.traces[name].mul_(self.gamma * self.lambda_).add_(param.grad.data)

                    # Parameter update: theta <- theta + alpha * delta * e_t
                    param.data.add_(self.traces[name], alpha=lr * delta)

        if done:
            self.reset_traces()

        return {
            "loss": 0.5 * (delta ** 2),
            "mean_td_error": delta,
            "mean_abs_td_error": abs(delta),
            "mean_value": value_t_val,
            "target": target,
        }
