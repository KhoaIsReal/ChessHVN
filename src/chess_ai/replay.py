"""Experience replay buffer and trajectory management for chess self-play."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional, Tuple
import torch


@dataclass
class Transition:
    """A single transition in a chess game."""
    state: torch.Tensor          # [input_dim]
    action: str                  # UCI move string
    reward: float                # Reward from White perspective
    next_state: torch.Tensor     # [input_dim]
    done: bool                   # Whether this transition ended the game
    turn: bool                   # True for White, False for Black
    fen: str                     # FEN of the state


@dataclass
class Episode:
    """A full chess game trajectory."""
    transitions: List[Transition]
    result: str                  # '1-0', '0-1', '1/2-1/2'
    winner: Optional[str]        # 'white', 'black', 'draw'

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def length(self) -> int:
        return len(self.transitions)

    def assign_terminal_rewards(self, white_reward: float) -> None:
        """Assign terminal reward from White perspective to the final transition."""
        if not self.transitions:
            return
        last_t = self.transitions[-1]
        last_t.reward = white_reward
        last_t.done = True


class ReplayBuffer:
    """Fixed-capacity ring buffer storing transitions for TD learning."""

    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position: int = 0

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def push_episode(self, episode: Episode) -> None:
        """Add all transitions from an episode."""
        for t in episode.transitions:
            self.push(t)

    def sample_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of transitions.

        Returns:
          states: [B, input_dim]
          rewards: [B, 1]
          next_states: [B, input_dim]
          dones: [B, 1]
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = torch.stack([t.state for t in batch], dim=0)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(-1)
        next_states = torch.stack([t.next_state for t in batch], dim=0)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(-1)

        if device is not None:
            states = states.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

        return states, rewards, next_states, dones

    def clear(self) -> None:
        self.buffer.clear()
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)
