"""Unit tests for TD(0), terminal state non-bootstrapping, and TD(lambda)."""

import pytest
import torch
import torch.optim as optim

from chess_ai.config import TDConfig
from chess_ai.model import ChessValueNet
from chess_ai.td_learning import TDLearner


def test_td0_loss_and_learning():
    input_dim = 64
    model = ChessValueNet(input_dim=input_dim, hidden_dims=[64, 32])
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    learner = TDLearner(model=model, optimizer=optimizer)

    states = torch.randn(8, input_dim)
    rewards = torch.tensor([0.0] * 8)
    next_states = torch.randn(8, input_dim)
    dones = torch.tensor([0.0] * 8)

    initial_loss = learner.train_step_td0(states, rewards, next_states, dones)["loss"]

    # Over multiple steps on the same batch, loss should decrease
    for _ in range(15):
        final_loss = learner.train_step_td0(states, rewards, next_states, dones)["loss"]

    assert final_loss < initial_loss, f"Expected loss to decrease: {initial_loss} -> {final_loss}"


def test_td0_terminal_no_bootstrap():
    """Verify that terminal states (done=1.0) do NOT bootstrap from V(s_{t+1})."""
    input_dim = 32
    model = ChessValueNet(input_dim=input_dim, hidden_dims=[32, 16])
    learner = TDLearner(model=model)

    state = torch.randn(1, input_dim)
    reward = torch.tensor([1.0])  # Win reward
    next_state = torch.randn(1, input_dim)
    done = torch.tensor([1.0])     # Game over

    with torch.no_grad():
        next_val = model(next_state)
        # Even if next_val is non-zero, done=1 cancels it
        expected_target = reward.item()  # 1.0

    # Call train_step_td0
    metrics = learner.train_step_td0(state, reward, next_state, done)
    assert abs(metrics["mean_target"] - expected_target) < 1e-5


def test_td_lambda_returns():
    """Verify forward-view lambda returns for lambda=0 (TD(0)) and lambda=1 (Monte Carlo)."""
    # 3 transitions: r1=0, r2=0, r3=1.0 (win), done at step 3
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.1, 0.2, 0.5, 0.0])  # V(s0), V(s1), V(s2), V(s3=terminal)
    dones = torch.tensor([0.0, 0.0, 1.0])
    gamma = 1.0

    # 1. Lambda = 0: equivalent to TD(0) 1-step targets:
    # G_0 = r_1 + gamma * V(s_1) = 0.0 + 0.2 = 0.2
    # G_1 = r_2 + gamma * V(s_2) = 0.0 + 0.5 = 0.5
    # G_2 = r_3 = 1.0 (terminal)
    returns_lambda0 = TDLearner.compute_lambda_returns(rewards, values, dones, gamma=gamma, lambda_=0.0)
    assert torch.allclose(returns_lambda0, torch.tensor([0.2, 0.5, 1.0]), atol=1e-5)

    # 2. Lambda = 1: equivalent to Monte Carlo returns:
    # G_0 = 1.0
    # G_1 = 1.0
    # G_2 = 1.0
    returns_lambda1 = TDLearner.compute_lambda_returns(rewards, values, dones, gamma=gamma, lambda_=1.0)
    assert torch.allclose(returns_lambda1, torch.tensor([1.0, 1.0, 1.0]), atol=1e-5)


def test_td_lambda_exact_online_update():
    """Verify exact online eligibility trace update."""
    input_dim = 32
    model = ChessValueNet(input_dim=input_dim, hidden_dims=[32, 16])
    cfg = TDConfig(td_method="td_lambda", learning_rate=0.01)
    learner = TDLearner(model=model, config=cfg)

    state = torch.randn(input_dim)
    next_state = torch.randn(input_dim)

    learner.reset_traces()
    res1 = learner.train_step_td_lambda_exact_online(state, reward=0.0, next_state=next_state, done=False)
    assert "loss" in res1
    assert "mean_td_error" in res1
    assert len(learner.traces) > 0, "Traces should be populated after online step"
