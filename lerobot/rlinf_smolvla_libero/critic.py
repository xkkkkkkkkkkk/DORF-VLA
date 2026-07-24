from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQHead(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_q_heads: int):
        super().__init__()
        _require_positive_integer("obs_dim", obs_dim)
        _require_positive_integer("action_dim", action_dim)
        _require_positive_integer("hidden_dim", hidden_dim)
        _require_positive_integer("num_q_heads", num_q_heads)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim + action_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_q_heads)
            ]
        )

    def forward(self, obs_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        _require_2d("obs_features", obs_features)
        _require_2d("actions", actions)
        if obs_features.shape[1] != self.obs_dim:
            raise ValueError(
                "obs_features must have shape [batch, obs_dim], "
                f"got shape {tuple(obs_features.shape)} with obs_dim={self.obs_dim}."
            )
        if actions.shape[1] != self.action_dim:
            raise ValueError(
                "actions must have shape [batch, action_dim], "
                f"got shape {tuple(actions.shape)} with action_dim={self.action_dim}."
            )
        _require_same_batch("obs_features", obs_features, "actions", actions)

        x = torch.cat([obs_features, actions], dim=-1)
        return torch.cat([head(x) for head in self.heads], dim=-1)


class EntropyTemperature(nn.Module):
    def __init__(self, initial_alpha: float):
        super().__init__()
        if initial_alpha <= 0:
            raise ValueError(f"initial_alpha must be positive, got {initial_alpha}.")
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(initial_alpha), dtype=torch.float32)))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()


def aggregate_q(q_values: torch.Tensor, agg: str) -> torch.Tensor:
    _require_2d("q_values", q_values)
    if agg == "min":
        return q_values.min(dim=-1, keepdim=True).values
    if agg == "mean":
        return q_values.mean(dim=-1, keepdim=True)
    raise ValueError(f"Unsupported Q aggregation: {agg}")


def critic_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    discount: torch.Tensor,
    next_q: torch.Tensor,
    next_log_pi: torch.Tensor,
    alpha: torch.Tensor,
    agg: str,
    backup_entropy: bool,
) -> torch.Tensor:
    _require_scalar("alpha", alpha)
    if done.dtype != torch.bool:
        raise ValueError(f"done must be a bool tensor, got dtype {done.dtype}.")
    _require_column("reward", reward)
    _require_column("done", done)
    _require_column("discount", discount)
    _require_2d("next_q", next_q)
    _require_column("next_log_pi", next_log_pi)
    _require_same_batch("reward", reward, "done", done)
    _require_same_batch("reward", reward, "discount", discount)
    _require_same_batch("reward", reward, "next_q", next_q)
    _require_same_batch("reward", reward, "next_log_pi", next_log_pi)

    next_value = aggregate_q(next_q, agg=agg)
    if backup_entropy:
        next_value = next_value - alpha * next_log_pi
    not_done = (~done).to(dtype=reward.dtype)
    return reward + not_done * discount * next_value


def critic_loss(q_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _require_2d("q_data", q_data)
    _require_column("target", target)
    _require_same_batch("q_data", q_data, "target", target)
    return F.mse_loss(q_data, target.expand_as(q_data))


def actor_loss(
    q_pi: torch.Tensor,
    log_pi: torch.Tensor,
    alpha: torch.Tensor,
    agg: str,
    kl_estimate: torch.Tensor | None = None,
    kl_penalty_coef: float = 0.0,
    entropy_regularization: bool = False,
) -> torch.Tensor:
    _require_scalar("alpha", alpha)
    _require_2d("q_pi", q_pi)
    _require_column("log_pi", log_pi)
    _require_same_batch("q_pi", q_pi, "log_pi", log_pi)
    objective = -aggregate_q(q_pi, agg=agg)
    if entropy_regularization:
        objective = objective + alpha * log_pi
    if kl_estimate is not None:
        _require_column("kl_estimate", kl_estimate)
        _require_same_batch("q_pi", q_pi, "kl_estimate", kl_estimate)
        if kl_penalty_coef < 0:
            raise ValueError(f"kl_penalty_coef must be non-negative, got {kl_penalty_coef}.")
        objective = objective + kl_penalty_coef * kl_estimate
    return objective.mean()


def alpha_loss(log_alpha: torch.Tensor, log_pi: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """Standard SAC temperature objective, optimized in log-alpha space."""
    _require_scalar("log_alpha", log_alpha)
    _require_column("log_pi", log_pi)
    return -(log_alpha * (log_pi.detach() + target_entropy)).mean()


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")


def _require_2d(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape {tuple(tensor.shape)}.")


def _require_column(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape [batch, 1], got shape {tuple(tensor.shape)}.")


def _require_scalar(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(tensor.shape)}.")


def _require_same_batch(reference_name: str, reference: torch.Tensor, other_name: str, other: torch.Tensor) -> None:
    if reference.shape[0] != other.shape[0]:
        raise ValueError(
            f"{reference_name} and {other_name} must have the same batch size, "
            f"got {reference.shape[0]} and {other.shape[0]}."
        )


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}.")

    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
