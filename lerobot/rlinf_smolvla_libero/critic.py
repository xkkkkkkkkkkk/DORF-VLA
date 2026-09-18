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


def pairwise_q_ranking_loss(
    preferred_q: torch.Tensor,
    rejected_q: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """Require a genuinely better action to outrank a worse paired action."""
    _require_2d("preferred_q", preferred_q)
    _require_2d("rejected_q", rejected_q)
    _require_same_batch("preferred_q", preferred_q, "rejected_q", rejected_q)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 0.0:
        raise ValueError(f"margin must be a non-negative number, got {margin!r}.")
    preferred = preferred_q.mean(dim=-1)
    rejected = rejected_q.mean(dim=-1)
    return torch.relu(float(margin) - preferred + rejected).mean()


def conservative_q_penalty(
    q_data: torch.Tensor,
    q_random: torch.Tensor,
    *,
    agg: str,
    margin: float = 0.1,
) -> torch.Tensor:
    """Penalize random actions that outrank replay actions.

    This is a bounded, one-sided ranking loss. Unlike a log-sum-exp term with
    ``-q_data``, it cannot keep decreasing by pushing replay-action Q values
    upward without limit.
    """
    _require_2d("q_data", q_data)
    if q_random.ndim != 3:
        raise ValueError(
            "q_random must have shape [batch, num_random_actions, num_q_heads], "
            f"got {tuple(q_random.shape)}."
        )
    if q_random.shape[0] != q_data.shape[0] or q_random.shape[2] != q_data.shape[1]:
        raise ValueError("q_random batch/head dimensions must match q_data.")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 0.0:
        raise ValueError(f"margin must be a non-negative number, got {margin!r}.")

    random_q = aggregate_q(q_random.reshape(-1, q_random.shape[-1]), agg=agg)
    random_q = random_q.reshape(q_random.shape[0], q_random.shape[1])
    data_q = aggregate_q(q_data, agg=agg).squeeze(-1)
    violations = random_q - data_q.unsqueeze(1) + float(margin)
    return torch.relu(violations).mean()


def sample_critic_actions(
    actions: torch.Tensor,
    num_samples: int,
    *,
    strategy: str,
    noise_std: float,
) -> torch.Tensor:
    """Sample comparison actions in the replay action coordinate system.

    SmolVLA's action processor uses mean/std normalization, so a hard-coded
    ``[-1, 1]`` proposal distribution is not generally the policy's support.
    The default local Gaussian keeps comparisons around the executed action
    without imposing an artificial bound.  ``unit_uniform`` remains available
    only as an explicit compatibility/control condition.
    """
    _require_2d("actions", actions)
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"num_samples must be a positive integer, got {num_samples}.")
    if not isinstance(strategy, str) or strategy not in {"replay_local_gaussian", "unit_uniform"}:
        raise ValueError(
            "strategy must be one of {'replay_local_gaussian', 'unit_uniform'}, "
            f"got {strategy!r}."
        )
    if isinstance(noise_std, bool) or not isinstance(noise_std, (int, float)) or noise_std < 0.0:
        raise ValueError(f"noise_std must be a non-negative number, got {noise_std!r}.")

    if strategy == "unit_uniform":
        return torch.empty(
            actions.shape[0],
            num_samples,
            actions.shape[1],
            device=actions.device,
            dtype=actions.dtype,
        ).uniform_(-1.0, 1.0)

    noise = torch.randn(
        actions.shape[0],
        num_samples,
        actions.shape[1],
        device=actions.device,
        dtype=actions.dtype,
    ).mul(float(noise_std))
    return actions.detach().unsqueeze(1) + noise


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
