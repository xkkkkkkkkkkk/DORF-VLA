from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SACFlowConfig:
    gamma: float = 0.96
    tau: float = 0.005
    initial_alpha: float = 0.01
    target_entropy: float | None = None
    critic_actor_ratio: int = 4
    actor_warmup_updates: int = 2000
    num_updates_per_step: int = 64
    replay_capacity: int = 200
    min_buffer_size: int = 2
    batch_size: int = 8
    num_q_heads: int = 10
    hidden_dim: int = 256
    noise_std_train: float = 0.3
    noise_std_rollout: float = 0.02
    backup_entropy: bool = True
    agg_q: str = "min"
    actor_agg_q: str = "mean"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    actor_train_scope: str = "action_path"
    wandb_enable: bool = True
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # 数值超参在构造时尽早失败，避免训练更新中出现难追踪的 NaN 或空采样。
        for name in (
            "gamma",
            "initial_alpha",
            "noise_std_train",
            "noise_std_rollout",
            "actor_lr",
            "critic_lr",
            "alpha_lr",
            "grad_clip_norm",
        ):
            _require_positive_number(name, getattr(self, name))

        if not isinstance(self.tau, (int, float)) or isinstance(self.tau, bool) or not 0.0 <= float(self.tau) <= 1.0:
            raise ValueError(f"tau must be a number in [0, 1], got {self.tau}.")

        for name in ("critic_actor_ratio", "actor_warmup_updates", "num_updates_per_step", "batch_size"):
            _require_minimum_integer(name, getattr(self, name), minimum=1)

        for name in ("replay_capacity", "min_buffer_size", "num_q_heads", "hidden_dim"):
            _require_minimum_integer(name, getattr(self, name), minimum=1)

        if self.agg_q not in {"min", "mean"}:
            raise ValueError(f"agg_q must be one of {{'min', 'mean'}}, got {self.agg_q!r}.")
        if self.actor_agg_q not in {"min", "mean"}:
            raise ValueError(f"actor_agg_q must be one of {{'min', 'mean'}}, got {self.actor_agg_q!r}.")

        if self.target_entropy is not None:
            _require_number("target_entropy", self.target_entropy)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device must be a non-empty string, got {self.device!r}.")
        if self.actor_train_scope != "action_path":
            raise ValueError(f"actor_train_scope must be 'action_path', got {self.actor_train_scope!r}.")
        if not isinstance(self.wandb_enable, bool):
            raise ValueError(f"wandb_enable must be a bool, got {self.wandb_enable!r}.")
        if self.wandb_project is not None and not isinstance(self.wandb_project, str):
            raise ValueError(f"wandb_project must be a string or None, got {self.wandb_project!r}.")
        if self.wandb_run_name is not None and not isinstance(self.wandb_run_name, str):
            raise ValueError(f"wandb_run_name must be a string or None, got {self.wandb_run_name!r}.")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"wandb_mode must be one of online/offline/disabled, got {self.wandb_mode!r}.")
        if isinstance(self.wandb_tags, list):
            self.wandb_tags = tuple(self.wandb_tags)
        if not isinstance(self.wandb_tags, tuple) or not all(isinstance(tag, str) for tag in self.wandb_tags):
            raise ValueError(f"wandb_tags must be a tuple of strings, got {self.wandb_tags!r}.")


def _require_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}.")


def _require_positive_number(name: str, value: float) -> None:
    _require_number(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _require_minimum_integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value}.")
