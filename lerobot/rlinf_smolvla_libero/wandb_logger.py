from __future__ import annotations

from typing import Any, Callable, Mapping

from .config import SACFlowConfig


_SAC_METRIC_NAMES = {
    "critic_loss": "train/sac/critic_loss",
    "actor_loss": "train/sac/actor_loss",
    "alpha_loss": "train/sac/alpha_loss",
    "alpha": "train/sac/alpha",
    "entropy": "train/actor/entropy",
    "log_pi": "train/actor/log_pi",
    "kl_estimate": "train/actor/kl_estimate",
    "kl_penalty": "train/actor/kl_penalty",
    "q_mean": "train/critic/q_mean",
    "q_min": "train/critic/q_min",
    "q_max": "train/critic/q_max",
    "q_std": "train/critic/q_std",
    "q_head_span": "train/critic/q_head_span",
    "target_q_mean": "train/critic/target_q_mean",
    "target_q_std": "train/critic/target_q_std",
    "batch_positive_reward_fraction": "train/batch/positive_reward_fraction",
    "critic_grad_norm": "train/critic/grad_norm",
    "actor_grad_norm": "train/actor/grad_norm",
    "critic_update_count": "train/critic/update_count",
    "actor_update_count": "train/actor/update_count",
    "alpha_update_count": "train/alpha/update_count",
}


class SACFlowWandBLogger:
    """SAC-Flow 的可选 WandB 日志封装。"""

    def __init__(
        self,
        config: SACFlowConfig,
        *,
        import_wandb_fn: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self._import_wandb_fn = import_wandb_fn or _import_wandb
        self._wandb: Any | None = None
        self._started = False

    @property
    def enabled(self) -> bool:
        return bool(self.config.wandb_enable)

    def start(self, run_config: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self._wandb = self._import_wandb_fn()
        self._wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            mode=self.config.wandb_mode,
            tags=list(self.config.wandb_tags),
            config=dict(run_config),
        )
        self._started = True

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        if not self.enabled:
            return
        if self._wandb is None:
            raise RuntimeError("WandB logger must be started before logging metrics.")
        self._wandb.log(dict(metrics), step=step)

    def finish(self) -> None:
        if not self.enabled or self._wandb is None or not self._started:
            return
        self._wandb.finish()
        self._started = False


def format_sac_update_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """把 trainer 内部 metric key 转成 RLinf 风格 WandB key。"""
    formatted: dict[str, Any] = {}
    for key, value in metrics.items():
        output_key = _SAC_METRIC_NAMES.get(key)
        if output_key is not None:
            formatted[output_key] = value
    return formatted


def _import_wandb() -> Any:
    import wandb

    return wandb
