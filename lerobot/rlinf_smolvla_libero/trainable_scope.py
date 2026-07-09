from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTION_PATH_SCOPE = "action_path"

_ACTION_PATH_PREFIXES = (
    "vlm_with_expert.lm_expert.",
    "model.vlm_with_expert.lm_expert.",
    "action_in_proj.",
    "model.action_in_proj.",
    "action_out_proj.",
    "model.action_out_proj.",
    "action_time_mlp_in.",
    "model.action_time_mlp_in.",
    "action_time_mlp_out.",
    "model.action_time_mlp_out.",
)


@dataclass(frozen=True)
class TrainableParameterAudit:
    """SAC-Flow actor 参数范围审计结果。"""

    scope: str
    total_params: int
    trainable_params: int
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]

    @property
    def trainable_ratio(self) -> float:
        if self.total_params == 0:
            return 0.0
        return self.trainable_params / self.total_params


def apply_actor_trainable_scope(policy: Any, *, scope: str) -> TrainableParameterAudit:
    """应用 SmolVLA SAC-Flow actor 训练范围，并返回可审计摘要。"""
    if scope != ACTION_PATH_SCOPE:
        raise ValueError(f"actor_train_scope must be {ACTION_PATH_SCOPE!r}, got {scope!r}.")

    named_parameters = _named_parameters(policy)
    for _, parameter in named_parameters:
        _set_requires_grad(parameter, False)

    for name, parameter in named_parameters:
        if _is_action_path_parameter(name):
            _set_requires_grad(parameter, True)

    return audit_trainable_parameters(policy, scope=scope)


def audit_trainable_parameters(module: Any, *, scope: str) -> TrainableParameterAudit:
    """统计模块中 trainable / frozen 参数数量和名称。"""
    total_params = 0
    trainable_params = 0
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in _named_parameters(module):
        count = _parameter_numel(parameter)
        total_params += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable_params += count
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    return TrainableParameterAudit(
        scope=scope,
        total_params=total_params,
        trainable_params=trainable_params,
        trainable_names=tuple(trainable_names),
        frozen_names=tuple(frozen_names),
    )


def iter_trainable_parameters(module: Any) -> list[Any]:
    """返回 requires_grad=True 的参数列表，供优化器显式使用。"""
    return [parameter for _, parameter in _named_parameters(module) if bool(getattr(parameter, "requires_grad", False))]


def _is_action_path_parameter(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _ACTION_PATH_PREFIXES)


def _named_parameters(module: Any) -> list[tuple[str, Any]]:
    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        raise AttributeError("actor policy must provide callable named_parameters")
    return list(named_parameters())


def _set_requires_grad(parameter: Any, value: bool) -> None:
    setter = getattr(parameter, "requires_grad_", None)
    if callable(setter):
        setter(value)
    else:
        parameter.requires_grad = value


def _parameter_numel(parameter: Any) -> int:
    numel = getattr(parameter, "numel", None)
    if callable(numel):
        return int(numel())
    return 1
