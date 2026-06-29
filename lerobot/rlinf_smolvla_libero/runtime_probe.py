from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_LIBERO_ROOT_REQUIRED = "LEROBOT_LIBERO_ROOT is required for LIBERO SAC-Flow smoke runs."
_POLICY_PATH_REQUIRED = "--policy.path is required to reuse the baseline SmolVLA checkpoint/config overrides."


@dataclass(frozen=True)
class SACFlowRuntimeProbe:
    """真实训练前的轻量运行条件快照；不加载模型，不创建环境。"""

    libero_root: Path
    policy_path: Path
    cli_overrides: tuple[str, ...]
    sac_flow_device: str | None = None
    train_steps: int | None = None
    batch_size: int | None = None
    dataset_repo_id: str | None = None
    env_type: str | None = None


def require_existing_path(path: str | Path, *, label: str) -> Path:
    """检查路径存在；用于入口阶段尽早暴露配置错误。"""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise RuntimeError(f"{label} path does not exist: {resolved}")
    return resolved


def _extract_override_value(cli_overrides: Sequence[str], key: str) -> str | None:
    prefix = f"--{key}="
    split_form = f"--{key}"
    for index, item in enumerate(cli_overrides):
        if item.startswith(prefix):
            value = item[len(prefix) :]
            return value or None
        if item == split_form and index + 1 < len(cli_overrides):
            value = cli_overrides[index + 1]
            return value or None
    return None


def build_runtime_probe(
    *,
    env: Mapping[str, str],
    cli_overrides: Sequence[str],
    train_cfg: Any | None = None,
) -> SACFlowRuntimeProbe:
    """收集 SAC-Flow 真实运行所需的最小路径和训练配置快照。

    这个函数故意只处理字符串、文件系统路径和已解析配置，不 import LIBERO、robosuite 或 SmolVLA，
    因此可以在无 GPU/无显示的服务器 smoke 阶段运行。
    """
    libero_root_value = env.get("LEROBOT_LIBERO_ROOT")
    if not libero_root_value:
        raise RuntimeError(_LIBERO_ROOT_REQUIRED)
    libero_root = require_existing_path(libero_root_value, label="LEROBOT_LIBERO_ROOT")

    policy_path_value = _extract_override_value(cli_overrides, "policy.path")
    if not policy_path_value:
        raise RuntimeError(_POLICY_PATH_REQUIRED)
    policy_path = require_existing_path(policy_path_value, label="policy checkpoint")

    sac_flow_device = _extract_override_value(cli_overrides, "sac-flow.device")
    dataset = getattr(train_cfg, "dataset", None)
    env_cfg = getattr(train_cfg, "env", None)
    return SACFlowRuntimeProbe(
        libero_root=libero_root,
        policy_path=policy_path,
        cli_overrides=tuple(cli_overrides),
        sac_flow_device=sac_flow_device,
        train_steps=getattr(train_cfg, "steps", None),
        batch_size=getattr(train_cfg, "batch_size", None),
        dataset_repo_id=getattr(dataset, "repo_id", None),
        env_type=getattr(env_cfg, "type", None),
    )
