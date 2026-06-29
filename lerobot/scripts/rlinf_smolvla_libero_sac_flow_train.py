"""SmolVLA LIBERO SAC-Flow training entry.

Real training requires LEROBOT_LIBERO_ROOT and the same SmolVLA checkpoint/config overrides
used for baseline SmolVLA evaluation. The current --dry-run path only checks entry guards,
imports, and SAC-Flow defaults; it does not create LIBERO environments, load SmolVLA weights,
start training, or use GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_LIBERO_ROOT_REQUIRED = "LEROBOT_LIBERO_ROOT is required for LIBERO SAC-Flow smoke runs."
_REAL_TRAINING_NOT_READY = "Real SmolVLA/LIBERO SAC-Flow training is not wired yet; a later task will connect the real SmolVLA policy and LIBERO environment."


def require_libero_root() -> str:
    """检查 LIBERO 根目录变量；入口阶段失败，避免误入真实训练流程。"""
    libero_root = os.environ.get("LEROBOT_LIBERO_ROOT")
    if not libero_root:
        raise RuntimeError(_LIBERO_ROOT_REQUIRED)
    return libero_root


def _validate_dry_run_imports() -> None:
    """确认入口依赖解析到本任务的轻量 SAC-Flow 接口。"""
    from lerobot.rlinf_smolvla_libero.actor_adapter import SmolVLASACFlowActor
    from lerobot.rlinf_smolvla_libero.config import SACFlowConfig
    from lerobot.rlinf_smolvla_libero.libero_adapter import execute_action_chunk

    required_objects = (
        SACFlowConfig,
        SmolVLASACFlowActor,
        execute_action_chunk,
    )
    missing_names = [getattr(obj, "__name__", "") for obj in required_objects if obj is None]
    if missing_names:
        raise RuntimeError(f"SAC-Flow dry-run imports are incomplete: {missing_names}.")


def run_dry_run() -> None:
    require_libero_root()
    _validate_dry_run_imports()

    from lerobot.rlinf_smolvla_libero.config import SACFlowConfig

    config = SACFlowConfig()
    # 验证几个后续训练最依赖的默认值，避免入口 smoke 只测到 argparse 而漏掉配置漂移。
    expected_defaults = {
        "gamma": 0.96,
        "critic_actor_ratio": 4,
        "num_updates_per_step": 64,
    }
    mismatches = {
        name: getattr(config, name)
        for name, expected in expected_defaults.items()
        if getattr(config, name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Unexpected SACFlowConfig defaults: {mismatches}.")

    print("SAC-Flow dry-run passed")




def _normalize_split_override_forms(cli_overrides: list[str]) -> list[str]:
    """把 `--key value` 形式规范成 LeRobot parser 支持的 `--key=value`。"""
    normalized: list[str] = []
    index = 0
    while index < len(cli_overrides):
        item = cli_overrides[index]
        if item.startswith("--") and "=" not in item and index + 1 < len(cli_overrides):
            next_item = cli_overrides[index + 1]
            if not next_item.startswith("--"):
                normalized.append(f"{item}={next_item}")
                index += 2
                continue
        normalized.append(item)
        index += 1
    return normalized


def _split_lerobot_and_sac_flow_overrides(cli_overrides: list[str]) -> tuple[list[str], list[str]]:
    """拆分 LeRobot 配置参数和 SAC-Flow 专属参数，避免 draccus 看到未知字段。"""
    lerobot_overrides: list[str] = []
    sac_flow_overrides: list[str] = []
    for item in _normalize_split_override_forms(cli_overrides):
        if item.startswith("--sac-flow."):
            sac_flow_overrides.append(item)
        else:
            lerobot_overrides.append(item)
    return lerobot_overrides, sac_flow_overrides


def parse_train_config_from_overrides(cli_overrides: list[str]):
    """只解析 TrainPipelineConfig；不 validate，不创建 dataset/policy/env。"""
    import draccus

    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig

    lerobot_overrides, _ = _split_lerobot_and_sac_flow_overrides(cli_overrides)
    if hasattr(TrainPipelineConfig, "__get_path_fields__"):
        lerobot_overrides = parser.filter_path_args(TrainPipelineConfig.__get_path_fields__(), lerobot_overrides)
    return draccus.parse(config_class=TrainPipelineConfig, args=lerobot_overrides)

def run_runtime_probe(cli_overrides: list[str]) -> None:
    from lerobot.rlinf_smolvla_libero.runtime_probe import build_runtime_probe

    # 先做路径级探测，保证缺 LEROBOT_LIBERO_ROOT / --policy.path 时错误最早、最清楚。
    build_runtime_probe(env=os.environ, cli_overrides=cli_overrides)

    train_cfg = None
    try:
        train_cfg = parse_train_config_from_overrides(cli_overrides)
    except ModuleNotFoundError as exc:
        if exc.name != "draccus":
            raise
        # 本地 Codex Python 可能缺少 LeRobot parser 依赖；服务器环境会执行真实解析。
        train_cfg = None
    except Exception as exc:
        if "Missing required field(s) `dataset`" not in str(exc):
            raise
        # 兼容只做路径/设备 smoke 的调用；完整 TrainPipelineConfig probe 需要 --dataset.repo_id。
        train_cfg = None

    probe = build_runtime_probe(env=os.environ, cli_overrides=cli_overrides, train_cfg=train_cfg)
    device_text = probe.sac_flow_device if probe.sac_flow_device is not None else "not-set"
    extra = ""
    if probe.train_steps is not None:
        extra = f" train_steps={probe.train_steps} batch_size={probe.batch_size}"
    print(
        f"SAC-Flow runtime probe passed: libero_root={probe.libero_root} "
        f"policy_path={probe.policy_path} device={device_text}{extra}"
    )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe SAC-Flow smoke entry for SmolVLA LIBERO.")
    parser.add_argument("--dry-run", action="store_true", help="Check imports/config/env only; do not load models, envs, train, or use GPU.")
    parser.add_argument("--probe-runtime", action="store_true", help="Check LIBERO root and baseline SmolVLA overrides without loading models/envs.")
    args, cli_overrides = parser.parse_known_args(argv)

    if args.dry_run:
        run_dry_run()
        return 0

    if args.probe_runtime:
        run_runtime_probe(cli_overrides)
        return 0

    require_libero_root()
    raise NotImplementedError(_REAL_TRAINING_NOT_READY)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
