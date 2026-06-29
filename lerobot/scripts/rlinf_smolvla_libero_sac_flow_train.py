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



def run_runtime_probe(cli_overrides: list[str]) -> None:
    from lerobot.rlinf_smolvla_libero.runtime_probe import build_runtime_probe

    probe = build_runtime_probe(env=os.environ, cli_overrides=cli_overrides)
    device_text = probe.sac_flow_device if probe.sac_flow_device is not None else "not-set"
    print(f"SAC-Flow runtime probe passed: libero_root={probe.libero_root} policy_path={probe.policy_path} device={device_text}")

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
