"""SmolVLA LIBERO SAC-Flow training entry.

Real training requires LEROBOT_LIBERO_ROOT and the same SmolVLA checkpoint/config overrides
used for baseline SmolVLA evaluation. The current --dry-run path only checks entry guards,
imports, and SAC-Flow defaults; it does not create LIBERO environments, load SmolVLA weights,
start training, or use GPU.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_LIBERO_ROOT_REQUIRED = "LEROBOT_LIBERO_ROOT is required for LIBERO SAC-Flow smoke runs."
_CHOOSE_EXPLICIT_MODE = "Choose --dry-run, --probe-runtime, --preflight-device, --gpu-smoke, or --train-run."


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


def _should_parse_train_config(cli_overrides: list[str]) -> bool:
    """只有提供完整训练配置线索时才解析 TrainPipelineConfig。"""
    normalized = _normalize_split_override_forms(cli_overrides)
    return any(item.startswith("--dataset.") or item.startswith("--config_path=") for item in normalized)


@contextmanager
def _temporary_cli_overrides(cli_overrides: list[str]):
    """临时同步 sys.argv，避免 LeRobot validate 看不到传入的覆盖参数。"""
    original_argv = list(sys.argv)
    sys.argv = [original_argv[0], *_normalize_split_override_forms(cli_overrides)]
    try:
        yield
    finally:
        sys.argv = original_argv


def require_train_config_hint(cli_overrides: list[str]) -> None:
    """GPU smoke 必须提供真实 TrainPipelineConfig 来源，避免进入模糊 parser 错误。"""
    if not _should_parse_train_config(cli_overrides):
        raise RuntimeError(
            "SAC-Flow GPU smoke requires --config_path or --dataset.repo_id from the baseline SmolVLA run."
        )


def _register_builtin_policy_config_choices() -> None:
    """导入内置 policy factory，确保 smolvla 等配置类已注册给 draccus。"""
    import lerobot.policies.factory  # noqa: F401


def parse_train_config_from_overrides(cli_overrides: list[str]):
    """只解析 TrainPipelineConfig；不 validate，不创建 dataset/policy/env。"""
    _register_builtin_policy_config_choices()

    import draccus

    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig

    lerobot_overrides, _ = _split_lerobot_and_sac_flow_overrides(cli_overrides)
    if hasattr(TrainPipelineConfig, "__get_path_fields__"):
        lerobot_overrides = parser.filter_path_args(TrainPipelineConfig.__get_path_fields__(), lerobot_overrides)
    with _temporary_cli_overrides(lerobot_overrides):
        return draccus.parse(config_class=TrainPipelineConfig, args=lerobot_overrides)


def run_runtime_probe(cli_overrides: list[str]) -> None:
    from lerobot.rlinf_smolvla_libero.runtime_probe import build_runtime_probe

    # 先做路径级探测，保证缺 LEROBOT_LIBERO_ROOT / --policy.path 时错误最早、最清楚。
    build_runtime_probe(env=os.environ, cli_overrides=cli_overrides)

    train_cfg = None
    if _should_parse_train_config(cli_overrides):
        try:
            train_cfg = parse_train_config_from_overrides(cli_overrides)
        except ModuleNotFoundError as exc:
            if exc.name != "draccus":
                raise
            # 本地 Codex Python 可能缺少 LeRobot parser 依赖；服务器环境会执行真实解析。
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

def run_device_preflight(cli_overrides: list[str]) -> None:
    from lerobot.rlinf_smolvla_libero.checkpointing import assert_sac_flow_device_ready
    from lerobot.rlinf_smolvla_libero.runtime_probe import _extract_override_value

    device = _extract_override_value(cli_overrides, "sac-flow.device") or "cpu"
    assert_sac_flow_device_ready(device)
    print(f"SAC-Flow device preflight passed: device={device}")


def _extract_int_override(cli_overrides: list[str], key: str, default: int) -> int:
    from lerobot.rlinf_smolvla_libero.runtime_probe import _extract_override_value

    value = _extract_override_value(cli_overrides, key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"--{key} must be an integer, got {value!r}.") from exc


def _extract_float_override(cli_overrides: list[str], key: str, default: float) -> float:
    from lerobot.rlinf_smolvla_libero.runtime_probe import _extract_override_value

    value = _extract_override_value(cli_overrides, key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"--{key} must be a number, got {value!r}.") from exc


def _extract_str_override(cli_overrides: list[str], key: str, default: str | None = None) -> str | None:
    from lerobot.rlinf_smolvla_libero.runtime_probe import _extract_override_value

    value = _extract_override_value(cli_overrides, key)
    return default if value is None else value


def _extract_bool_override(cli_overrides: list[str], key: str, default: bool) -> bool:
    from lerobot.rlinf_smolvla_libero.runtime_probe import _extract_override_value

    value = _extract_override_value(cli_overrides, key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"--{key} must be a boolean, got {value!r}.")


def _extract_tags_override(cli_overrides: list[str], key: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = _extract_str_override(cli_overrides, key)
    if value is None:
        return default
    return tuple(tag.strip() for tag in value.split(",") if tag.strip())


def run_gpu_smoke(
    cli_overrides: list[str],
    *,
    confirm_gpu_smoke: bool,
    run_fn=None,
    logger_cls=None,
    parse_train_config_fn=None,
) -> None:
    from lerobot.rlinf_smolvla_libero.config import SACFlowConfig
    from lerobot.rlinf_smolvla_libero.runtime_probe import _extract_override_value, build_runtime_probe
    from lerobot.rlinf_smolvla_libero.smoke_runner import SACFlowSmokeConfig, run_sac_flow_gpu_smoke
    from lerobot.rlinf_smolvla_libero.wandb_logger import SACFlowWandBLogger

    device = _extract_override_value(cli_overrides, "sac-flow.device") or "cpu"
    smoke_cfg = SACFlowSmokeConfig(
        device=device,
        confirm_gpu_smoke=confirm_gpu_smoke,
        max_train_steps=_extract_int_override(cli_overrides, "sac-flow.max-train-steps", 2),
        max_chunk_steps=_extract_int_override(cli_overrides, "sac-flow.max-chunk-steps", 1),
        num_updates_per_step=_extract_int_override(cli_overrides, "sac-flow.num-updates-per-step", 1),
        batch_size=_extract_int_override(cli_overrides, "sac-flow.batch-size", 1),
        min_buffer_size=_extract_int_override(cli_overrides, "sac-flow.min-buffer-size", 1),
    )

    build_runtime_probe(env=os.environ, cli_overrides=cli_overrides)
    require_train_config_hint(cli_overrides)
    lerobot_overrides, _ = _split_lerobot_and_sac_flow_overrides(cli_overrides)
    sac_config = SACFlowConfig(
        device=device,
        wandb_enable=_extract_bool_override(cli_overrides, "sac-flow.wandb-enable", False),
        wandb_project=_extract_str_override(cli_overrides, "sac-flow.wandb-project", os.environ.get("WANDB_PROJECT")),
        wandb_run_name=_extract_str_override(cli_overrides, "sac-flow.wandb-run-name", os.environ.get("WANDB_RUN_NAME")),
        wandb_mode=_extract_str_override(cli_overrides, "sac-flow.wandb-mode", os.environ.get("WANDB_MODE", "online"))
        or "online",
        wandb_tags=_extract_tags_override(cli_overrides, "sac-flow.wandb-tags", ("smoke", "action_path")),
    )
    parse_fn = parse_train_config_fn or parse_train_config_from_overrides
    run = run_fn or run_sac_flow_gpu_smoke
    logger_factory = logger_cls or SACFlowWandBLogger
    with _temporary_cli_overrides(lerobot_overrides):
        train_cfg = parse_fn(cli_overrides)
        logger = logger_factory(sac_config)
        logger.start(
            {
                "run_type": "gpu_smoke",
                "actor_train_scope": sac_config.actor_train_scope,
                "max_train_steps": smoke_cfg.max_train_steps,
                "num_updates_per_step": smoke_cfg.num_updates_per_step,
                "batch_size": smoke_cfg.batch_size,
            }
        )
        try:
            result = run(
                train_cfg=train_cfg,
                smoke_cfg=smoke_cfg,
                sac_config=sac_config,
                logger=logger,
            )
        finally:
            logger.finish()
    checkpoint_text = result.checkpoint_dir if result.checkpoint_dir is not None else "not-saved"
    print(f"SAC-Flow GPU smoke passed: steps={result.steps} checkpoint={checkpoint_text}")


def run_train_run(
    cli_overrides: list[str],
    *,
    run_fn=None,
    logger_cls=None,
    parse_train_config_fn=None,
) -> None:
    from lerobot.rlinf_smolvla_libero.config import SACFlowConfig
    from lerobot.rlinf_smolvla_libero.runtime_probe import _extract_override_value, build_runtime_probe
    from lerobot.rlinf_smolvla_libero.smoke_runner import SACFlowRunConfig, run_sac_flow_training_run
    from lerobot.rlinf_smolvla_libero.wandb_logger import SACFlowWandBLogger

    device = _extract_override_value(cli_overrides, "sac-flow.device") or "cpu"
    run_cfg = SACFlowRunConfig(
        device=device,
        max_train_steps=_extract_int_override(cli_overrides, "sac-flow.max-train-steps", 100),
        max_chunk_steps=_extract_int_override(cli_overrides, "sac-flow.max-chunk-steps", 1),
        num_updates_per_step=_extract_int_override(cli_overrides, "sac-flow.num-updates-per-step", 4),
        batch_size=_extract_int_override(cli_overrides, "sac-flow.batch-size", 2),
        min_buffer_size=_extract_int_override(cli_overrides, "sac-flow.min-buffer-size", 2),
        replay_capacity=_extract_int_override(cli_overrides, "sac-flow.replay-capacity", 64),
    )
    sac_config = SACFlowConfig(
        device=device,
        actor_train_scope=_extract_str_override(cli_overrides, "sac-flow.actor-train-scope", "action_path"),
        actor_lr=_extract_float_override(cli_overrides, "sac-flow.actor-lr", 1e-5),
        critic_lr=_extract_float_override(cli_overrides, "sac-flow.critic-lr", 3e-4),
        alpha_lr=_extract_float_override(cli_overrides, "sac-flow.alpha-lr", 3e-4),
        wandb_enable=_extract_bool_override(cli_overrides, "sac-flow.wandb-enable", True),
        wandb_project=_extract_str_override(
            cli_overrides,
            "sac-flow.wandb-project",
            os.environ.get("WANDB_PROJECT"),
        ),
        wandb_run_name=_extract_str_override(
            cli_overrides,
            "sac-flow.wandb-run-name",
            os.environ.get("WANDB_RUN_NAME"),
        ),
        wandb_mode=_extract_str_override(
            cli_overrides,
            "sac-flow.wandb-mode",
            os.environ.get("WANDB_MODE", "online"),
        )
        or "online",
        wandb_tags=_extract_tags_override(cli_overrides, "sac-flow.wandb-tags", ("action_path",)),
    )

    build_runtime_probe(env=os.environ, cli_overrides=cli_overrides)
    require_train_config_hint(cli_overrides)
    parse_fn = parse_train_config_fn or parse_train_config_from_overrides
    run = run_fn or run_sac_flow_training_run
    logger_factory = logger_cls or SACFlowWandBLogger
    lerobot_overrides, _ = _split_lerobot_and_sac_flow_overrides(cli_overrides)
    with _temporary_cli_overrides(lerobot_overrides):
        train_cfg = parse_fn(cli_overrides)
        logger = logger_factory(sac_config)
        logger.start(
            {
                "actor_train_scope": sac_config.actor_train_scope,
                "max_train_steps": run_cfg.max_train_steps,
                "num_updates_per_step": run_cfg.num_updates_per_step,
                "batch_size": run_cfg.batch_size,
            }
        )
        try:
            result = run(
                train_cfg=train_cfg,
                run_cfg=run_cfg,
                sac_config=sac_config,
                logger=logger,
            )
        finally:
            logger.finish()
    checkpoint_text = result.checkpoint_dir if result.checkpoint_dir is not None else "not-saved"
    print(f"SAC-Flow training run finished: steps={result.steps} checkpoint={checkpoint_text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe SAC-Flow smoke entry for SmolVLA LIBERO.")
    parser.add_argument("--dry-run", action="store_true", help="Check imports/config/env only; do not load models, envs, train, or use GPU.")
    parser.add_argument("--probe-runtime", action="store_true", help="Check LIBERO root and baseline SmolVLA overrides without loading models/envs.")
    parser.add_argument("--preflight-device", action="store_true", help="Check requested SAC-Flow device availability without loading models/envs.")
    parser.add_argument("--gpu-smoke", action="store_true", help="Run the hard-budget real SmolVLA/LIBERO SAC-Flow smoke.")
    parser.add_argument("--confirm-gpu-smoke", action="store_true", help="Required for --gpu-smoke.")
    parser.add_argument("--train-run", action="store_true", help="Run a controlled SmolVLA/LIBERO SAC-Flow training run.")
    args, cli_overrides = parser.parse_known_args(argv)

    if args.dry_run:
        run_dry_run()
        return 0

    if args.probe_runtime:
        run_runtime_probe(cli_overrides)
        return 0

    if args.preflight_device:
        run_device_preflight(cli_overrides)
        return 0

    if args.gpu_smoke:
        run_gpu_smoke(cli_overrides, confirm_gpu_smoke=args.confirm_gpu_smoke)
        return 0

    if args.train_run:
        run_train_run(cli_overrides)
        return 0

    require_libero_root()
    raise RuntimeError(_CHOOSE_EXPLICIT_MODE)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
