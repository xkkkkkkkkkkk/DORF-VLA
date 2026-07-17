from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNG state after a training phase for reproducible continuation."""
    import random

    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import torch
    except ImportError:
        return state

    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state["numpy"] = np.random.get_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore process RNG state saved by :func:`capture_rng_state`."""
    import random

    import torch

    for key in ("python", "torch"):
        if key not in state:
            raise RuntimeError(f"RNG checkpoint is missing {key!r}.")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state, but CUDA is unavailable.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Checkpoint contains NumPy RNG state, but NumPy is unavailable.") from exc
        np.random.set_state(state["numpy"])


def assert_sac_flow_device_ready(device: str, *, cuda_available_fn: Callable[[], bool] | None = None) -> None:
    """检查 SAC-Flow 运行设备是否满足当前进程条件。"""
    if not isinstance(device, str) or not device:
        raise ValueError(f"device must be a non-empty string, got {device!r}.")
    if not device.startswith("cuda"):
        return

    if cuda_available_fn is None:
        import torch

        cuda_available_fn = torch.cuda.is_available
    if not cuda_available_fn():
        raise RuntimeError(f"CUDA device {device!r} was requested, but CUDA is not available in this process.")


def save_sac_flow_checkpoint(
    *,
    output_dir: str | Path,
    step: int,
    policy: Any | None,
    q_network: Any,
    target_q_network: Any,
    temperature: Any,
    config: Any,
    trainer: Any | None = None,
    replay_buffer: Any | None = None,
    policy_preprocessor: Any | None = None,
    policy_postprocessor: Any | None = None,
    extra_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
    torch_save_fn: Callable[[dict[str, Any], Path], None] | None = None,
) -> Path:
    """保存 policy 权重和 SAC-Flow 独有状态，返回 checkpoint 目录。"""
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}.")
    checkpoint_dir = Path(output_dir).expanduser() / f"checkpoint_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if policy is not None and callable(getattr(policy, "save_pretrained", None)):
        policy_dir = checkpoint_dir / "policy"
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(policy_dir)
        _save_processor_if_available(
            policy_preprocessor,
            policy_dir,
            config_filename="policy_preprocessor.json",
        )
        _save_processor_if_available(
            policy_postprocessor,
            policy_dir,
            config_filename="policy_postprocessor.json",
        )

    if torch_save_fn is None:
        import torch

        torch_save_fn = torch.save

    payload = {
        "step": int(step),
        "q_network": q_network.state_dict(),
        "target_q_network": target_q_network.state_dict(),
        "temperature": temperature.state_dict(),
        "config": asdict(config) if is_dataclass(config) else dict(config),
        "extra_state": dict(extra_state or {}),
        "rng_state": capture_rng_state() if rng_state is None else dict(rng_state),
    }
    if trainer is not None:
        payload["trainer"] = trainer.state_dict()
    if replay_buffer is not None:
        payload["replay_buffer"] = replay_buffer.state_dict()
    torch_save_fn(payload, checkpoint_dir / "sac_flow_state.pt")
    return checkpoint_dir


def _save_processor_if_available(processor: Any, policy_dir: Path, *, config_filename: str) -> None:
    """Keep a SAC policy checkpoint directly consumable by ``lerobot_eval.py``."""
    save_pretrained = getattr(processor, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(policy_dir, config_filename=config_filename)


def load_sac_flow_checkpoint(
    *,
    checkpoint_dir: str | Path,
    q_network: Any,
    target_q_network: Any,
    temperature: Any,
    trainer: Any | None = None,
    replay_buffer: Any | None = None,
    restore_rng: bool = False,
    map_location: str | None = "cpu",
    torch_load_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """恢复 SAC-Flow critic、target critic 和 temperature 状态。"""
    checkpoint_path = Path(checkpoint_dir).expanduser() / "sac_flow_state.pt"
    if torch_load_fn is None:
        import torch

        torch_load_fn = torch.load

    payload = torch_load_fn(checkpoint_path, map_location=map_location)
    for key in ("q_network", "target_q_network", "temperature"):
        if key not in payload:
            raise RuntimeError(f"SAC-Flow checkpoint is missing {key!r}.")

    q_network.load_state_dict(payload["q_network"])
    target_q_network.load_state_dict(payload["target_q_network"])
    temperature.load_state_dict(payload["temperature"])
    if trainer is not None:
        if "trainer" not in payload:
            raise RuntimeError("SAC-Flow checkpoint is missing 'trainer'.")
        trainer.load_state_dict(payload["trainer"])
    if replay_buffer is not None:
        if "replay_buffer" not in payload:
            raise RuntimeError("SAC-Flow checkpoint is missing 'replay_buffer'.")
        replay_buffer.load_state_dict(payload["replay_buffer"])
    if restore_rng:
        if "rng_state" not in payload:
            raise RuntimeError("SAC-Flow checkpoint is missing 'rng_state'.")
        restore_rng_state(payload["rng_state"])
    return payload
