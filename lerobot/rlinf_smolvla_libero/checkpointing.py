from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable


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
    extra_state: dict[str, Any] | None = None,
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
    }
    torch_save_fn(payload, checkpoint_dir / "sac_flow_state.pt")
    return checkpoint_dir


def load_sac_flow_checkpoint(
    *,
    checkpoint_dir: str | Path,
    q_network: Any,
    target_q_network: Any,
    temperature: Any,
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
    return payload
