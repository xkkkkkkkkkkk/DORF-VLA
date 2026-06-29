from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SmolVLAPolicyRuntime:
    """SAC-Flow 真实训练前需要的 SmolVLA 运行组件。"""

    train_cfg: Any
    dataset: Any
    dataset_meta: Any
    policy: Any
    preprocessor: Any
    postprocessor: Any


def build_smolvla_policy_runtime(
    train_cfg: Any,
    *,
    validate_config: bool = True,
    make_dataset_fn: Callable[[Any], Any] | None = None,
    make_policy_fn: Callable[..., Any] | None = None,
    make_processors_fn: Callable[..., tuple[Any, Any]] | None = None,
) -> SmolVLAPolicyRuntime:
    """按 LeRobot 正常训练路径加载 dataset metadata、policy 和 processors。

    这个函数只组织 LeRobot factory 调用顺序；真实 checkpoint 权重、processor 和 dataset
    是否加载成功由注入的 factory 或 LeRobot 原 factory 决定。
    """
    if validate_config and hasattr(train_cfg, "validate"):
        train_cfg.validate()

    policy_cfg = getattr(train_cfg, "policy", None)
    if policy_cfg is None:
        raise RuntimeError("TrainPipelineConfig.policy is required before building SAC-Flow runtime components.")

    if make_dataset_fn is None:
        from lerobot.datasets.factory import make_dataset as make_dataset_fn
    if make_policy_fn is None:
        from lerobot.policies.factory import make_policy as make_policy_fn
    if make_processors_fn is None:
        from lerobot.policies.factory import make_pre_post_processors as make_processors_fn

    dataset = make_dataset_fn(train_cfg)
    dataset_meta = getattr(dataset, "meta", None)
    if dataset_meta is None:
        raise RuntimeError("LeRobot dataset must expose .meta for SAC-Flow runtime construction.")

    policy = make_policy_fn(
        cfg=policy_cfg,
        ds_meta=dataset_meta,
        rename_map=getattr(train_cfg, "rename_map", {}),
    )

    processor_kwargs: dict[str, Any] = {}
    postprocessor_kwargs: dict[str, Any] = {}
    dataset_stats = getattr(dataset_meta, "stats", None)
    pretrained_path = getattr(policy_cfg, "pretrained_path", None)

    if (pretrained_path and not getattr(train_cfg, "resume", False)) or not pretrained_path:
        processor_kwargs["dataset_stats"] = dataset_stats

    if pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "normalizer_processor": {
                "stats": dataset_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {
                "rename_map": getattr(train_cfg, "rename_map", {}),
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_processors_fn(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    return SmolVLAPolicyRuntime(
        train_cfg=train_cfg,
        dataset=dataset,
        dataset_meta=dataset_meta,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
