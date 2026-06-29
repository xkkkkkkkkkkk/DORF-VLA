import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


def load_libero_env_config():
    """轻量加载 configs.py，避免本地缺少可选依赖时阻塞配置层测试。"""
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "lerobot" / "envs" / "configs.py"
    module_name = "_libero_env_configs_under_test"

    stubbed_names = (
        "draccus",
        "lerobot.configs",
        "lerobot.configs.types",
        "lerobot.robots",
        "lerobot.teleoperators",
        "lerobot.teleoperators.config",
        "lerobot.utils",
        "lerobot.utils.constants",
    )
    original_modules = {name: sys.modules.get(name) for name in stubbed_names}
    parent_attr_specs = (
        ("lerobot", "configs"),
        ("lerobot", "robots"),
        ("lerobot", "teleoperators"),
        ("lerobot", "utils"),
        ("lerobot.teleoperators", "config"),
    )
    original_parent_attrs = {}
    for parent_name, attr_name in parent_attr_specs:
        parent = sys.modules.get(parent_name)
        original_parent_attrs[(parent_name, attr_name)] = (
            parent is not None and hasattr(parent, attr_name),
            getattr(parent, attr_name, None) if parent is not None else None,
        )

    class ChoiceRegistry:
        @classmethod
        def register_subclass(cls, name=None):
            def decorator(subclass):
                return subclass

            return decorator

        @classmethod
        def get_choice_name(cls, subclass):
            return subclass.__name__

    class FeatureType:
        STATE = "STATE"
        VISUAL = "VISUAL"
        ENV = "ENV"
        ACTION = "ACTION"

    @dataclass
    class PolicyFeature:
        type: object
        shape: tuple

    draccus_stub = types.ModuleType("draccus")
    draccus_stub.ChoiceRegistry = ChoiceRegistry

    configs_pkg_stub = types.ModuleType("lerobot.configs")
    configs_types_stub = types.ModuleType("lerobot.configs.types")
    configs_types_stub.FeatureType = FeatureType
    configs_types_stub.PolicyFeature = PolicyFeature

    robots_stub = types.ModuleType("lerobot.robots")
    robots_stub.RobotConfig = type("RobotConfig", (), {})

    teleoperators_stub = types.ModuleType("lerobot.teleoperators")
    teleop_config_stub = types.ModuleType("lerobot.teleoperators.config")
    teleop_config_stub.TeleoperatorConfig = type("TeleoperatorConfig", (), {})

    utils_pkg_stub = types.ModuleType("lerobot.utils")
    constants_stub = types.ModuleType("lerobot.utils.constants")
    for name, value in {
        "ACTION": "action",
        "LIBERO_KEY_EEF_MAT": "robot_state/eef/mat",
        "LIBERO_KEY_EEF_POS": "robot_state/eef/pos",
        "LIBERO_KEY_EEF_QUAT": "robot_state/eef/quat",
        "LIBERO_KEY_GRIPPER_QPOS": "robot_state/gripper/qpos",
        "LIBERO_KEY_GRIPPER_QVEL": "robot_state/gripper/qvel",
        "LIBERO_KEY_JOINTS_POS": "robot_state/joints/pos",
        "LIBERO_KEY_JOINTS_VEL": "robot_state/joints/vel",
        "LIBERO_KEY_PIXELS_AGENTVIEW": "pixels/agentview_image",
        "LIBERO_KEY_PIXELS_EYE_IN_HAND": "pixels/robot0_eye_in_hand_image",
        "OBS_ENV_STATE": "observation.environment_state",
        "OBS_IMAGE": "observation.image",
        "OBS_IMAGES": "observation.images",
        "OBS_STATE": "observation.state",
    }.items():
        setattr(constants_stub, name, value)

    sys.modules["draccus"] = draccus_stub
    sys.modules["lerobot.configs"] = configs_pkg_stub
    sys.modules["lerobot.configs.types"] = configs_types_stub
    sys.modules["lerobot.robots"] = robots_stub
    sys.modules["lerobot.teleoperators"] = teleoperators_stub
    sys.modules["lerobot.teleoperators.config"] = teleop_config_stub
    sys.modules["lerobot.utils"] = utils_pkg_stub
    sys.modules["lerobot.utils.constants"] = constants_stub

    try:
        module = types.ModuleType(module_name)
        module.__file__ = str(module_path)
        sys.modules[module_name] = module
        source = module_path.read_text(encoding="utf-8")
        code = compile("from __future__ import annotations\n" + source, str(module_path), "exec")
        exec(code, module.__dict__)
        return module.LiberoEnv
    finally:
        sys.modules.pop(module_name, None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        for (parent_name, attr_name), (had_attr, original_attr) in original_parent_attrs.items():
            parent = sys.modules.get(parent_name)
            if parent is None:
                continue
            if had_attr:
                setattr(parent, attr_name, original_attr)
            elif hasattr(parent, attr_name):
                delattr(parent, attr_name)


LiberoEnv = load_libero_env_config()


class LiberoEnvConfigGymKwargsTest(unittest.TestCase):
    def test_gym_kwargs_forward_declared_observation_size_and_camera_mapping(self):
        camera_name_mapping = {
            "agentview_image": "observation.images.image",
            "robot0_eye_in_hand_image": "observation.images.image2",
        }
        cfg = LiberoEnv(
            observation_height=128,
            observation_width=160,
            camera_name_mapping=camera_name_mapping,
        )

        kwargs = cfg.gym_kwargs
        for key in (
            "obs_type",
            "render_mode",
            "observation_height",
            "observation_width",
            "camera_name_mapping",
        ):
            self.assertIn(key, kwargs)

        self.assertEqual(kwargs["obs_type"], "pixels_agent_pos")
        self.assertEqual(kwargs["render_mode"], "rgb_array")
        self.assertEqual(kwargs["observation_height"], 128)
        self.assertEqual(kwargs["observation_width"], 160)
        self.assertEqual(kwargs["camera_name_mapping"], camera_name_mapping)

    def test_gym_kwargs_preserves_task_ids_when_configured(self):
        cfg = LiberoEnv(task_ids=[1, 3])

        kwargs = cfg.gym_kwargs
        self.assertIn("task_ids", kwargs)
        self.assertIn("observation_height", kwargs)
        self.assertIn("observation_width", kwargs)
        self.assertEqual(kwargs["task_ids"], [1, 3])
        self.assertEqual(kwargs["observation_height"], 360)
        self.assertEqual(kwargs["observation_width"], 360)


if __name__ == "__main__":
    unittest.main()
