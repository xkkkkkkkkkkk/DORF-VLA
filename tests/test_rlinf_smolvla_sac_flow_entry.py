import ast
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = Path("lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py")


class SACFlowEntryTest(unittest.TestCase):
    def run_script(self, *args, env=None):
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            text=True,
            capture_output=True,
            env=merged_env,
            check=False,
        )


    def test_module_docstring_documents_runtime_requirements(self):
        source = SCRIPT.read_text(encoding="utf-8")
        docstring = ast.get_docstring(ast.parse(source))
        self.assertIsNotNone(docstring)
        self.assertIn("LEROBOT_LIBERO_ROOT", docstring)
        self.assertIn("SmolVLA checkpoint/config overrides", docstring)


    def test_dry_run_does_not_assert_cpu_as_training_device(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn('"device": "cpu"', source)

    def test_sac_flow_config_allows_cuda_device_string(self):
        from lerobot.rlinf_smolvla_libero.config import SACFlowConfig

        config = SACFlowConfig(device="cuda:0")
        self.assertEqual(config.device, "cuda:0")

    def test_dry_run_requires_libero_root(self):
        env = os.environ.copy()
        env.pop("LEROBOT_LIBERO_ROOT", None)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run"],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LEROBOT_LIBERO_ROOT is required for LIBERO SAC-Flow smoke runs.", result.stderr)

    def test_dry_run_succeeds_with_libero_root(self):
        result = self.run_script("--dry-run", env={"LEROBOT_LIBERO_ROOT": "/tmp/libero"})
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("SAC-Flow dry-run passed", result.stdout)


    def test_probe_runtime_accepts_lerobot_style_overrides(self):
        import tempfile

        with tempfile.TemporaryDirectory() as libero_root, tempfile.TemporaryDirectory() as policy_path:
            result = self.run_script(
                "--probe-runtime",
                f"--policy.path={policy_path}",
                "--env.type=libero",
                "--sac-flow.device=cuda:0",
                env={"LEROBOT_LIBERO_ROOT": libero_root},
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("SAC-Flow runtime probe passed", result.stdout)
        self.assertIn("device=cuda:0", result.stdout)

    def test_probe_runtime_fails_before_training_when_policy_path_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as libero_root:
            result = self.run_script("--probe-runtime", env={"LEROBOT_LIBERO_ROOT": libero_root})

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--policy.path is required", result.stderr)


    def test_probe_runtime_parses_lerobot_overrides_without_model_or_env_creation(self):
        import tempfile

        if importlib.util.find_spec("draccus") is None:
            self.skipTest("draccus is not installed in the local lightweight Python")

        with tempfile.TemporaryDirectory() as libero_root, tempfile.TemporaryDirectory() as policy_path:
            result = self.run_script(
                "--probe-runtime",
                "--dataset.repo_id=local/test",
                f"--policy.path={policy_path}",
                "--env.type=libero",
                "--batch_size=2",
                "--steps=3",
                "--sac-flow.device=cuda:0",
                env={"LEROBOT_LIBERO_ROOT": libero_root},
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("SAC-Flow runtime probe passed", result.stdout)
        self.assertIn("train_steps=3", result.stdout)
        self.assertIn("batch_size=2", result.stdout)
        self.assertIn("device=cuda:0", result.stdout)

    def test_probe_runtime_accepts_split_policy_path_form(self):
        import tempfile

        with tempfile.TemporaryDirectory() as libero_root, tempfile.TemporaryDirectory() as policy_path:
            result = self.run_script(
                "--probe-runtime",
                "--dataset.repo_id=local/test",
                "--policy.path",
                policy_path,
                "--env.type=libero",
                env={"LEROBOT_LIBERO_ROOT": libero_root},
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("SAC-Flow runtime probe passed", result.stdout)

    def test_train_config_hint_accepts_split_config_path_form(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("sac_flow_entry", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertTrue(module._should_parse_train_config(["--config_path", "/tmp/train.yaml"]))

    def test_temporary_cli_overrides_expose_policy_path_to_validate_helpers(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("sac_flow_entry", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        original_argv = list(sys.argv)
        with module._temporary_cli_overrides(["--policy.path=HuggingFaceVLA/smolvla_libero"]):
            self.assertIn("--policy.path=HuggingFaceVLA/smolvla_libero", sys.argv)

        self.assertEqual(sys.argv, original_argv)

    def test_gpu_smoke_keeps_policy_path_visible_during_runtime_build(self):
        import importlib.util
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch

        spec = importlib.util.spec_from_file_location("sac_flow_entry", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as libero_root, tempfile.TemporaryDirectory() as policy_path:
            cli_overrides = [
                f"--policy.path={policy_path}",
                "--dataset.repo_id=local/test",
                "--sac-flow.device=cpu",
            ]

            def fake_smoke(**kwargs):
                self.assertIn(f"--policy.path={policy_path}", sys.argv)
                return SimpleNamespace(steps=2, checkpoint_dir=None)

            old_libero_root = os.environ.get("LEROBOT_LIBERO_ROOT")
            os.environ["LEROBOT_LIBERO_ROOT"] = libero_root
            try:
                with patch.object(module, "parse_train_config_from_overrides", return_value=SimpleNamespace()):
                    with patch("lerobot.rlinf_smolvla_libero.smoke_runner.run_sac_flow_gpu_smoke", fake_smoke):
                        module.run_gpu_smoke(cli_overrides, confirm_gpu_smoke=True)
            finally:
                if old_libero_root is None:
                    os.environ.pop("LEROBOT_LIBERO_ROOT", None)
                else:
                    os.environ["LEROBOT_LIBERO_ROOT"] = old_libero_root

    def test_train_run_builds_wandb_enabled_config_and_calls_runner(self):
        import importlib.util
        import tempfile
        from types import SimpleNamespace

        spec = importlib.util.spec_from_file_location("sac_flow_entry", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        events = []

        class FakeLogger:
            def __init__(self, config):
                events.append(("logger", config.wandb_enable, config.wandb_project, config.actor_train_scope))

            def start(self, run_config):
                events.append(("start", run_config["actor_train_scope"]))

            def finish(self):
                events.append(("finish",))

        def fake_run(**kwargs):
            events.append(
                (
                    "run",
                    kwargs["run_cfg"].max_train_steps,
                    kwargs["run_cfg"].num_updates_per_step,
                    kwargs["sac_config"].wandb_enable,
                    kwargs["sac_config"].actor_train_scope,
                )
            )
            return SimpleNamespace(steps=100, checkpoint_dir=None)

        with tempfile.TemporaryDirectory() as libero_root, tempfile.TemporaryDirectory() as policy_path:
            old_libero_root = os.environ.get("LEROBOT_LIBERO_ROOT")
            os.environ["LEROBOT_LIBERO_ROOT"] = libero_root
            try:
                module.run_train_run(
                    [
                        f"--policy.path={policy_path}",
                        "--dataset.repo_id=local/test",
                        "--sac-flow.max-train-steps=100",
                        "--sac-flow.num-updates-per-step=4",
                        "--sac-flow.wandb-project=manual-project",
                    ],
                    run_fn=fake_run,
                    logger_cls=FakeLogger,
                    parse_train_config_fn=lambda overrides: SimpleNamespace(),
                )
            finally:
                if old_libero_root is None:
                    os.environ.pop("LEROBOT_LIBERO_ROOT", None)
                else:
                    os.environ["LEROBOT_LIBERO_ROOT"] = old_libero_root

        self.assertEqual(events[0], ("logger", True, "manual-project", "action_path"))
        self.assertEqual(events[1], ("start", "action_path"))
        self.assertEqual(events[2], ("run", 100, 4, True, "action_path"))
        self.assertEqual(events[3], ("finish",))

    def test_preflight_device_accepts_cpu_without_model_or_env_creation(self):
        result = self.run_script("--preflight-device", "--sac-flow.device=cpu")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("SAC-Flow device preflight passed", result.stdout)
        self.assertIn("device=cpu", result.stdout)

    def test_help_documents_confirmation_for_every_smoke_run(self):
        result = self.run_script("--help")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Required for --gpu-smoke.", result.stdout)
        self.assertNotIn("requests a CUDA device", result.stdout)

    def test_gpu_smoke_requires_explicit_confirmation_before_training(self):
        result = self.run_script(
            "--gpu-smoke",
            "--sac-flow.device=cuda:0",
            env={"LEROBOT_LIBERO_ROOT": "/tmp/libero"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("confirm", result.stderr)
        self.assertNotIn("Real SmolVLA/LIBERO SAC-Flow training is not wired yet", result.stderr)

    def test_gpu_smoke_requires_train_config_hint_before_draccus_parse(self):
        import tempfile

        with tempfile.TemporaryDirectory() as libero_root, tempfile.TemporaryDirectory() as policy_path:
            result = self.run_script(
                "--gpu-smoke",
                "--confirm-gpu-smoke",
                f"--policy.path={policy_path}",
                "--sac-flow.device=cpu",
                env={"LEROBOT_LIBERO_ROOT": libero_root},
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--config_path or --dataset.repo_id", result.stderr)
        self.assertNotIn("No module named 'draccus'", result.stderr)

    def test_non_dry_run_requires_explicit_mode(self):
        result = self.run_script(env={"LEROBOT_LIBERO_ROOT": "/tmp/libero"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Choose --dry-run, --probe-runtime, --preflight-device, --gpu-smoke, or --train-run", result.stderr)


if __name__ == "__main__":
    unittest.main()
