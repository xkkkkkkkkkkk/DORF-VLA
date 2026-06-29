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

    def test_non_dry_run_is_not_implemented(self):
        result = self.run_script(env={"LEROBOT_LIBERO_ROOT": "/tmp/libero"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Real SmolVLA/LIBERO SAC-Flow training is not wired yet", result.stderr)


if __name__ == "__main__":
    unittest.main()
