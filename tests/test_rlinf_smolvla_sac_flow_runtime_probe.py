import os
import tempfile
import unittest
from pathlib import Path

from lerobot.rlinf_smolvla_libero.runtime_probe import build_runtime_probe, require_existing_path


class SACFlowRuntimeProbeTest(unittest.TestCase):
    def test_requires_existing_libero_root(self):
        env = {}
        with self.assertRaisesRegex(RuntimeError, "LEROBOT_LIBERO_ROOT is required"):
            build_runtime_probe(env=env, cli_overrides=["--policy.path=/tmp/checkpoint"])

    def test_requires_policy_path_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"LEROBOT_LIBERO_ROOT": tmpdir}
            with self.assertRaisesRegex(RuntimeError, "--policy.path"):
                build_runtime_probe(env=env, cli_overrides=[])

    def test_collects_libero_root_and_policy_path_without_loading_env(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as checkpoint:
            env = {"LEROBOT_LIBERO_ROOT": root}
            probe = build_runtime_probe(
                env=env,
                cli_overrides=[
                    f"--policy.path={checkpoint}",
                    "--env.type=libero",
                    "--sac-flow.device=cuda:0",
                ],
            )

        self.assertEqual(probe.libero_root, Path(root))
        self.assertEqual(probe.policy_path, Path(checkpoint))
        self.assertIn("--env.type=libero", probe.cli_overrides)
        self.assertEqual(probe.sac_flow_device, "cuda:0")

    def test_require_existing_path_reports_missing_path(self):
        missing = Path("/definitely/missing/smolvla/checkpoint")
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            require_existing_path(missing, label="policy checkpoint")


if __name__ == "__main__":
    unittest.main()
