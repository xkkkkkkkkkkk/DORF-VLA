import ast
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

    def test_non_dry_run_is_not_implemented(self):
        result = self.run_script(env={"LEROBOT_LIBERO_ROOT": "/tmp/libero"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Real SmolVLA/LIBERO SAC-Flow training is not wired yet", result.stderr)


if __name__ == "__main__":
    unittest.main()
