from pathlib import Path
import unittest


SCRIPT_DIR = Path("scripts/rlinf_smolvla_libero")


class SACFlowScriptTest(unittest.TestCase):
    def test_gpu_smoke_script_supports_opt_in_wandb_and_uses_safe_budget(self):
        script = (SCRIPT_DIR / "gpu_smoke.sh").read_text(encoding="utf-8")

        self.assertIn('WANDB_ENABLE="${WANDB_ENABLE:-false}"', script)
        self.assertIn('WANDB_PROJECT="${WANDB_PROJECT:-smolvla-sac-flow-smoke-', script)
        self.assertIn("WANDB_MODE=disabled", script)
        self.assertIn("--sac-flow.wandb-enable=true", script)
        self.assertIn("--gpu-smoke", script)
        self.assertIn("--confirm-gpu-smoke", script)
        self.assertIn("--sac-flow.actor-train-scope=action_path", script)
        self.assertIn("--sac-flow.max-train-steps=2", script)
        self.assertIn("--sac-flow.save-checkpoint=false", script)
        self.assertIn("--sac-flow.num-updates-per-step=1", script)
        self.assertIn("--dataset.root=/root/autodl-fs/hf_libero_full", script)
        self.assertIn("--env.observation_height=256", script)
        self.assertIn("--env.observation_width=256", script)

    def test_short_run_script_enables_wandb_and_supports_manual_project(self):
        script = (SCRIPT_DIR / "short_run_wandb.sh").read_text(encoding="utf-8")

        self.assertIn('WANDB_PROJECT="${WANDB_PROJECT:-smolvla-sac-flow-short-', script)
        self.assertIn("WANDB_MODE=online", script)
        self.assertIn("--train-run", script)
        self.assertIn("--sac-flow.wandb-enable=true", script)
        self.assertIn("--sac-flow.actor-train-scope=action_path", script)
        self.assertIn("--sac-flow.max-train-steps=280", script)
        self.assertIn("--sac-flow.num-envs=4", script)
        self.assertIn("--sac-flow.batch-size=8", script)
        self.assertIn("--sac-flow.min-buffer-size=256", script)
        self.assertIn("--sac-flow.replay-capacity=2048", script)
        self.assertIn("--sac-flow.actor-warmup-updates=2000", script)
        self.assertIn('--sac-flow.save-checkpoint="${SAC_FLOW_SAVE_CHECKPOINT:-false}"', script)
        self.assertIn("--sac-flow.noise-std-train=0.02", script)
        self.assertIn("--sac-flow.actor-lr=3e-6", script)
        self.assertIn("--sac-flow.actor-agg-q=min", script)
        self.assertIn("--sac-flow.entropy-regularization=false", script)
        self.assertIn("--sac-flow.backup-entropy=false", script)
        self.assertIn("--env.observation_height=256", script)
        self.assertIn("--env.observation_width=256", script)

    def test_baseline_script_enables_wandb_and_keeps_gradual_updates(self):
        script = (SCRIPT_DIR / "baseline_run_wandb.sh").read_text(encoding="utf-8")

        self.assertIn('WANDB_PROJECT="${WANDB_PROJECT:-smolvla-sac-flow-baseline-', script)
        self.assertIn("WANDB_MODE=online", script)
        self.assertIn("--train-run", script)
        self.assertIn("--sac-flow.wandb-enable=true", script)
        self.assertIn("--sac-flow.actor-train-scope=action_path", script)
        self.assertIn("--sac-flow.num-updates-per-step=16", script)
        self.assertIn("--sac-flow.num-envs=4", script)
        self.assertIn("--sac-flow.batch-size=16", script)
        self.assertIn("--sac-flow.min-buffer-size=256", script)
        self.assertIn("--sac-flow.replay-capacity=2048", script)
        self.assertIn('--sac-flow.save-checkpoint="${SAC_FLOW_SAVE_CHECKPOINT:-false}"', script)
        self.assertIn("--sac-flow.actor-lr=3e-6", script)
        self.assertIn("--sac-flow.actor-agg-q=min", script)
        self.assertIn("--sac-flow.entropy-regularization=false", script)
        self.assertIn("--sac-flow.backup-entropy=false", script)
        self.assertIn("--env.observation_height=256", script)
        self.assertIn("--env.observation_width=256", script)


if __name__ == "__main__":
    unittest.main()
