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
        self.assertIn('--sac-flow.heldout-num-steps="${SAC_FLOW_HELDOUT_NUM_STEPS:-256}"', script)
        self.assertIn('--sac-flow.heldout-seed="${SAC_FLOW_HELDOUT_SEED:-2000}"', script)
        self.assertIn('--sac-flow.root-cause-diagnostics="${SAC_FLOW_ROOT_CAUSE_DIAGNOSTICS:-false}"', script)
        self.assertIn(
            '--sac-flow.root-cause-max-transitions-per-task="${SAC_FLOW_ROOT_CAUSE_MAX_TRANSITIONS_PER_TASK:-32}"',
            script,
        )
        self.assertIn(
            '--sac-flow.root-cause-gradient-repeats="${SAC_FLOW_ROOT_CAUSE_GRADIENT_REPEATS:-3}"',
            script,
        )
        self.assertIn("--sac-flow.noise-std-train=0.02", script)
        self.assertIn("--sac-flow.actor-lr=3e-6", script)
        self.assertIn(
            '--sac-flow.critic-random-action-strategy="${SAC_FLOW_CRITIC_RANDOM_ACTION_STRATEGY:-replay_local_gaussian}"',
            script,
        )
        self.assertIn(
            '--sac-flow.critic-random-action-std="${SAC_FLOW_CRITIC_RANDOM_ACTION_STD:-0.05}"',
            script,
        )
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
        self.assertIn('--sac-flow.heldout-num-steps="${SAC_FLOW_HELDOUT_NUM_STEPS:-256}"', script)
        self.assertIn('--sac-flow.heldout-seed="${SAC_FLOW_HELDOUT_SEED:-2000}"', script)
        self.assertIn('--sac-flow.root-cause-diagnostics="${SAC_FLOW_ROOT_CAUSE_DIAGNOSTICS:-false}"', script)
        self.assertIn(
            '--sac-flow.root-cause-max-transitions-per-task="${SAC_FLOW_ROOT_CAUSE_MAX_TRANSITIONS_PER_TASK:-32}"',
            script,
        )
        self.assertIn(
            '--sac-flow.root-cause-gradient-repeats="${SAC_FLOW_ROOT_CAUSE_GRADIENT_REPEATS:-3}"',
            script,
        )
        self.assertIn("--sac-flow.actor-lr=3e-6", script)
        self.assertIn("--sac-flow.actor-agg-q=min", script)
        self.assertIn(
            '--sac-flow.critic-random-action-strategy="${SAC_FLOW_CRITIC_RANDOM_ACTION_STRATEGY:-replay_local_gaussian}"',
            script,
        )
        self.assertIn(
            '--sac-flow.critic-random-action-std="${SAC_FLOW_CRITIC_RANDOM_ACTION_STD:-0.05}"',
            script,
        )
        self.assertIn("--sac-flow.entropy-regularization=false", script)
        self.assertIn("--sac-flow.backup-entropy=false", script)
        self.assertIn("--env.observation_height=256", script)
        self.assertIn("--env.observation_width=256", script)

    def test_intervention_recheck_is_bounded_paired_critic_only_run(self):
        script = (SCRIPT_DIR / "intervention_recheck.sh").read_text(encoding="utf-8")

        self.assertIn('TASK_IDS="${SAC_FLOW_TASK_IDS:-[9]}"', script)
        self.assertIn('NUM_ENVS="${SAC_FLOW_NUM_ENVS:-10}"', script)
        self.assertIn('--env.task_ids="${TASK_IDS}"', script)
        self.assertIn("--env.paired_init_states=true", script)
        self.assertIn('--sac-flow.seed="${SAC_FLOW_SEED:-0}"', script)
        self.assertIn('--output_dir="${OUTPUT_DIR}"', script)
        self.assertIn("--sac-flow.max-train-steps=280", script)
        self.assertIn('--sac-flow.num-envs="${NUM_ENVS}"', script)
        self.assertIn("--sac-flow.replay-capacity=8192", script)
        self.assertIn("--sac-flow.actor-updates-enabled=false", script)
        self.assertIn("--sac-flow.save-checkpoint=true", script)
        self.assertIn("--sac-flow.root-cause-diagnostics=true", script)
        self.assertIn("--sac-flow.critic-intervention-fraction=0.5", script)
        self.assertIn("--sac-flow.critic-intervention-noise-std=0.3", script)
        self.assertIn("--sac-flow.critic-intervention-balanced-sampling=true", script)
        self.assertIn("--sac-flow.critic-intervention-pairing=true", script)
        self.assertIn("--sac-flow.critic-pairwise-coef=1.0", script)
        self.assertIn("--sac-flow.critic-pairwise-margin=0.05", script)
        self.assertIn("--sac-flow.critic-step-penalty=0.001", script)
        self.assertIn("--sac-flow.critic-conservative-coef=0", script)
        self.assertIn("--sac-flow.critic-monte-carlo-coef=0.1", script)
        self.assertIn("--sac-flow.wandb-enable=false", script)


if __name__ == "__main__":
    unittest.main()
