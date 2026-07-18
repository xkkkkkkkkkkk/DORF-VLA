import unittest

from lerobot.rlinf_smolvla_libero.config import SACFlowConfig


class FakeWandB:
    def __init__(self):
        self.init_calls = []
        self.log_calls = []
        self.finish_calls = 0

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return object()

    def log(self, metrics, step=None):
        self.log_calls.append((metrics, step))

    def finish(self):
        self.finish_calls += 1


class SACFlowWandBLoggerTest(unittest.TestCase):
    def test_disabled_logger_does_not_import_wandb(self):
        from lerobot.rlinf_smolvla_libero.wandb_logger import SACFlowWandBLogger

        def fail_import():
            raise AssertionError("wandb should not be imported")

        logger = SACFlowWandBLogger(SACFlowConfig(wandb_enable=False), import_wandb_fn=fail_import)
        logger.start({"scope": "action_path"})
        logger.log({"train/sac/critic_loss": 1.0}, step=1)
        logger.finish()

    def test_enabled_logger_initializes_and_logs_to_wandb(self):
        from lerobot.rlinf_smolvla_libero.wandb_logger import SACFlowWandBLogger

        fake_wandb = FakeWandB()
        config = SACFlowConfig(
            wandb_enable=True,
            wandb_project="manual-project",
            wandb_run_name="manual-run",
            wandb_mode="online",
            wandb_tags=("short", "action_path"),
        )

        logger = SACFlowWandBLogger(config, import_wandb_fn=lambda: fake_wandb)
        logger.start({"actor_train_scope": "action_path"})
        logger.log({"train/sac/critic_loss": 1.0}, step=7)
        logger.finish()

        self.assertEqual(fake_wandb.init_calls[0]["project"], "manual-project")
        self.assertEqual(fake_wandb.init_calls[0]["name"], "manual-run")
        self.assertEqual(fake_wandb.init_calls[0]["mode"], "online")
        self.assertEqual(fake_wandb.init_calls[0]["tags"], ["short", "action_path"])
        self.assertEqual(fake_wandb.init_calls[0]["config"], {"actor_train_scope": "action_path"})
        self.assertEqual(fake_wandb.log_calls, [({"train/sac/critic_loss": 1.0}, 7)])
        self.assertEqual(fake_wandb.finish_calls, 1)

    def test_formats_sac_update_metrics_with_rlinf_style_names(self):
        from lerobot.rlinf_smolvla_libero.wandb_logger import format_sac_update_metrics

        formatted = format_sac_update_metrics(
            {
                "critic_loss": 1.0,
                "actor_loss": 2.0,
                "alpha_loss": 3.0,
                "alpha": 0.5,
                "entropy": 4.0,
                "log_pi": -5.0,
                "kl_estimate": 0.3,
                "kl_penalty": 0.015,
                "q_mean": 6.0,
            }
        )

        self.assertEqual(
            formatted,
            {
                "train/sac/critic_loss": 1.0,
                "train/sac/actor_loss": 2.0,
                "train/sac/alpha_loss": 3.0,
                "train/sac/alpha": 0.5,
                "train/actor/entropy": 4.0,
                "train/actor/log_pi": -5.0,
                "train/actor/kl_estimate": 0.3,
                "train/actor/kl_penalty": 0.015,
                "train/critic/q_mean": 6.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
