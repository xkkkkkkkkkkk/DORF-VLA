import unittest

import torch
import torch.nn as nn

from lerobot.rlinf_smolvla_libero.config import SACFlowConfig
from lerobot.rlinf_smolvla_libero.critic import MultiQHead
from lerobot.rlinf_smolvla_libero.trainer import SACFlowTrainer


class DummyActor(nn.Module):
    def __init__(self, obs_dim=3, action_dim=2):
        super().__init__()
        self.linear = nn.Linear(obs_dim, action_dim)
        self.feature = nn.Linear(obs_dim, obs_dim)

    def encode_obs(self, obs):
        return self.feature(obs["states"])

    def sample_chunk(self, obs, train):
        features = self.encode_obs(obs)
        flat_actions = self.linear(features)
        # SAC trainer expects log_pi [B,1]; keep differentiable wrt actor for actor_loss.
        log_pi = -0.5 * flat_actions.pow(2).sum(dim=-1, keepdim=True)
        raw_chunk = flat_actions.reshape(flat_actions.shape[0], 1, flat_actions.shape[1])
        return flat_actions, log_pi, features, raw_chunk

    def sample_chunk_with_kl(self, obs):
        flat_actions, log_pi, features, raw_chunk = self.sample_chunk(obs, train=True)
        # Keep the KL differentiable so this exercises the actor gradient path.
        kl_estimate = 0.1 + 0.01 * flat_actions.sum(dim=-1, keepdim=True)
        return flat_actions, log_pi, features, raw_chunk, kl_estimate


def make_batch(batch_size=4):
    return {
        "curr_obs": {"states": torch.randn(batch_size, 3)},
        "next_obs": {"states": torch.randn(batch_size, 3)},
        "actions": torch.randn(batch_size, 2),
        "rewards": torch.randn(batch_size, 1),
        "terminations": torch.zeros(batch_size, 1, dtype=torch.bool),
        "discounts": torch.full((batch_size, 1), 0.96),
        "horizons": torch.ones(batch_size, 1, dtype=torch.long),
    }


class SACFlowTrainerTest(unittest.TestCase):
    def test_config_defaults_match_rlinf_style_values(self):
        cfg = SACFlowConfig()
        self.assertEqual(cfg.gamma, 0.96)
        self.assertEqual(cfg.tau, 0.005)
        self.assertEqual(cfg.initial_alpha, 0.01)
        self.assertIsNone(cfg.target_entropy)
        self.assertEqual(cfg.critic_actor_ratio, 4)
        self.assertEqual(cfg.actor_warmup_updates, 2000)
        self.assertTrue(cfg.actor_updates_enabled)
        self.assertEqual(cfg.num_updates_per_step, 64)
        self.assertEqual(cfg.replay_capacity, 200)
        self.assertEqual(cfg.min_buffer_size, 2)
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.num_q_heads, 10)
        self.assertEqual(cfg.hidden_dim, 256)
        self.assertEqual(cfg.noise_std_train, 0.3)
        self.assertEqual(cfg.noise_std_rollout, 0.02)
        self.assertFalse(cfg.entropy_regularization)
        self.assertFalse(cfg.backup_entropy)
        self.assertEqual(cfg.agg_q, "min")
        self.assertEqual(cfg.actor_agg_q, "min")
        self.assertEqual(cfg.actor_lr, 3e-4)
        self.assertEqual(cfg.kl_penalty_coef, 0.05)
        self.assertEqual(cfg.critic_lr, 3e-4)
        self.assertEqual(cfg.alpha_lr, 3e-4)
        self.assertEqual(cfg.grad_clip_norm, 1.0)
        self.assertEqual(cfg.device, "cpu")

    def test_update_sac_runs_critic_actor_alpha_and_target_update(self):
        torch.manual_seed(0)
        actor = DummyActor()
        q = MultiQHead(3, 2, 16, 2)
        target_q = MultiQHead(3, 2, 16, 2)
        target_q.load_state_dict(q.state_dict())
        cfg = SACFlowConfig(
            critic_actor_ratio=1,
            actor_warmup_updates=1,
            hidden_dim=16,
            num_q_heads=2,
            entropy_regularization=True,
            backup_entropy=True,
        )
        trainer = SACFlowTrainer(
            actor=actor,
            q_network=q,
            target_q_network=target_q,
            config=cfg,
            actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.Adam(q.parameters(), lr=1e-3),
        )
        trainer.update_step = 1
        before_log_alpha = trainer.temperature.log_alpha.detach().clone()
        before_target = [p.detach().clone() for p in target_q.parameters()]
        metrics = trainer.update_sac(make_batch())
        self.assertIn("critic_loss", metrics)
        self.assertIn("actor_loss", metrics)
        self.assertIn("alpha_loss", metrics)
        self.assertIn("kl_estimate", metrics)
        self.assertIn("kl_penalty", metrics)
        self.assertIn("alpha", metrics)
        self.assertIn("critic_grad_norm", metrics)
        self.assertIn("q_head_span", metrics)
        self.assertIn("target_q_mean", metrics)
        self.assertIn("batch_positive_reward_fraction", metrics)
        self.assertEqual(metrics["critic_update_count"], 2.0)
        self.assertEqual(metrics["actor_update_count"], 1.0)
        self.assertEqual(trainer.update_step, 2)
        self.assertFalse(torch.equal(before_log_alpha, trainer.temperature.log_alpha.detach()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_target, target_q.parameters())))

    def test_update_sac_skips_actor_until_ratio(self):
        torch.manual_seed(0)
        actor = DummyActor()
        q = MultiQHead(3, 2, 16, 2)
        target_q = MultiQHead(3, 2, 16, 2)
        target_q.load_state_dict(q.state_dict())
        cfg = SACFlowConfig(critic_actor_ratio=4, actor_warmup_updates=1, hidden_dim=16, num_q_heads=2)
        trainer = SACFlowTrainer(
            actor=actor,
            q_network=q,
            target_q_network=target_q,
            config=cfg,
            actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.Adam(q.parameters(), lr=1e-3),
        )
        trainer.update_step = 1
        before_actor = [p.detach().clone() for p in actor.parameters()]
        before_target = [p.detach().clone() for p in target_q.parameters()]
        metrics = trainer.update_sac(make_batch())
        self.assertIn("critic_loss", metrics)
        self.assertNotIn("actor_loss", metrics)
        self.assertEqual(trainer.update_step, 2)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_actor, actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_target, target_q.parameters())))

    def test_default_update_excludes_path_entropy_and_keeps_alpha_fixed(self):
        torch.manual_seed(0)
        actor = DummyActor()
        q = MultiQHead(3, 2, 16, 2)
        target_q = MultiQHead(3, 2, 16, 2)
        target_q.load_state_dict(q.state_dict())
        cfg = SACFlowConfig(critic_actor_ratio=1, actor_warmup_updates=1, hidden_dim=16, num_q_heads=2)
        trainer = SACFlowTrainer(actor=actor, q_network=q, target_q_network=target_q, config=cfg)
        trainer.update_step = 1

        before_log_alpha = trainer.temperature.log_alpha.detach().clone()
        metrics = trainer.update_sac(make_batch())

        self.assertIn("actor_loss", metrics)
        self.assertNotIn("alpha_loss", metrics)
        torch.testing.assert_close(trainer.temperature.log_alpha.detach(), before_log_alpha)

    def test_update_sac_skips_actor_during_critic_warmup(self):
        actor = DummyActor()
        q = MultiQHead(3, 2, 16, 2)
        target_q = MultiQHead(3, 2, 16, 2)
        cfg = SACFlowConfig(critic_actor_ratio=1, actor_warmup_updates=2, hidden_dim=16, num_q_heads=2)
        trainer = SACFlowTrainer(actor=actor, q_network=q, target_q_network=target_q, config=cfg)

        before_actor = [parameter.detach().clone() for parameter in actor.parameters()]
        metrics = trainer.update_sac(make_batch())

        self.assertIn("critic_loss", metrics)
        self.assertNotIn("actor_loss", metrics)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_actor, actor.parameters())))

    def test_actor_updates_can_be_explicitly_disabled_after_warmup(self):
        actor = DummyActor()
        q = MultiQHead(3, 2, 16, 2)
        target_q = MultiQHead(3, 2, 16, 2)
        cfg = SACFlowConfig(
            critic_actor_ratio=1,
            actor_warmup_updates=1,
            actor_updates_enabled=False,
            hidden_dim=16,
            num_q_heads=2,
        )
        trainer = SACFlowTrainer(actor=actor, q_network=q, target_q_network=target_q, config=cfg)
        trainer.update_step = 1
        before_actor = [parameter.detach().clone() for parameter in actor.parameters()]

        metrics = trainer.update_sac(make_batch())

        self.assertNotIn("actor_loss", metrics)
        self.assertEqual(metrics["actor_update_count"], 0.0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_actor, actor.parameters())))

    def test_infers_target_entropy_from_action_dim_when_missing(self):
        trainer = SACFlowTrainer(
            actor=DummyActor(),
            q_network=MultiQHead(3, 2, 16, 2),
            target_q_network=MultiQHead(3, 2, 16, 2),
            config=SACFlowConfig(target_entropy=None),
        )
        self.assertEqual(trainer.target_entropy, -2.0)

    def test_update_sac_reports_missing_batch_keys(self):
        trainer = SACFlowTrainer(
            actor=DummyActor(),
            q_network=MultiQHead(3, 2, 16, 2),
            target_q_network=MultiQHead(3, 2, 16, 2),
            config=SACFlowConfig(),
        )
        batch = make_batch()
        del batch["discounts"]
        with self.assertRaisesRegex(KeyError, "discounts"):
            trainer.update_sac(batch)


if __name__ == "__main__":
    unittest.main()
