import unittest

import torch

from lerobot.rlinf_smolvla_libero.critic import (
    EntropyTemperature,
    MultiQHead,
    actor_loss,
    aggregate_q,
    alpha_loss,
    critic_loss,
    critic_target,
    soft_update,
)


class SACFlowCriticTest(unittest.TestCase):
    def test_q_head_outputs_one_value_per_head(self):
        q = MultiQHead(obs_dim=5, action_dim=3, hidden_dim=16, num_q_heads=4)
        values = q(torch.zeros(2, 5), torch.zeros(2, 3))
        self.assertEqual(values.shape, (2, 4))

    def test_q_head_accepts_positional_constructor_args(self):
        q = MultiQHead(5, 3, 16, 2)
        values = q(torch.zeros(2, 5), torch.zeros(2, 3))
        self.assertEqual(values.shape, (2, 2))

    def test_q_head_rejects_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "obs_dim must be a positive integer"):
            MultiQHead(0, 3, 16, 2)
        with self.assertRaisesRegex(ValueError, "num_q_heads must be a positive integer"):
            MultiQHead(5, 3, 16, 0)

    def test_q_head_rejects_feature_dim_mismatch(self):
        q = MultiQHead(obs_dim=5, action_dim=3, hidden_dim=16, num_q_heads=2)
        with self.assertRaisesRegex(ValueError, r"obs_features must have shape \[batch, obs_dim\]"):
            q(torch.zeros(2, 4), torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, r"actions must have shape \[batch, action_dim\]"):
            q(torch.zeros(2, 5), torch.zeros(2, 2))

    def test_q_head_rejects_non_2d_inputs(self):
        q = MultiQHead(obs_dim=5, action_dim=3, hidden_dim=16, num_q_heads=2)
        with self.assertRaises(ValueError):
            q(torch.zeros(2, 5, 1), torch.zeros(2, 3))

    def test_q_head_rejects_batch_mismatch(self):
        q = MultiQHead(obs_dim=5, action_dim=3, hidden_dim=16, num_q_heads=2)
        with self.assertRaises(ValueError):
            q(torch.zeros(2, 5), torch.zeros(3, 3))

    def test_critic_target_uses_discount_and_entropy_backup(self):
        reward = torch.tensor([[1.0]])
        done = torch.tensor([[False]])
        discount = torch.tensor([[0.5]])
        next_q = torch.tensor([[2.0, 3.0]])
        next_log_pi = torch.tensor([[-4.0]])
        alpha = torch.tensor(0.25)
        target = critic_target(reward, done, discount, next_q, next_log_pi, alpha, agg="min", backup_entropy=True)
        self.assertAlmostEqual(float(target.item()), 1.0 + 0.5 * (2.0 - 0.25 * -4.0))

    def test_critic_target_requires_bool_done(self):
        with self.assertRaises(ValueError):
            critic_target(
                torch.ones(1, 1),
                torch.zeros(1, 1),
                torch.ones(1, 1),
                torch.ones(1, 2),
                torch.zeros(1, 1),
                torch.tensor(0.2),
                agg="min",
                backup_entropy=True,
            )

    def test_critic_target_rejects_vector_alpha(self):
        with self.assertRaisesRegex(ValueError, "alpha must be a scalar tensor"):
            critic_target(
                torch.ones(2, 1),
                torch.zeros(2, 1, dtype=torch.bool),
                torch.ones(2, 1),
                torch.ones(2, 2),
                torch.zeros(2, 1),
                torch.ones(2),
                agg="min",
                backup_entropy=True,
            )

    def test_critic_target_rejects_column_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, r"reward must have shape \[batch, 1\]"):
            critic_target(
                torch.ones(2),
                torch.zeros(2, 1, dtype=torch.bool),
                torch.ones(2, 1),
                torch.ones(2, 2),
                torch.zeros(2, 1),
                torch.tensor(0.2),
                agg="min",
                backup_entropy=True,
            )
        with self.assertRaisesRegex(ValueError, r"next_log_pi must have shape \[batch, 1\]"):
            critic_target(
                torch.ones(2, 1),
                torch.zeros(2, 1, dtype=torch.bool),
                torch.ones(2, 1),
                torch.ones(2, 2),
                torch.zeros(2),
                torch.tensor(0.2),
                agg="min",
                backup_entropy=True,
            )

    def test_critic_loss_rejects_bad_target_shape(self):
        with self.assertRaisesRegex(ValueError, r"target must have shape \[batch, 1\]"):
            critic_loss(torch.ones(2, 2), torch.ones(2))

    def test_aggregate_q_rejects_non_2d_values(self):
        with self.assertRaises(ValueError):
            aggregate_q(torch.ones(1, 2, 1), agg="min")

    def test_aggregate_q_accepts_positional_agg(self):
        values = aggregate_q(torch.tensor([[1.0, 2.0]]), "mean")
        self.assertEqual(values.tolist(), [[1.5]])

    def test_soft_update_rejects_tau_outside_unit_interval(self):
        source = MultiQHead(obs_dim=1, action_dim=1, hidden_dim=4, num_q_heads=1)
        target = MultiQHead(obs_dim=1, action_dim=1, hidden_dim=4, num_q_heads=1)
        with self.assertRaises(ValueError):
            soft_update(source, target, tau=1.1)

    def test_actor_loss_matches_sac_formula(self):
        q_pi = torch.tensor([[2.0, 3.0]])
        log_pi = torch.tensor([[-1.5]])
        loss = actor_loss(q_pi, log_pi, torch.tensor(0.2), agg="min")
        self.assertAlmostEqual(float(loss.item()), 0.2 * -1.5 - 2.0)

    def test_actor_loss_accepts_positional_agg(self):
        q_pi = torch.tensor([[2.0, 3.0]])
        log_pi = torch.tensor([[-1.5]])
        loss = actor_loss(q_pi, log_pi, torch.tensor(0.2), "min")
        self.assertAlmostEqual(float(loss.item()), 0.2 * -1.5 - 2.0)

    def test_actor_loss_rejects_bad_log_pi_shape(self):
        with self.assertRaisesRegex(ValueError, r"log_pi must have shape \[batch, 1\]"):
            actor_loss(torch.ones(2, 2), torch.zeros(2), torch.tensor(0.2), "min")

    def test_actor_loss_rejects_vector_alpha(self):
        with self.assertRaisesRegex(ValueError, "alpha must be a scalar tensor"):
            actor_loss(torch.ones(2, 2), torch.zeros(2, 1), torch.ones(2), "min")

    def test_alpha_loss_matches_formula_and_rejects_bad_shape(self):
        log_alpha = torch.tensor(-1.6094379)
        log_pi = torch.tensor([[-1.5], [-2.5]])
        target_entropy = -3.0
        loss = alpha_loss(log_alpha, log_pi, target_entropy)
        expected = -(log_alpha * (log_pi.detach() + target_entropy)).mean()
        self.assertTrue(torch.allclose(loss, expected))

        with self.assertRaisesRegex(ValueError, r"log_pi must have shape \[batch, 1\]"):
            alpha_loss(log_alpha, torch.zeros(2), target_entropy)

    def test_alpha_loss_rejects_vector_alpha(self):
        with self.assertRaisesRegex(ValueError, "log_alpha must be a scalar tensor"):
            alpha_loss(torch.ones(2), torch.zeros(2, 1), -3.0)

    def test_alpha_is_positive(self):
        temp = EntropyTemperature(initial_alpha=0.01)
        self.assertGreater(float(temp.alpha.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
