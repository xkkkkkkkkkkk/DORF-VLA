import unittest

import torch

from lerobot.rlinf_smolvla_libero.critic import (
    EntropyTemperature,
    MultiQHead,
    actor_loss,
    aggregate_q,
    alpha_loss,
    conservative_q_penalty,
    critic_loss,
    critic_target,
    pairwise_q_ranking_loss,
    sample_critic_actions,
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

    def test_conservative_q_penalty_compares_random_actions_to_replay_actions(self):
        q_data = torch.tensor([[1.0, 1.1], [0.5, 0.6]])
        q_random = torch.tensor(
            [
                [[1.4, 1.5], [1.2, 1.3]],
                [[0.9, 1.0], [0.8, 0.9]],
            ]
        )

        penalty = conservative_q_penalty(q_data, q_random, agg="min")

        self.assertGreater(float(penalty), 0.0)

    def test_conservative_q_penalty_has_a_finite_lower_bound(self):
        q_data = torch.tensor([[1.0, 1.1]])
        q_random = torch.tensor([[[1.4, 1.5], [1.2, 1.3]]])

        penalty = conservative_q_penalty(q_data, q_random, agg="min", margin=0.1)
        larger_data_q_penalty = conservative_q_penalty(
            q_data + 1000.0,
            q_random,
            agg="min",
            margin=0.1,
        )

        self.assertGreaterEqual(float(penalty), 0.0)
        self.assertEqual(float(larger_data_q_penalty), 0.0)

    def test_conservative_q_penalty_is_invariant_to_common_q_shift(self):
        q_data = torch.tensor([[1.0, 1.1], [0.5, 0.6]])
        q_random = torch.tensor(
            [
                [[1.4, 1.5], [1.2, 1.3]],
                [[0.9, 1.0], [0.8, 0.9]],
            ]
        )

        original = conservative_q_penalty(q_data, q_random, agg="min", margin=0.1)
        shifted = conservative_q_penalty(q_data + 37.0, q_random + 37.0, agg="min", margin=0.1)

        torch.testing.assert_close(original, shifted)

    def test_conservative_q_penalty_rejects_negative_margin(self):
        with self.assertRaisesRegex(ValueError, "margin must be a non-negative number"):
            conservative_q_penalty(
                torch.ones(1, 2),
                torch.ones(1, 2, 2),
                agg="min",
                margin=-0.1,
            )

    def test_pairwise_q_ranking_loss_requires_preferred_action_margin(self):
        preferred_q = torch.tensor([[1.0, 1.2], [0.5, 0.6]])
        rejected_q = torch.tensor([[0.8, 0.9], [0.7, 0.8]])

        loss = pairwise_q_ranking_loss(preferred_q, rejected_q, margin=0.1)

        self.assertAlmostEqual(float(loss), 0.15)

    def test_pairwise_q_ranking_loss_is_zero_when_order_is_already_correct(self):
        loss = pairwise_q_ranking_loss(
            torch.tensor([[2.0, 2.0]]),
            torch.tensor([[1.0, 1.0]]),
            margin=0.05,
        )

        self.assertEqual(float(loss), 0.0)

    def test_sample_critic_actions_stays_in_replay_coordinate_system(self):
        actions = torch.tensor([[2.0, -1.5], [1.0, 0.5]])
        torch.manual_seed(0)
        sampled = sample_critic_actions(
            actions,
            4,
            strategy="replay_local_gaussian",
            noise_std=0.05,
        )

        self.assertEqual(sampled.shape, (2, 4, 2))
        self.assertTrue(torch.allclose(sampled.mean(dim=1), actions, atol=0.1))
        self.assertTrue(torch.all(sampled[0, :, 0] > 1.0))

    def test_sample_critic_actions_keeps_unit_uniform_as_explicit_control(self):
        torch.manual_seed(0)
        sampled = sample_critic_actions(
            torch.tensor([[2.0, -1.5]]),
            32,
            strategy="unit_uniform",
            noise_std=0.05,
        )

        self.assertEqual(sampled.shape, (1, 32, 2))
        self.assertLessEqual(float(sampled.max()), 1.0)
        self.assertGreaterEqual(float(sampled.min()), -1.0)

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
        loss = actor_loss(q_pi, log_pi, torch.tensor(0.2), agg="min", entropy_regularization=True)
        self.assertAlmostEqual(float(loss.item()), 0.2 * -1.5 - 2.0)

    def test_actor_loss_accepts_positional_agg(self):
        q_pi = torch.tensor([[2.0, 3.0]])
        log_pi = torch.tensor([[-1.5]])
        loss = actor_loss(q_pi, log_pi, torch.tensor(0.2), "min", entropy_regularization=True)
        self.assertAlmostEqual(float(loss.item()), 0.2 * -1.5 - 2.0)

    def test_actor_loss_adds_trajectory_kl_penalty(self):
        q_pi = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        log_pi = torch.tensor([[-1.5], [-0.5]])
        kl_estimate = torch.tensor([[0.4], [0.2]])
        loss = actor_loss(
            q_pi,
            log_pi,
            torch.tensor(0.2),
            "min",
            kl_estimate,
            0.05,
            entropy_regularization=True,
        )
        expected = (0.2 * log_pi - torch.tensor([[2.0], [4.0]]) + 0.05 * kl_estimate).mean()
        self.assertTrue(torch.allclose(loss, expected))

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
