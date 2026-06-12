import unittest

from lerobot.scripts.smolvla_sac_flow_utils import (
    ChunkDecisionStats,
    chunk_discount_factor,
    chunk_reward_sum,
    flatten_action_chunk,
    unflatten_action_chunk,
)


class SmolVLASACFlowUtilsTest(unittest.TestCase):
    def test_flatten_and_unflatten_action_chunk_roundtrip(self):
        action_chunk = [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        ]

        flat = flatten_action_chunk(action_chunk)
        restored = unflatten_action_chunk(flat, chunk_size=3, action_dim=4)

        self.assertEqual(flat[0], list(range(12)))
        self.assertEqual(restored, action_chunk)

    def test_chunk_reward_sum_and_discount_factor_follow_smdp_semantics(self):
        rewards = [1.0, 2.0, 3.0]
        gamma = 0.5

        reward_sum = chunk_reward_sum(rewards, gamma=gamma)
        discount = chunk_discount_factor(num_steps=3, gamma=gamma)

        self.assertAlmostEqual(reward_sum, 1.0 + 0.5 * 2.0 + 0.25 * 3.0)
        self.assertAlmostEqual(discount, 0.125)

    def test_chunk_decision_stats_counts_discounted_reward_and_horizon(self):
        stats = ChunkDecisionStats()
        stats.add_step(reward=1.0)
        stats.add_step(reward=2.0)
        stats.add_step(reward=3.0, terminal=True)

        self.assertEqual(stats.horizon, 3)
        self.assertTrue(stats.terminal)
        self.assertEqual(stats.raw_reward_sum, 6.0)


if __name__ == "__main__":
    unittest.main()
