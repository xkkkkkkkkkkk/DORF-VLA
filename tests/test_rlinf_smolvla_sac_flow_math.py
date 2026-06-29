import unittest

from lerobot.rlinf_smolvla_libero.replay import chunk_discount, discounted_chunk_reward, flatten_chunk


class SACFlowMathTest(unittest.TestCase):
    def test_discounted_chunk_reward_uses_stepwise_gamma(self):
        self.assertAlmostEqual(discounted_chunk_reward([1.0, 2.0, 3.0], gamma=0.5), 2.75)

    def test_chunk_discount_uses_actual_executed_horizon(self):
        self.assertAlmostEqual(chunk_discount(horizon=3, gamma=0.5), 0.125)

    def test_flatten_chunk_preserves_batch_dimension(self):
        flat = flatten_chunk([[[1.0, 2.0], [3.0, 4.0]]])
        self.assertEqual(flat, [[1.0, 2.0, 3.0, 4.0]])

    def test_chunk_discount_rejects_negative_horizon(self):
        with self.assertRaisesRegex(ValueError, "horizon must be non-negative"):
            chunk_discount(horizon=-1, gamma=0.99)

    def test_flatten_chunk_rejects_2d_list(self):
        with self.assertRaisesRegex(ValueError, "Expected action chunk shape \\[batch, chunk, action_dim\\]"):
            flatten_chunk([[1.0, 2.0]])

    def test_flatten_chunk_rejects_ragged_action_dim(self):
        with self.assertRaisesRegex(ValueError, "Expected action chunk shape \\[batch, chunk, action_dim\\]"):
            flatten_chunk([[[1.0], [2.0, 3.0]]])

    def test_discounted_chunk_reward_rejects_none_rewards(self):
        with self.assertRaisesRegex(ValueError, "rewards must be a sequence"):
            discounted_chunk_reward(None, gamma=0.99)

    def test_discounted_chunk_reward_rejects_non_iterable_rewards(self):
        with self.assertRaisesRegex(ValueError, "rewards must be a sequence"):
            discounted_chunk_reward(1.0, gamma=0.99)


if __name__ == "__main__":
    unittest.main()
