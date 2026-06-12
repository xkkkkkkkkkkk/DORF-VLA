import math
import unittest

from lerobot.scripts.smolvla_fm_rl_utils import (
    compute_group_advantages,
    compute_rl_fm_weights,
)


class SmolVLAFMRLUtilsTest(unittest.TestCase):
    def test_compute_group_advantages_matches_grpo_group_normalization(self):
        returns = [1.0, 3.0, 2.0, 2.0]

        advantages = compute_group_advantages(returns, group_size=2)

        self.assertAlmostEqual(advantages[0], -1.0 / math.sqrt(2.0), places=5)
        self.assertAlmostEqual(advantages[1], 1.0 / math.sqrt(2.0), places=5)
        self.assertAlmostEqual(advantages[2], 0.0, places=5)
        self.assertAlmostEqual(advantages[3], 0.0, places=5)

    def test_compute_rl_fm_weights_exponentiates_and_clips_advantages(self):
        advantages = [-10.0, 0.0, 10.0]

        weights = compute_rl_fm_weights(advantages, beta=1.0, min_weight=0.1, max_weight=3.0)

        self.assertEqual(weights, [0.1, 1.0, 3.0])

    def test_group_advantages_rejects_non_multiple_group_size(self):
        with self.assertRaises(ValueError):
            compute_group_advantages([1.0, 2.0, 3.0], group_size=2)

    def test_singleton_group_advantage_is_zero(self):
        advantages = compute_group_advantages([4.0], group_size=1)

        self.assertEqual(advantages, [0.0])


if __name__ == "__main__":
    unittest.main()
