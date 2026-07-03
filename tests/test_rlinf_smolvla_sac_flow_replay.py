import unittest

import torch

from lerobot.rlinf_smolvla_libero.replay import ChunkReplayBuffer, ChunkTransition


class ChunkReplayBufferTest(unittest.TestCase):
    def make_transition(self, value: float) -> ChunkTransition:
        obs = {"states": torch.tensor([[value, value + 1]], dtype=torch.float32)}
        next_obs = {"states": torch.tensor([[value + 2, value + 3]], dtype=torch.float32)}
        return ChunkTransition(
            curr_obs=obs,
            actions=torch.tensor([[value, value + 0.5]], dtype=torch.float32),
            next_obs=next_obs,
            rewards=[1.0],
            done=False,
            horizon=1,
            discount=0.96,
            chunk_reward=1.0,
        )

    def test_sample_returns_batched_transition_tensors(self):
        buffer = ChunkReplayBuffer(capacity=4, seed=123)
        buffer.add(self.make_transition(1.0))
        buffer.add(self.make_transition(2.0))
        batch = buffer.sample(batch_size=2, device=torch.device("cpu"))
        self.assertEqual(
            set(batch.keys()),
            {"curr_obs", "next_obs", "actions", "rewards", "terminations", "discounts", "horizons"},
        )
        self.assertEqual(batch["actions"].shape, (2, 2))
        self.assertEqual(batch["curr_obs"]["states"].shape, (2, 2))
        self.assertEqual(batch["next_obs"]["states"].shape, (2, 2))
        self.assertEqual(batch["rewards"].shape, (2, 1))
        self.assertEqual(batch["terminations"].shape, (2, 1))
        self.assertEqual(batch["terminations"].dtype, torch.bool)
        self.assertEqual(batch["discounts"].shape, (2, 1))
        self.assertEqual(batch["horizons"].shape, (2, 1))
        self.assertEqual(batch["horizons"].dtype, torch.long)

    def test_rejects_invalid_capacity_and_horizon(self):
        with self.assertRaisesRegex(ValueError, "capacity must be positive"):
            ChunkReplayBuffer(capacity=0, seed=123)
        buffer = ChunkReplayBuffer(capacity=2, seed=123)
        bad = self.make_transition(1.0)
        bad.horizon = 0
        with self.assertRaisesRegex(ValueError, "transition horizon must be positive"):
            buffer.add(bad)

    def test_sample_rejects_empty_buffer(self):
        buffer = ChunkReplayBuffer(capacity=2, seed=123)
        with self.assertRaisesRegex(RuntimeError, "empty ChunkReplayBuffer"):
            buffer.sample(batch_size=1, device=torch.device("cpu"))

    def test_sample_rejects_unbatched_actions(self):
        buffer = ChunkReplayBuffer(capacity=2, seed=123)
        bad = self.make_transition(1.0)
        bad.actions = torch.tensor([1.0, 2.0], dtype=torch.float32)
        buffer.add(bad)

        with self.assertRaisesRegex(
            ValueError,
            r"transition\.actions must have shape \[1, action_dim\] or \[batch, action_dim\]",
        ):
            buffer.sample(batch_size=1, device=torch.device("cpu"))

    def test_sample_filters_non_tensor_observation_metadata(self):
        buffer = ChunkReplayBuffer(capacity=2, seed=123)
        transition = self.make_transition(1.0)
        transition.curr_obs.update(
            {
                "action": None,
                "next.reward": 1.0,
                "info": {"success": False},
                "task": ["pick up the object"],
            }
        )
        transition.next_obs.update(
            {
                "action": None,
                "next.reward": 0.0,
                "info": {"success": True},
                "task": ["pick up the object"],
            }
        )
        buffer.add(transition)

        batch = buffer.sample(batch_size=1, device=torch.device("cpu"))

        self.assertEqual(set(batch["curr_obs"].keys()), {"states"})
        self.assertEqual(set(batch["next_obs"].keys()), {"states"})
        self.assertEqual(batch["curr_obs"]["states"].shape, (1, 2))

    def test_sample_rejects_mismatched_observation_keys(self):
        buffer = ChunkReplayBuffer(capacity=4, seed=123)
        first = self.make_transition(1.0)
        second = self.make_transition(2.0)
        second.curr_obs["extra"] = torch.tensor([[42.0]], dtype=torch.float32)
        buffer.add(first)
        buffer.add(second)

        with self.assertRaisesRegex(
            ValueError,
            "observation keys must match across sampled transitions",
        ):
            buffer.sample(batch_size=2, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
