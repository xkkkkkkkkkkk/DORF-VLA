import unittest

import torch

from lerobot.rlinf_smolvla_libero.libero_adapter import ChunkRolloutResult, execute_action_chunk


class DummyEnv:
    def __init__(self, rewards, dones=None, truncateds=None, successes=None):
        self.rewards = list(rewards)
        self.dones = list(dones or [False] * len(rewards))
        self.truncateds = list(truncateds or [False] * len(rewards))
        self.successes = list(successes or [False] * len(rewards))
        self.actions = []
        self.step_count = 0

    def step(self, action):
        self.actions.append(action)
        idx = self.step_count
        self.step_count += 1
        next_obs = {"states": torch.tensor([[10.0 + idx, 20.0 + idx]])}
        info = {"success": self.successes[idx]}
        return next_obs, self.rewards[idx], self.dones[idx], self.truncateds[idx], info


class BadArityEnv:
    def step(self, action):
        return {"states": torch.tensor([[0.0, 0.0]])}, 1.0, False, {}


def identity_action_postprocessor(action):
    return action


class LiberoAdapterTest(unittest.TestCase):
    def make_obs(self):
        return {"states": torch.tensor([[1.0, 2.0]])}

    def test_execute_action_chunk_accumulates_transition(self):
        env = DummyEnv(rewards=[1.0, 2.0, 3.0])
        raw_chunk = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
        result = execute_action_chunk(
            env=env,
            curr_obs=self.make_obs(),
            raw_chunk=raw_chunk,
            gamma=0.5,
            max_chunk_steps=3,
            action_postprocessor=identity_action_postprocessor,
        )
        self.assertIsInstance(result, ChunkRolloutResult)
        self.assertEqual(result.transition.horizon, 3)
        self.assertEqual(result.transition.chunk_reward, 1.0 + 0.5 * 2.0 + 0.25 * 3.0)
        self.assertEqual(result.transition.discount, 0.125)
        self.assertFalse(result.transition.done)
        self.assertEqual(result.transition.actions.shape, (1, 6))
        self.assertEqual(result.raw_rewards, [1.0, 2.0, 3.0])
        self.assertEqual(len(env.actions), 3)

    def test_execute_action_chunk_stops_on_done(self):
        env = DummyEnv(rewards=[1.0, 2.0, 3.0], dones=[False, True, False])
        raw_chunk = torch.zeros(1, 3, 2)
        result = execute_action_chunk(
            env=env,
            curr_obs=self.make_obs(),
            raw_chunk=raw_chunk,
            gamma=0.9,
            max_chunk_steps=3,
        )
        self.assertEqual(result.transition.horizon, 2)
        self.assertTrue(result.transition.done)
        self.assertEqual(result.raw_rewards, [1.0, 2.0])
        self.assertEqual(len(env.actions), 2)

    def test_execute_action_chunk_stops_on_truncated_without_terminal_done(self):
        env = DummyEnv(rewards=[1.0, 2.0, 3.0], truncateds=[False, True, False])
        raw_chunk = torch.zeros(1, 3, 2)
        result = execute_action_chunk(
            env=env,
            curr_obs=self.make_obs(),
            raw_chunk=raw_chunk,
            gamma=0.9,
            max_chunk_steps=3,
        )
        self.assertEqual(result.transition.horizon, 2)
        self.assertTrue(result.truncated)
        self.assertFalse(result.transition.done)
        self.assertEqual(len(env.actions), 2)

    def test_execute_action_chunk_stops_on_success_info(self):
        env = DummyEnv(rewards=[0.0, 1.0, 1.0], successes=[False, True, False])
        raw_chunk = torch.zeros(1, 3, 2)
        result = execute_action_chunk(
            env=env,
            curr_obs=self.make_obs(),
            raw_chunk=raw_chunk,
            gamma=0.99,
            max_chunk_steps=3,
            stop_on_success=True,
        )
        self.assertEqual(result.transition.horizon, 2)
        self.assertTrue(result.success)
        self.assertTrue(result.transition.done)

    def test_rejects_non_positive_max_chunk_steps(self):
        raw_chunk = torch.zeros(1, 3, 2)
        for max_chunk_steps in (0, -1):
            with self.subTest(max_chunk_steps=max_chunk_steps):
                with self.assertRaisesRegex(ValueError, "max_chunk_steps must be positive"):
                    execute_action_chunk(
                        env=DummyEnv([1.0]),
                        curr_obs=self.make_obs(),
                        raw_chunk=raw_chunk,
                        gamma=0.99,
                        max_chunk_steps=max_chunk_steps,
                    )

    def test_rejects_bad_env_step_return_arity(self):
        with self.assertRaisesRegex(ValueError, r"env.step\(action\) must return"):
            execute_action_chunk(
                env=BadArityEnv(),
                curr_obs=self.make_obs(),
                raw_chunk=torch.zeros(1, 1, 2),
                gamma=0.99,
            )

    def test_rejects_bad_chunk_shape(self):
        with self.assertRaisesRegex(ValueError, "raw_chunk must have shape"):
            execute_action_chunk(env=DummyEnv([1.0]), curr_obs=self.make_obs(), raw_chunk=torch.zeros(3, 2), gamma=0.99)


if __name__ == "__main__":
    unittest.main()
