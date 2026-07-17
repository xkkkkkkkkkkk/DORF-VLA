import unittest

import torch
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from lerobot.rlinf_smolvla_libero.libero_adapter import (
    BatchedChunkRolloutResult,
    ChunkRolloutResult,
    execute_action_chunk,
    execute_batched_action_chunk,
)


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


class VectorOneEnv:
    num_envs = 1

    def __init__(self, reward=1.0, done=False, truncated=False, info=None, next_obs=None):
        self.reward = reward
        self.done = done
        self.truncated = truncated
        self.info = info or {}
        self.next_obs = next_obs or {"states": torch.tensor([[3.0, 4.0]])}
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return self.next_obs, self.reward, self.done, self.truncated, self.info


class VectorTwoEnv(VectorOneEnv):
    num_envs = 2

    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return (
            {"states": torch.tensor([[3.0, 4.0], [5.0, 6.0]])},
            torch.tensor([1.0, 2.0]),
            torch.tensor([False, True]),
            torch.tensor([False, False]),
            {"final_info": {"is_success": torch.tensor([False, True])}},
        )


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

    def test_execute_action_chunk_stores_full_chunk_action_when_only_prefix_runs(self):
        env = DummyEnv(rewards=[1.0, 2.0, 3.0])
        raw_chunk = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])

        result = execute_action_chunk(
            env=env,
            curr_obs=self.make_obs(),
            raw_chunk=raw_chunk,
            gamma=0.5,
            max_chunk_steps=1,
            action_postprocessor=identity_action_postprocessor,
        )

        self.assertEqual(result.transition.horizon, 1)
        self.assertEqual(result.transition.actions.shape, (1, 6))
        torch.testing.assert_close(result.transition.actions, raw_chunk.reshape(1, -1))
        self.assertEqual(len(env.actions), 1)

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


    def test_vector_env_num_envs_one_receives_batched_action(self):
        env = VectorOneEnv()
        raw_chunk = torch.tensor([[[0.1, 0.2]]])

        execute_action_chunk(env=env, curr_obs=self.make_obs(), raw_chunk=raw_chunk, gamma=0.99)

        self.assertEqual(len(env.actions), 1)
        self.assertEqual(tuple(env.actions[0].shape), (1, 2))
        torch.testing.assert_close(env.actions[0], torch.tensor([[0.1, 0.2]]))

    def test_rejects_vector_env_when_chunk_batch_size_does_not_match(self):
        with self.assertRaisesRegex(ValueError, r"num_envs=2.*batch size 1"):
            execute_action_chunk(
                env=VectorTwoEnv(),
                curr_obs=self.make_obs(),
                raw_chunk=torch.zeros(1, 1, 2),
                gamma=0.99,
            )

    def test_batched_chunk_splits_vector_env_transitions(self):
        env = VectorTwoEnv()
        curr_obs = {"states": torch.tensor([[1.0, 2.0], [7.0, 8.0]])}
        raw_chunk = torch.tensor([[[0.1, 0.2]], [[0.3, 0.4]]])

        result = execute_batched_action_chunk(
            env=env,
            curr_obs=curr_obs,
            raw_chunk=raw_chunk,
            gamma=0.9,
            stop_on_success=True,
        )

        self.assertIsInstance(result, BatchedChunkRolloutResult)
        self.assertEqual(len(result.rollouts), 2)
        self.assertEqual(tuple(env.actions[0].shape), (2, 2))
        self.assertEqual(result.rollouts[0].transition.actions.shape, (1, 2))
        self.assertEqual(result.rollouts[1].transition.actions.shape, (1, 2))
        self.assertFalse(result.rollouts[0].success)
        self.assertTrue(result.rollouts[1].success)
        self.assertTrue(result.rollouts[1].transition.done)
        torch.testing.assert_close(result.rollouts[0].transition.curr_obs["states"], torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(result.rollouts[1].transition.curr_obs["states"], torch.tensor([[7.0, 8.0]]))

    def test_vector_env_scalar_step_outputs_are_unwrapped(self):
        scalar_cases = [
            ("list", [1.25], [False], [False]),
            ("tuple", (1.25,), (False,), (False,)),
            ("torch", torch.tensor([1.25]), torch.tensor([False]), torch.tensor([False])),
        ]
        if np is not None:
            scalar_cases.append(("numpy", np.array([1.25]), np.array([False]), np.array([False])))

        for name, reward, done, truncated in scalar_cases:
            with self.subTest(name=name):
                result = execute_action_chunk(
                    env=VectorOneEnv(reward=reward, done=done, truncated=truncated),
                    curr_obs=self.make_obs(),
                    raw_chunk=torch.zeros(1, 1, 2),
                    gamma=0.99,
                )

                self.assertEqual(result.raw_rewards, [1.25])
                self.assertFalse(result.transition.done)
                self.assertFalse(result.truncated)

    def test_vector_env_success_info_paths_are_unwrapped(self):
        success_infos = [
            ("success", {"success": [True]}),
            ("is_success", {"is_success": torch.tensor([True])}),
            ("final_info", {"final_info": {"is_success": torch.tensor([True])}}),
            ("final_info_list", {"final_info": [{"is_success": True}]}),
        ]
        if np is not None:
            success_infos.append(("numpy_success", {"success": np.array([True])}))
            success_infos.append(("final_info_numpy_object", {"final_info": np.array([{"is_success": True}], dtype=object)}))

        for name, info in success_infos:
            with self.subTest(name=name):
                result = execute_action_chunk(
                    env=VectorOneEnv(info=info),
                    curr_obs=self.make_obs(),
                    raw_chunk=torch.zeros(1, 1, 2),
                    gamma=0.99,
                    stop_on_success=True,
                )

                self.assertTrue(result.success)
                self.assertTrue(result.transition.done)

    def test_observation_preparer_can_use_next_obs_and_env(self):
        curr_obs = self.make_obs()
        raw_next_obs = {"raw": torch.tensor([[7.0]])}
        env = VectorOneEnv(next_obs=raw_next_obs)

        def prepare(next_obs, env_arg):
            self.assertIs(next_obs, raw_next_obs)
            self.assertIs(env_arg, env)
            return {"states": next_obs["raw"] + 1.0}

        result = execute_action_chunk(
            env=env,
            curr_obs=curr_obs,
            raw_chunk=torch.zeros(1, 1, 2),
            gamma=0.99,
            observation_preparer=prepare,
        )

        self.assertIs(result.transition.curr_obs, curr_obs)
        torch.testing.assert_close(result.transition.next_obs["states"], torch.tensor([[8.0]]))

    def test_observation_preparer_can_accept_only_next_obs(self):
        raw_next_obs = {"raw": torch.tensor([[7.0]])}

        result = execute_action_chunk(
            env=VectorOneEnv(next_obs=raw_next_obs),
            curr_obs=self.make_obs(),
            raw_chunk=torch.zeros(1, 1, 2),
            gamma=0.99,
            observation_preparer=lambda next_obs: {"states": next_obs["raw"] + 2.0},
        )

        torch.testing.assert_close(result.transition.next_obs["states"], torch.tensor([[9.0]]))


if __name__ == "__main__":
    unittest.main()
