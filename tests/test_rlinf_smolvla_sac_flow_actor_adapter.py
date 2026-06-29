import unittest

import torch

from lerobot.rlinf_smolvla_libero.actor_adapter import SmolVLASACFlowActor


class DummyPolicy:
    def __init__(self):
        self.sample_calls = []
        self.encode_calls = []

    def sac_sample_action_chunk(self, batch, *, train, rollout_noise_std, train_noise_std):
        self.sample_calls.append((batch, train, rollout_noise_std, train_noise_std))
        batch_size = batch["states"].shape[0]
        raw_chunk = torch.arange(batch_size * 2 * 3, dtype=torch.float32).reshape(batch_size, 2, 3)
        log_pi = torch.arange(batch_size, dtype=torch.float32)
        obs_features = torch.ones(batch_size, 5)
        return raw_chunk, log_pi, obs_features

    def sac_encode_observation(self, batch):
        self.encode_calls.append(batch)
        return torch.ones(batch["states"].shape[0], 5)


class SmolVLASACFlowActorTest(unittest.TestCase):
    def make_obs(self):
        return {"states": torch.zeros(2, 4)}

    def test_sample_chunk_flattens_actions_and_formats_log_pi(self):
        actor = SmolVLASACFlowActor(
            policy=DummyPolicy(),
            device=torch.device("cpu"),
            train_noise_std=0.3,
            rollout_noise_std=0.02,
        )
        flat_actions, log_pi, obs_features, raw_chunk = actor.sample_chunk(self.make_obs(), train=True)
        self.assertEqual(raw_chunk.shape, (2, 2, 3))
        self.assertEqual(flat_actions.shape, (2, 6))
        self.assertEqual(log_pi.shape, (2, 1))
        self.assertEqual(obs_features.shape, (2, 5))
        self.assertEqual(actor.policy.sample_calls[0][1:], (True, 0.02, 0.3))

    def test_sample_chunk_keeps_column_log_pi(self):
        class ColumnPolicy(DummyPolicy):
            def sac_sample_action_chunk(self, batch, *, train, rollout_noise_std, train_noise_std):
                raw_chunk, _, obs_features = super().sac_sample_action_chunk(
                    batch, train=train, rollout_noise_std=rollout_noise_std, train_noise_std=train_noise_std
                )
                return raw_chunk, torch.zeros(batch["states"].shape[0], 1), obs_features

        actor = SmolVLASACFlowActor(ColumnPolicy(), torch.device("cpu"), 0.3, 0.02)
        _, log_pi, _, _ = actor.sample_chunk(self.make_obs(), train=False)
        self.assertEqual(log_pi.shape, (2, 1))
        self.assertEqual(actor.policy.sample_calls[0][1:], (False, 0.02, 0.3))

    def test_encode_obs_returns_2d_features(self):
        actor = SmolVLASACFlowActor(DummyPolicy(), torch.device("cpu"), 0.3, 0.02)
        obs_features = actor.encode_obs(self.make_obs())
        self.assertEqual(obs_features.shape, (2, 5))

    def test_constructor_requires_policy_methods(self):
        with self.assertRaisesRegex(AttributeError, "sac_sample_action_chunk"):
            SmolVLASACFlowActor(object(), torch.device("cpu"), 0.3, 0.02)

    def test_constructor_requires_encode_method(self):
        class SampleOnlyPolicy:
            def sac_sample_action_chunk(self, batch, *, train, rollout_noise_std, train_noise_std):
                return None, None, None

        with self.assertRaisesRegex(AttributeError, "sac_encode_observation"):
            SmolVLASACFlowActor(SampleOnlyPolicy(), torch.device("cpu"), 0.3, 0.02)

    def test_encode_obs_rejects_non_tensor_features(self):
        class FakeFeatures:
            ndim = 2
            shape = (2, 5)

        class BadEncodePolicy(DummyPolicy):
            def sac_encode_observation(self, batch):
                return FakeFeatures()

        actor = SmolVLASACFlowActor(BadEncodePolicy(), torch.device("cpu"), 0.3, 0.02)
        with self.assertRaisesRegex(ValueError, "obs_features must be a tensor"):
            actor.encode_obs(self.make_obs())

    def test_sample_chunk_rejects_wrong_policy_return_arity(self):
        class BadArityPolicy(DummyPolicy):
            def sac_sample_action_chunk(self, batch, *, train, rollout_noise_std, train_noise_std):
                raw_chunk, log_pi, _ = super().sac_sample_action_chunk(
                    batch, train=train, rollout_noise_std=rollout_noise_std, train_noise_std=train_noise_std
                )
                return raw_chunk, log_pi

        actor = SmolVLASACFlowActor(BadArityPolicy(), torch.device("cpu"), 0.3, 0.02)
        with self.assertRaisesRegex(
            ValueError,
            r"policy\.sac_sample_action_chunk must return \(raw_chunk, log_pi, obs_features\)",
        ):
            actor.sample_chunk(self.make_obs(), train=True)

    def test_sample_chunk_rejects_empty_raw_chunk_dimensions(self):
        class EmptyRawChunkPolicy(DummyPolicy):
            def sac_sample_action_chunk(self, batch, *, train, rollout_noise_std, train_noise_std):
                batch_size = batch["states"].shape[0]
                return torch.zeros(batch_size, 0, 3), torch.zeros(batch_size), torch.ones(batch_size, 5)

        actor = SmolVLASACFlowActor(EmptyRawChunkPolicy(), torch.device("cpu"), 0.3, 0.02)
        with self.assertRaisesRegex(
            ValueError,
            r"raw_chunk must have non-empty shape \[batch, chunk, action_dim\].*got shape=",
        ):
            actor.sample_chunk(self.make_obs(), train=True)

    def test_sample_chunk_rejects_obs_feature_batch_mismatch(self):
        class BadObsFeatureBatchPolicy(DummyPolicy):
            def sac_sample_action_chunk(self, batch, *, train, rollout_noise_std, train_noise_std):
                raw_chunk, log_pi, _ = super().sac_sample_action_chunk(
                    batch, train=train, rollout_noise_std=rollout_noise_std, train_noise_std=train_noise_std
                )
                return raw_chunk, log_pi, torch.ones(batch["states"].shape[0] + 1, 5)

        actor = SmolVLASACFlowActor(BadObsFeatureBatchPolicy(), torch.device("cpu"), 0.3, 0.02)
        with self.assertRaisesRegex(ValueError, "obs_features batch size must match actions/log_pi batch size"):
            actor.sample_chunk(self.make_obs(), train=True)

    def test_sample_chunk_rejects_bad_log_pi_shape(self):
        class BadPolicy(DummyPolicy):
            def sac_sample_action_chunk(self, batch, *, train, rollout_noise_std, train_noise_std):
                raw_chunk, _, obs_features = super().sac_sample_action_chunk(
                    batch, train=train, rollout_noise_std=rollout_noise_std, train_noise_std=train_noise_std
                )
                return raw_chunk, torch.zeros(batch["states"].shape[0], 1, 1), obs_features

        actor = SmolVLASACFlowActor(BadPolicy(), torch.device("cpu"), 0.3, 0.02)
        with self.assertRaisesRegex(ValueError, "log_pi must have shape"):
            actor.sample_chunk(self.make_obs(), train=True)


if __name__ == "__main__":
    unittest.main()
