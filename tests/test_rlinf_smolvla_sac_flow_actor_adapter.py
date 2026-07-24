import unittest

import torch

from lerobot.rlinf_smolvla_libero.actor_adapter import SmolVLASACFlowActor


class DummyPolicy:
    def __init__(self):
        self.sample_calls = []
        self.encode_calls = []
        self._param = torch.nn.Parameter(torch.ones(()))

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

    def parameters(self):
        return iter([self._param])


class SmolVLASACFlowActorTest(unittest.TestCase):
    def make_obs(self):
        return {"states": torch.zeros(2, 4)}

    def test_sample_chunk_uses_only_replanned_environment_action_for_critic(self):
        actor = SmolVLASACFlowActor(
            policy=DummyPolicy(),
            device=torch.device("cpu"),
            train_noise_std=0.3,
            rollout_noise_std=0.02,
        )
        flat_actions, log_pi, obs_features, raw_chunk = actor.sample_chunk(self.make_obs(), train=True)
        self.assertEqual(raw_chunk.shape, (2, 2, 3))
        self.assertEqual(flat_actions.shape, (2, 3))
        torch.testing.assert_close(flat_actions, raw_chunk[:, 0])
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

    def test_rejects_multi_step_critic_actions(self):
        with self.assertRaisesRegex(ValueError, "critic_action_steps must be 1"):
            SmolVLASACFlowActor(DummyPolicy(), torch.device("cpu"), 0.3, 0.02, critic_action_steps=2)

    def test_sample_chunk_with_kl_normalizes_full_trajectory_likelihood_ratio(self):
        class TrajectoryPolicy(DummyPolicy):
            def sac_sample_action_chunk(
                self, batch, *, train, rollout_noise_std, train_noise_std, return_trajectory=False
            ):
                raw_chunk, log_pi, obs_features = super().sac_sample_action_chunk(
                    batch, train=train, rollout_noise_std=rollout_noise_std, train_noise_std=train_noise_std
                )
                if not return_trajectory:
                    return raw_chunk, log_pi, obs_features
                trajectory = torch.stack((torch.zeros_like(raw_chunk), raw_chunk), dim=0)
                return raw_chunk, log_pi, obs_features, trajectory

        class ReferencePolicy:
            def sac_log_prob_action_trajectory(self, batch, trajectory, *, noise_std):
                self.batch = batch
                self.trajectory = trajectory
                self.noise_std = noise_std
                return torch.zeros(trajectory.shape[1])

        reference = ReferencePolicy()
        actor = SmolVLASACFlowActor(TrajectoryPolicy(), torch.device("cpu"), 0.3, 0.02, reference)
        _, log_pi, _, _, kl_estimate = actor.sample_chunk_with_kl(self.make_obs())

        self.assertTrue(torch.allclose(kl_estimate, log_pi / 12.0))
        self.assertEqual(reference.trajectory.shape, (2, 2, 2, 3))
        self.assertEqual(reference.noise_std, 0.3)

    def test_kl_keeps_reference_parameters_frozen_but_backpropagates_to_live_actor(self):
        class LivePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def sac_sample_action_chunk(
                self, batch, *, train, rollout_noise_std, train_noise_std, return_trajectory=False
            ):
                raw_chunk = self.weight * torch.ones(batch["states"].shape[0], 2, 3)
                log_pi = raw_chunk.sum(dim=(1, 2))
                obs_features = torch.ones(raw_chunk.shape[0], 5)
                if return_trajectory:
                    return raw_chunk, log_pi, obs_features, torch.stack((torch.zeros_like(raw_chunk), raw_chunk))
                return raw_chunk, log_pi, obs_features

            def sac_encode_observation(self, batch):
                return torch.ones(batch["states"].shape[0], 5)

        class FrozenReference(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)

            def sac_log_prob_action_trajectory(self, batch, trajectory, *, noise_std):
                return self.weight * trajectory[-1].sum(dim=(1, 2))

        live = LivePolicy()
        reference = FrozenReference()
        actor = SmolVLASACFlowActor(live, torch.device("cpu"), 0.3, 0.02, reference)
        *_, kl_estimate = actor.sample_chunk_with_kl(self.make_obs())
        kl_estimate.mean().backward()

        self.assertIsNotNone(live.weight.grad)
        self.assertIsNone(reference.weight.grad)

    def test_encode_obs_returns_2d_features(self):
        actor = SmolVLASACFlowActor(DummyPolicy(), torch.device("cpu"), 0.3, 0.02)
        obs_features = actor.encode_obs(self.make_obs())
        self.assertEqual(obs_features.shape, (2, 5))

    def test_parameters_delegate_to_wrapped_policy_for_trainer_optimizers(self):
        policy = DummyPolicy()
        actor = SmolVLASACFlowActor(policy, torch.device("cpu"), 0.3, 0.02)

        self.assertEqual(list(actor.parameters()), [policy._param])

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
