import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 本地轻量环境可能没有安装 torch。
    torch = None


def _load_modeling_smolvla():
    sentinel = object()
    module_names = [
        "lerobot.policies",
        "lerobot.policies.pretrained",
        "lerobot.policies.rtc",
        "lerobot.policies.rtc.modeling_rtc",
        "lerobot.policies.smolvla",
        "lerobot.policies.smolvla.configuration_smolvla",
        "lerobot.policies.smolvla.smolvlm_with_expert",
        "lerobot.policies.utils",
        "lerobot.utils",
        "lerobot.utils.constants",
        "lerobot.utils.utils",
    ]
    originals = {name: sys.modules.get(name, sentinel) for name in module_names}

    def package(name):
        module = types.ModuleType(name)
        module.__path__ = []
        return module

    policies = package("lerobot.policies")
    pretrained = types.ModuleType("lerobot.policies.pretrained")
    pretrained.PreTrainedPolicy = torch.nn.Module

    rtc = package("lerobot.policies.rtc")
    rtc_modeling = types.ModuleType("lerobot.policies.rtc.modeling_rtc")
    rtc_modeling.RTCProcessor = object

    smolvla_pkg = package("lerobot.policies.smolvla")
    config = types.ModuleType("lerobot.policies.smolvla.configuration_smolvla")
    config.SmolVLAConfig = object
    smolvlm = types.ModuleType("lerobot.policies.smolvla.smolvlm_with_expert")
    smolvlm.SmolVLMWithExpertModel = object

    policy_utils = types.ModuleType("lerobot.policies.utils")
    policy_utils.populate_queues = lambda queues, batch, exclude_keys=None: queues

    utils_pkg = package("lerobot.utils")
    constants = types.ModuleType("lerobot.utils.constants")
    constants.ACTION = "action"
    constants.OBS_LANGUAGE_ATTENTION_MASK = "observation.language_attention_mask"
    constants.OBS_LANGUAGE_TOKENS = "observation.language_tokens"
    constants.OBS_STATE = "observation.state"
    utils = types.ModuleType("lerobot.utils.utils")
    utils.get_safe_dtype = lambda dtype, device_type: dtype

    sys.modules.update(
        {
            "lerobot.policies": policies,
            "lerobot.policies.pretrained": pretrained,
            "lerobot.policies.rtc": rtc,
            "lerobot.policies.rtc.modeling_rtc": rtc_modeling,
            "lerobot.policies.smolvla": smolvla_pkg,
            "lerobot.policies.smolvla.configuration_smolvla": config,
            "lerobot.policies.smolvla.smolvlm_with_expert": smolvlm,
            "lerobot.policies.utils": policy_utils,
            "lerobot.utils": utils_pkg,
            "lerobot.utils.constants": constants,
            "lerobot.utils.utils": utils,
        }
    )

    try:
        module_path = (
            Path(__file__).resolve().parents[1]
            / "lerobot"
            / "policies"
            / "smolvla"
            / "modeling_smolvla.py"
        )
        spec = importlib.util.spec_from_file_location("_modeling_smolvla_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@unittest.skipIf(torch is None, "torch is required for SmolVLA SAC interface contract tests")
class SmolVLASACInterfaceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smolvla = _load_modeling_smolvla()

    def _policy(self, *, model, adapt_to_pi_aloha=False, action_dim=4, max_action_dim=4):
        policy = self.smolvla.SmolVLAPolicy.__new__(self.smolvla.SmolVLAPolicy)
        policy.config = SimpleNamespace(
            action_feature=SimpleNamespace(shape=(action_dim,)),
            adapt_to_pi_aloha=adapt_to_pi_aloha,
            chunk_size=3,
            max_action_dim=max_action_dim,
        )
        policy.model = model
        policy.prepare_images = lambda batch: ("images", "img_masks")
        policy.prepare_state = lambda batch: batch[self.smolvla.OBS_STATE]
        return policy

    def _batch(self, batch_size=2):
        return {
            self.smolvla.OBS_STATE: torch.zeros(batch_size, 6),
            self.smolvla.OBS_LANGUAGE_TOKENS: torch.zeros(batch_size, 5, dtype=torch.long),
            self.smolvla.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 5, dtype=torch.bool),
        }

    def test_sac_sample_action_chunk_rejects_log_prob_batch_mismatch(self):
        class FakeModel:
            def sample_actions_with_log_prob(self, *args, **kwargs):
                return torch.zeros(2, 3, 4), torch.zeros(3), torch.zeros(2, 8)

        policy = self._policy(model=FakeModel())

        with self.assertRaisesRegex(RuntimeError, "log_prob.*batch"):
            policy.sac_sample_action_chunk(
                self._batch(batch_size=2),
                train=False,
                rollout_noise_std=0.0,
                train_noise_std=0.0,
            )

    def test_sac_sample_action_chunk_rejects_obs_features_batch_mismatch(self):
        class FakeModel:
            def sample_actions_with_log_prob(self, *args, **kwargs):
                return torch.zeros(2, 3, 4), torch.zeros(2), torch.zeros(3, 8)

        policy = self._policy(model=FakeModel())

        with self.assertRaisesRegex(RuntimeError, "obs_features.*batch"):
            policy.sac_sample_action_chunk(
                self._batch(batch_size=2),
                train=False,
                rollout_noise_std=0.0,
                train_noise_std=0.0,
            )

    def test_sample_action_chunk_with_log_prob_forwards_noise_and_training_knobs(self):
        calls = []
        policy = self.smolvla.SmolVLAPolicy.__new__(self.smolvla.SmolVLAPolicy)

        def fake_sac_sample_action_chunk(batch, noise=None, *, train, rollout_noise_std, train_noise_std):
            calls.append((batch, noise, train, rollout_noise_std, train_noise_std))
            return "actions", "log_prob", "obs_features"

        policy.sac_sample_action_chunk = fake_sac_sample_action_chunk
        batch = {"sentinel": object()}
        noise = object()

        result = policy.sample_action_chunk_with_log_prob(
            batch,
            noise=noise,
            train=True,
            rollout_noise_std=0.12,
            train_noise_std=0.34,
        )

        self.assertEqual(result, ("actions", "log_prob", "obs_features"))
        self.assertEqual(calls, [(batch, noise, True, 0.12, 0.34)])

    def test_sac_sample_action_chunk_returns_unpadded_env_action_dim(self):
        class FakeModel:
            def sample_actions_with_log_prob(self, *args, **kwargs):
                return torch.zeros(2, 3, 4), torch.zeros(2), torch.zeros(2, 8)

        policy = self._policy(model=FakeModel(), action_dim=2, max_action_dim=4)

        actions, _, _ = policy.sac_sample_action_chunk(
            self._batch(batch_size=2),
            train=False,
            rollout_noise_std=0.0,
            train_noise_std=0.0,
        )

        self.assertEqual(actions.shape, (2, 3, 2))

    def test_flow_sample_actions_with_log_prob_rejects_bad_noise_shape_before_prefix_encode(self):
        flow = self.smolvla.VLAFlowMatching.__new__(self.smolvla.VLAFlowMatching)
        flow.config = SimpleNamespace(chunk_size=3, max_action_dim=4, num_steps=1)
        flow._build_prefix_context = lambda *args, **kwargs: self.fail("noise shape should be checked first")

        with self.assertRaisesRegex(RuntimeError, "noise shape"):
            flow.sample_actions_with_log_prob(
                images=None,
                img_masks=None,
                lang_tokens=None,
                lang_masks=None,
                state=torch.zeros(2, 6),
                noise=torch.zeros(2, 3, 5),
                train=False,
                rollout_noise_std=0.0,
                train_noise_std=0.0,
            )

    def test_flow_trajectory_scoring_matches_sampling_log_prob(self):
        flow = self.smolvla.VLAFlowMatching.__new__(self.smolvla.VLAFlowMatching)
        flow.config = SimpleNamespace(chunk_size=2, max_action_dim=3, num_steps=2)
        flow.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
        flow._build_prefix_context = lambda *args, **kwargs: {
            "prefix_pad_masks": None,
            "past_key_values": None,
            "obs_features": torch.ones(2, 5),
        }
        flow.denoise_step = lambda x_t, **kwargs: torch.full_like(x_t, 0.25)
        state = torch.zeros(2, 6)

        _, sampled_log_prob, _, trajectory = flow.sample_actions_with_log_prob(
            images=None,
            img_masks=None,
            lang_tokens=None,
            lang_masks=None,
            state=state,
            train=True,
            rollout_noise_std=0.0,
            train_noise_std=0.2,
            return_trajectory=True,
        )
        scored_log_prob = flow.log_prob_action_trajectory(
            None,
            None,
            None,
            None,
            state,
            trajectory,
            noise_std=0.2,
        )

        self.assertEqual(trajectory.shape, (3, 2, 2, 3))
        self.assertTrue(torch.allclose(sampled_log_prob, scored_log_prob))

    def test_flow_trajectory_scoring_requires_positive_noise_std(self):
        flow = self.smolvla.VLAFlowMatching.__new__(self.smolvla.VLAFlowMatching)
        flow.config = SimpleNamespace(chunk_size=2, max_action_dim=3, num_steps=2)
        with self.assertRaisesRegex(ValueError, "noise_std must be positive"):
            flow.log_prob_action_trajectory(
                None,
                None,
                None,
                None,
                torch.zeros(1, 6),
                torch.zeros(3, 1, 2, 3),
                noise_std=0.0,
            )

    def test_prepare_batch_does_not_mutate_input_when_decoding_pi_aloha_state(self):
        policy = self._policy(model=None, adapt_to_pi_aloha=True)
        policy._pi_aloha_decode_state = lambda state: state + 1.0
        batch = self._batch(batch_size=2)
        original_state = batch[self.smolvla.OBS_STATE].clone()

        prepared = policy._prepare_batch(batch)

        self.assertIsNot(prepared, batch)
        self.assertTrue(torch.equal(batch[self.smolvla.OBS_STATE], original_state))
        self.assertTrue(torch.equal(prepared[self.smolvla.OBS_STATE], original_state + 1.0))


if __name__ == "__main__":
    unittest.main()
