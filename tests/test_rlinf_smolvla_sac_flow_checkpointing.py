import tempfile
import unittest
from pathlib import Path

from lerobot.rlinf_smolvla_libero.checkpointing import (
    assert_sac_flow_device_ready,
    load_sac_flow_checkpoint,
    save_sac_flow_checkpoint,
)
from lerobot.rlinf_smolvla_libero.config import SACFlowConfig


class FakeModule:
    def __init__(self, name):
        self.name = name
        self.loaded = None

    def state_dict(self):
        return {self.name: 1}

    def load_state_dict(self, state):
        self.loaded = state


class FakePolicy:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = Path(path)


class CheckpointingTest(unittest.TestCase):
    def test_save_checkpoint_writes_policy_and_sac_state_payload(self):
        saved = {}

        def fake_torch_save(payload, path):
            saved[Path(path).name] = payload

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = FakePolicy()
            checkpoint_dir = save_sac_flow_checkpoint(
                output_dir=tmpdir,
                step=7,
                policy=policy,
                q_network=FakeModule("q"),
                target_q_network=FakeModule("target"),
                temperature=FakeModule("temp"),
                config=SACFlowConfig(device="cuda:0"),
                extra_state={"replay_size": 3},
                torch_save_fn=fake_torch_save,
            )

        self.assertEqual(checkpoint_dir.name, "checkpoint_000007")
        self.assertEqual(policy.saved_to, checkpoint_dir / "policy")
        payload = saved["sac_flow_state.pt"]
        self.assertEqual(payload["step"], 7)
        self.assertEqual(payload["q_network"], {"q": 1})
        self.assertEqual(payload["target_q_network"], {"target": 1})
        self.assertEqual(payload["temperature"], {"temp": 1})
        self.assertEqual(payload["config"]["device"], "cuda:0")
        self.assertEqual(payload["extra_state"], {"replay_size": 3})

    def test_load_checkpoint_restores_trainable_sac_state(self):
        q = FakeModule("q")
        target = FakeModule("target")
        temp = FakeModule("temp")
        payload = {
            "step": 11,
            "q_network": {"q": 2},
            "target_q_network": {"target": 3},
            "temperature": {"temp": 4},
            "config": {"gamma": 0.96},
            "extra_state": {"x": 1},
        }

        restored = load_sac_flow_checkpoint(
            checkpoint_dir=Path("/tmp/checkpoint_000011"),
            q_network=q,
            target_q_network=target,
            temperature=temp,
            torch_load_fn=lambda path, map_location=None: payload,
        )

        self.assertIs(restored, payload)
        self.assertEqual(q.loaded, {"q": 2})
        self.assertEqual(target.loaded, {"target": 3})
        self.assertEqual(temp.loaded, {"temp": 4})

    def test_cuda_preflight_rejects_unavailable_gpu_but_allows_cpu(self):
        assert_sac_flow_device_ready("cpu", cuda_available_fn=lambda: False)
        assert_sac_flow_device_ready("cuda:0", cuda_available_fn=lambda: True)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            assert_sac_flow_device_ready("cuda:0", cuda_available_fn=lambda: False)


if __name__ == "__main__":
    unittest.main()
