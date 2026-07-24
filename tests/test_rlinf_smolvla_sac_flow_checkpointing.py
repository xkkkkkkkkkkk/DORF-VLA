import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class FakeStateful:
    def __init__(self, state):
        self.state = state
        self.loaded = None

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.loaded = state


class FakePolicy:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = Path(path)


class FakeProcessor:
    def __init__(self):
        self.saved = []

    def save_pretrained(self, path, *, config_filename):
        self.saved.append((Path(path), config_filename))


class CheckpointingTest(unittest.TestCase):
    def test_save_checkpoint_writes_policy_and_sac_state_payload(self):
        saved = {}

        def fake_torch_save(payload, path):
            saved[Path(path).name] = payload

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = FakePolicy()
            preprocessor = FakeProcessor()
            postprocessor = FakeProcessor()
            checkpoint_dir = save_sac_flow_checkpoint(
                output_dir=tmpdir,
                step=7,
                policy=policy,
                policy_preprocessor=preprocessor,
                policy_postprocessor=postprocessor,
                q_network=FakeModule("q"),
                target_q_network=FakeModule("target"),
                temperature=FakeModule("temp"),
                config=SACFlowConfig(device="cuda:0"),
                extra_state={"replay_size": 3},
                torch_save_fn=fake_torch_save,
            )

        self.assertEqual(checkpoint_dir.name, "checkpoint_000007")
        self.assertEqual(policy.saved_to, checkpoint_dir / "policy")
        self.assertEqual(preprocessor.saved, [(checkpoint_dir / "policy", "policy_preprocessor.json")])
        self.assertEqual(postprocessor.saved, [(checkpoint_dir / "policy", "policy_postprocessor.json")])
        payload = saved["sac_flow_state.pt"]
        self.assertEqual(payload["step"], 7)
        self.assertEqual(payload["q_network"], {"q": 1})
        self.assertEqual(payload["target_q_network"], {"target": 1})
        self.assertEqual(payload["temperature"], {"temp": 1})
        self.assertEqual(payload["config"]["device"], "cuda:0")
        self.assertEqual(payload["extra_state"], {"replay_size": 3})

    def test_atomic_save_does_not_publish_a_corrupt_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint_000007"

            def fail_torch_save(payload, path):
                Path(path).write_bytes(b"incomplete")
                raise RuntimeError("disk full")

            with patch("torch.save", side_effect=fail_torch_save):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    save_sac_flow_checkpoint(
                        output_dir=tmpdir,
                        step=7,
                        policy=None,
                        q_network=FakeModule("q"),
                        target_q_network=FakeModule("target"),
                        temperature=FakeModule("temp"),
                        config=SACFlowConfig(device="cpu"),
                    )

            self.assertFalse((checkpoint_dir / "sac_flow_state.pt").exists())
            self.assertEqual(list(checkpoint_dir.glob(".sac_flow_state.pt.tmp-*")), [])

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

    def test_checkpoint_round_trip_restores_trainer_and_replay_state(self):
        saved = {}

        def fake_torch_save(payload, path):
            saved[Path(path).name] = payload

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = save_sac_flow_checkpoint(
                output_dir=tmpdir,
                step=280,
                policy=None,
                q_network=FakeModule("q"),
                target_q_network=FakeModule("target"),
                temperature=FakeModule("temp"),
                trainer=FakeStateful({"update_step": 1116}),
                replay_buffer=FakeStateful({"capacity": 256, "items": ["transition"]}),
                config=SACFlowConfig(device="cpu"),
                rng_state={"python": "state", "torch": "state"},
                torch_save_fn=fake_torch_save,
            )

        trainer = FakeStateful({})
        replay = FakeStateful({})
        payload = saved["sac_flow_state.pt"]
        load_sac_flow_checkpoint(
            checkpoint_dir=checkpoint_dir,
            q_network=FakeModule("q"),
            target_q_network=FakeModule("target"),
            temperature=FakeModule("temp"),
            trainer=trainer,
            replay_buffer=replay,
            restore_rng=False,
            torch_load_fn=lambda path, map_location=None: payload,
        )

        self.assertEqual(trainer.loaded, {"update_step": 1116})
        self.assertEqual(replay.loaded, {"capacity": 256, "items": ["transition"]})

    def test_cuda_preflight_rejects_unavailable_gpu_but_allows_cpu(self):
        assert_sac_flow_device_ready("cpu", cuda_available_fn=lambda: False)
        assert_sac_flow_device_ready("cuda:0", cuda_available_fn=lambda: True)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            assert_sac_flow_device_ready("cuda:0", cuda_available_fn=lambda: False)


if __name__ == "__main__":
    unittest.main()
