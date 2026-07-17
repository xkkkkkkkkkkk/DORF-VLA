import unittest
from types import SimpleNamespace

import torch

from lerobot.rlinf_smolvla_libero.config import SACFlowConfig
from lerobot.rlinf_smolvla_libero.replay import ChunkTransition
from lerobot.rlinf_smolvla_libero.training_loop import SACFlowOnlineLoop


class FakeReplay:
    def __init__(self):
        self.items = []
        self.sample_calls = []

    def __len__(self):
        return len(self.items)

    def add(self, transition):
        self.items.append(transition)

    def sample(self, *, batch_size, device):
        self.sample_calls.append((batch_size, device))
        return {"batch": len(self.sample_calls)}


class FakeActor:
    def __init__(self):
        self.calls = []

    def sample_chunk(self, obs, train):
        self.calls.append((obs, train))
        return "flat", "log_pi", "features", "raw_chunk"


class FakeTrainer:
    def __init__(self):
        self.batches = []

    def update_sac(self, batch):
        self.batches.append(batch)
        return {"critic_loss": float(batch["batch"])}


def make_transition(curr_obs, next_obs=None):
    return ChunkTransition(
        curr_obs=curr_obs,
        actions="actions",
        next_obs=next_obs or {"states": "next"},
        rewards=[1.0],
        done=False,
        horizon=1,
        discount=0.96,
        chunk_reward=1.0,
    )


class SACFlowOnlineLoopTest(unittest.TestCase):
    def test_collects_rollout_transition_into_replay(self):
        actor = FakeActor()
        replay = FakeReplay()
        rollout_calls = []

        def rollout_fn(**kwargs):
            rollout_calls.append(kwargs)
            return SimpleNamespace(
                transition=make_transition(kwargs["curr_obs"]),
                raw_rewards=[1.0],
                success=False,
                truncated=False,
            )

        loop = SACFlowOnlineLoop(
            actor=actor,
            env="env",
            replay_buffer=replay,
            trainer=FakeTrainer(),
            config=SACFlowConfig(gamma=0.96, min_buffer_size=10, num_updates_per_step=2),
            max_chunk_steps=1,
            rollout_fn=rollout_fn,
        )

        result = loop.collect_transition({"states": "obs"})

        self.assertEqual(actor.calls, [({"states": "obs"}, False)])
        self.assertEqual(len(replay.items), 1)
        self.assertIs(result.transition, replay.items[0])
        self.assertEqual(rollout_calls[0]["raw_chunk"], "raw_chunk")
        self.assertEqual(rollout_calls[0]["gamma"], 0.96)
        self.assertEqual(rollout_calls[0]["max_chunk_steps"], 1)

    def test_updates_only_after_min_buffer_size(self):
        replay = FakeReplay()
        trainer = FakeTrainer()
        loop = SACFlowOnlineLoop(
            actor=FakeActor(),
            env="env",
            replay_buffer=replay,
            trainer=trainer,
            config=SACFlowConfig(min_buffer_size=2, num_updates_per_step=3, batch_size=4, device="cuda:0"),
            rollout_fn=lambda **kwargs: None,
        )

        self.assertEqual(loop.update_if_ready(), [])
        replay.add(make_transition({"states": "a"}))
        self.assertEqual(loop.update_if_ready(), [])
        replay.add(make_transition({"states": "b"}))

        metrics = loop.update_if_ready()

        self.assertEqual(len(metrics), 3)
        self.assertEqual(replay.sample_calls, [(4, "cuda:0"), (4, "cuda:0"), (4, "cuda:0")])
        self.assertEqual(trainer.batches, [{"batch": 1}, {"batch": 2}, {"batch": 3}])

    def test_train_step_returns_next_obs_and_update_metrics(self):
        replay = FakeReplay()
        trainer = FakeTrainer()

        def rollout_fn(**kwargs):
            return SimpleNamespace(
                transition=make_transition(kwargs["curr_obs"], next_obs={"states": "next_obs"}),
                raw_rewards=[2.0],
                success=True,
                truncated=False,
            )

        loop = SACFlowOnlineLoop(
            actor=FakeActor(),
            env="env",
            replay_buffer=replay,
            trainer=trainer,
            config=SACFlowConfig(min_buffer_size=1, num_updates_per_step=1),
            rollout_fn=rollout_fn,
        )

        step = loop.train_step({"states": "obs"})

        self.assertEqual(step.next_obs, {"states": "next_obs"})
        self.assertTrue(step.rollout.success)
        self.assertEqual(step.update_metrics, [{"critic_loss": 1.0}])

    def test_batched_collection_adds_one_transition_per_vector_slot(self):
        replay = FakeReplay()

        class BatchedActor:
            def sample_chunk(self, obs, train):
                return "flat", "log_pi", "features", torch.zeros(2, 1, 2)

        transitions = [
            make_transition({"states": "first"}, {"states": "first-next"}),
            make_transition({"states": "second"}, {"states": "second-next"}),
        ]

        def batched_rollout_fn(**kwargs):
            return SimpleNamespace(
                rollouts=[
                    SimpleNamespace(transition=transitions[0], raw_rewards=[1.0], success=False, truncated=False),
                    SimpleNamespace(transition=transitions[1], raw_rewards=[2.0], success=True, truncated=False),
                ],
                next_obs={"states": "batched-next"},
            )

        loop = SACFlowOnlineLoop(
            actor=BatchedActor(),
            env="vector-env",
            replay_buffer=replay,
            trainer=FakeTrainer(),
            config=SACFlowConfig(min_buffer_size=10),
            batched_rollout_fn=batched_rollout_fn,
        )

        step = loop.train_step({"states": "batched"})

        self.assertEqual(replay.items, transitions)
        self.assertEqual(step.next_obs, {"states": "batched-next"})
        self.assertEqual(len(step.rollouts), 2)
        self.assertTrue(step.rollouts[1].success)


if __name__ == "__main__":
    unittest.main()
