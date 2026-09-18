import unittest
from types import SimpleNamespace

import torch

from lerobot.rlinf_smolvla_libero.config import SACFlowConfig
from lerobot.rlinf_smolvla_libero.replay import ChunkTransition
from lerobot.rlinf_smolvla_libero.training_loop import (
    SACFlowOnlineLoop,
    summarize_rollouts,
)


class FakeReplay:
    def __init__(self):
        self.items = []
        self.sample_calls = []

    def __len__(self):
        return len(self.items)

    def add(self, transition):
        self.items.append(transition)

    def sample(
        self,
        *,
        batch_size,
        device,
        positive_fraction=0.0,
        task_balanced=False,
        intervention_balanced=False,
    ):
        self.sample_calls.append(
            (batch_size, device, positive_fraction, task_balanced, intervention_balanced)
        )
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
    def test_collection_stats_separate_episode_success_from_positive_reward(self):
        continuing = make_transition({"states": "start"})
        continuing.chunk_reward = 0.0

        successful = make_transition({"states": "near"}, next_obs={"states": "reset"})
        successful.done = True
        successful.chunk_reward = 1.0
        successful.episode_success = True
        successful.episode_completed = True

        failed_timeout = make_transition({"states": "other"}, next_obs={"states": "reset"})
        failed_timeout.done = False
        failed_timeout.chunk_reward = 1.0
        failed_timeout.episode_completed = True
        failed_timeout.truncated = True

        results = [
            SimpleNamespace(
                rollout=SimpleNamespace(
                    transition=continuing,
                    raw_rewards=[0.0],
                    success=False,
                    truncated=False,
                    env_index=0,
                ),
                update_metrics=[],
                rollouts=(),
            ),
            SimpleNamespace(
                rollout=SimpleNamespace(
                    transition=successful,
                    raw_rewards=[1.0],
                    success=True,
                    truncated=False,
                    env_index=0,
                ),
                update_metrics=[],
                rollouts=(),
            ),
            SimpleNamespace(
                rollout=SimpleNamespace(
                    transition=failed_timeout,
                    raw_rewards=[1.0],
                    success=False,
                    truncated=True,
                    env_index=1,
                ),
                update_metrics=[],
                rollouts=(),
            ),
        ]

        stats = summarize_rollouts(results)

        self.assertEqual(stats.transitions_collected, 3)
        self.assertEqual(stats.completed_episode_count, 2)
        self.assertEqual(stats.successful_episode_count, 1)
        self.assertEqual(stats.truncated_episode_count, 1)
        self.assertEqual(stats.terminal_transition_count, 1)
        self.assertEqual(stats.positive_reward_transition_count, 2)
        self.assertEqual(stats.completed_episode_length_sum, 3)
        self.assertEqual(stats.episode_success_rate, 0.5)
        self.assertEqual(stats.positive_reward_transition_fraction, 2 / 3)

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

    def test_completed_episode_backfills_success_and_return_to_go(self):
        actor = FakeActor()
        replay = FakeReplay()
        transitions = []

        def rollout_fn(**kwargs):
            index = len(transitions)
            transition = make_transition(kwargs["curr_obs"], next_obs={"states": f"next-{index}"})
            transition.chunk_reward = 1.0 if index == 2 else 0.0
            transition.done = index == 2
            transition.episode_completed = index == 2
            transition.episode_success = index == 2
            transitions.append(transition)
            return SimpleNamespace(
                transition=transition,
                raw_rewards=[transition.chunk_reward],
                success=index == 2,
                truncated=False,
                env_index=0,
            )

        loop = SACFlowOnlineLoop(
            actor=actor,
            env="env",
            replay_buffer=replay,
            trainer=FakeTrainer(),
            config=SACFlowConfig(min_buffer_size=10),
            rollout_fn=rollout_fn,
        )

        loop.collect_transition({"states": "s0"})
        loop.collect_transition({"states": "s1"})
        loop.collect_transition({"states": "s2"})

        self.assertEqual([item.episode_success for item in transitions], [True, True, True])
        self.assertEqual([item.episode_completed for item in transitions], [False, False, True])
        self.assertAlmostEqual(transitions[0].episode_return_to_go, 0.96**2)
        self.assertAlmostEqual(transitions[1].episode_return_to_go, 0.96)
        self.assertAlmostEqual(transitions[2].episode_return_to_go, 1.0)

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
        self.assertEqual(
            replay.sample_calls,
            [
                (4, "cuda:0", 0.5, True, False),
                (4, "cuda:0", 0.5, True, False),
                (4, "cuda:0", 0.5, True, False),
            ],
        )
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

    def test_run_calls_step_callback_after_replay_update(self):
        replay = FakeReplay()
        callbacks = []

        def rollout_fn(**kwargs):
            return SimpleNamespace(
                transition=make_transition(kwargs["curr_obs"], next_obs={"states": "next_obs"}),
                raw_rewards=[1.0],
                success=False,
                truncated=False,
            )

        loop = SACFlowOnlineLoop(
            actor=FakeActor(),
            env="env",
            replay_buffer=replay,
            trainer=FakeTrainer(),
            config=SACFlowConfig(min_buffer_size=10),
            rollout_fn=rollout_fn,
            step_callback=lambda result, step: callbacks.append((step, len(replay), result.next_obs)),
        )

        loop.run({"states": "obs"}, num_steps=2)

        self.assertEqual(
            callbacks,
            [
                (0, 1, {"states": "next_obs"}),
                (1, 2, {"states": "next_obs"}),
            ],
        )

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

    def test_batched_intervention_collection_assigns_pair_metadata(self):
        replay = FakeReplay()

        class BatchedActor:
            def sample_chunk(self, obs, train):
                return "flat", "log_pi", "features", torch.zeros(2, 1, 2)

        class FakeIntervention:
            applied = (False, True)
            noise_l2 = (0.0, 0.4)
            task_indices = (2, 2)
            slot_indices = (0, 1)

            def apply(self, raw_chunk):
                return SimpleNamespace(
                    raw_chunk=raw_chunk,
                    applied=self.applied,
                    noise_l2=self.noise_l2,
                    task_indices=self.task_indices,
                    slot_indices=self.slot_indices,
                )

        transitions = [
            make_transition({"states": "clean"}, {"states": "clean-next"}),
            make_transition({"states": "intervention"}, {"states": "intervention-next"}),
        ]

        def batched_rollout_fn(**kwargs):
            for index, transition in enumerate(transitions):
                transition.episode_completed = True
                transition.episode_success = index == 0
            return SimpleNamespace(
                rollouts=[
                    SimpleNamespace(
                        transition=transitions[0],
                        raw_rewards=[1.0],
                        success=True,
                        truncated=False,
                        env_index=0,
                    ),
                    SimpleNamespace(
                        transition=transitions[1],
                        raw_rewards=[0.0],
                        success=False,
                        truncated=False,
                        env_index=1,
                    ),
                ],
                next_obs={"states": "next"},
            )

        loop = SACFlowOnlineLoop(
            actor=BatchedActor(),
            env="vector-env",
            replay_buffer=replay,
            trainer=FakeTrainer(),
            config=SACFlowConfig(
                min_buffer_size=10,
                critic_intervention_fraction=0.5,
                critic_intervention_pairing=True,
            ),
            batched_rollout_fn=batched_rollout_fn,
            intervention_collector=FakeIntervention(),
        )

        loop.collect_transition({"states": "batched"})

        self.assertEqual(
            [(item.pair_id, item.pair_branch, item.pair_anchor) for item in replay.items],
            [
                ("task=2:seed=0:noise=default:pair=0:episode=0", "clean", True),
                ("task=2:seed=0:noise=default:pair=0:episode=0", "intervention", True),
            ],
        )

    def test_collection_seed_separates_pair_ids_across_runs(self):
        class BatchedActor:
            def sample_chunk(self, obs, train):
                return "flat", "log_pi", "features", torch.zeros(2, 1, 2)

        class FakeIntervention:
            applied = (False, True)
            noise_l2 = (0.0, 0.3)
            task_indices = (2, 2)
            slot_indices = (0, 1)

            def apply(self, raw_chunk):
                return SimpleNamespace(
                    raw_chunk=raw_chunk,
                    applied=self.applied,
                    noise_l2=self.noise_l2,
                    task_indices=self.task_indices,
                    slot_indices=self.slot_indices,
                )

        def make_loop(seed):
            replay = FakeReplay()
            loop = SACFlowOnlineLoop(
                actor=BatchedActor(),
                env="vector-env",
                replay_buffer=replay,
                trainer=FakeTrainer(),
                config=SACFlowConfig(
                    min_buffer_size=10,
                    critic_intervention_fraction=0.5,
                    critic_intervention_pairing=True,
                ),
                batched_rollout_fn=lambda **kwargs: SimpleNamespace(
                    rollouts=[
                        SimpleNamespace(
                            transition=make_transition({"states": "clean"}),
                            raw_rewards=[1.0],
                            success=True,
                            truncated=False,
                            env_index=0,
                        ),
                        SimpleNamespace(
                            transition=make_transition({"states": "intervention"}),
                            raw_rewards=[0.0],
                            success=False,
                            truncated=False,
                            env_index=1,
                        ),
                    ],
                    next_obs={"states": "next"},
                ),
                intervention_collector=FakeIntervention(),
                collection_seed=seed,
            )
            loop.collect_transition({"states": "batched"})
            return [item.pair_id for item in replay.items]

        self.assertNotEqual(make_loop(7), make_loop(11))

    def test_collection_stats_report_clean_and_intervention_episode_outcomes(self):
        clean = make_transition({"states": "clean"})
        clean.episode_completed = True
        clean.episode_success = True
        clean.intervention_applied = False
        clean.intervention_noise_l2 = 0.0
        clean.intervention_task_index = 2
        intervention = make_transition({"states": "intervention"})
        intervention.episode_completed = True
        intervention.episode_success = False
        intervention.intervention_applied = True
        intervention.intervention_noise_l2 = 0.5
        intervention.intervention_task_index = 2
        rollouts = [
            SimpleNamespace(
                transition=clean,
                raw_rewards=[1.0],
                success=True,
                truncated=False,
                env_index=0,
            ),
            SimpleNamespace(
                transition=intervention,
                raw_rewards=[0.0],
                success=False,
                truncated=False,
                env_index=1,
            ),
        ]

        stats = summarize_rollouts(
            [SimpleNamespace(rollout=rollouts[0], rollouts=tuple(rollouts))]
        )
        metrics = stats.metrics(prefix="env")

        self.assertEqual(metrics["env/clean/episode_success_rate"], 1.0)
        self.assertEqual(metrics["env/intervention/episode_success_rate"], 0.0)
        self.assertEqual(metrics["env/intervention_transition_fraction"], 0.5)
        self.assertEqual(metrics["env/task_2/clean/episodes_completed"], 1.0)
        self.assertEqual(metrics["env/task_2/intervention/episodes_completed"], 1.0)


if __name__ == "__main__":
    unittest.main()
