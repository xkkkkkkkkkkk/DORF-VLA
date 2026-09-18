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
            {
                "curr_obs",
                "next_obs",
                "actions",
                "rewards",
                "terminations",
                "discounts",
                "replay_positive_reward_fraction",
                "batch_positive_reward_fraction",
                "replay_positive_outcome_fraction",
                "batch_positive_outcome_fraction",
                "replay_intervention_fraction",
                "batch_intervention_fraction",
                "replay_intervention_labeled_fraction",
                "batch_intervention_labeled_fraction",
                "intervention_applied",
                "intervention_noise_l2",
                "episode_returns",
                "episode_return_mask",
                "batch_task_count",
                "horizons",
            },
        )
        self.assertEqual(batch["actions"].shape, (2, 2))
        self.assertEqual(batch["curr_obs"]["states"].shape, (2, 2))
        self.assertEqual(batch["next_obs"]["states"].shape, (2, 2))
        self.assertEqual(batch["rewards"].shape, (2, 1))
        self.assertEqual(batch["terminations"].shape, (2, 1))
        self.assertEqual(batch["terminations"].dtype, torch.bool)
        self.assertEqual(batch["discounts"].shape, (2, 1))
        self.assertEqual(batch["replay_positive_reward_fraction"], 1.0)
        self.assertEqual(batch["batch_positive_reward_fraction"], 1.0)
        self.assertEqual(batch["replay_positive_outcome_fraction"], 1.0)
        self.assertEqual(batch["batch_positive_outcome_fraction"], 1.0)
        self.assertEqual(batch["episode_returns"].shape, (2, 1))
        self.assertEqual(batch["episode_return_mask"].shape, (2, 1))
        self.assertEqual(batch["batch_task_count"], 1.0)
        self.assertEqual(batch["replay_intervention_labeled_fraction"], 0.0)
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

    def test_resume_can_increase_capacity_but_not_reduce_it(self):
        source = ChunkReplayBuffer(capacity=2, seed=123)
        source.add(self.make_transition(1.0))
        state = source.state_dict()

        expanded = ChunkReplayBuffer(capacity=4, seed=456)
        expanded.load_state_dict(state)
        self.assertEqual(expanded._items.maxlen, 4)
        self.assertEqual(len(expanded), 1)

        reduced = ChunkReplayBuffer(capacity=1, seed=456)
        with self.assertRaisesRegex(ValueError, "cannot be reduced"):
            reduced.load_state_dict(state)

    def test_intervention_labels_survive_replay_checkpoint_restore(self):
        source = ChunkReplayBuffer(capacity=2, seed=123)
        transition = self.make_transition(1.0)
        transition.intervention_applied = True
        transition.intervention_noise_l2 = 0.4
        transition.intervention_task_index = 3
        transition.intervention_slot_index = 1
        source.add(transition)

        restored = ChunkReplayBuffer(capacity=2, seed=456)
        restored.load_state_dict(source.state_dict())
        loaded = restored.items()[0]

        self.assertTrue(loaded.intervention_applied)
        self.assertEqual(loaded.intervention_noise_l2, 0.4)
        self.assertEqual(loaded.intervention_task_index, 3)
        self.assertEqual(loaded.intervention_slot_index, 1)

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

    def test_sample_right_pads_mixed_task_language_sequences(self):
        buffer = ChunkReplayBuffer(capacity=4, seed=123)
        first = self.make_transition(1.0)
        second = self.make_transition(2.0)
        first.curr_obs.update(
            {
                "observation.language.tokens": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "observation.language.attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        )
        second.curr_obs.update(
            {
                "observation.language.tokens": torch.tensor([[4, 5]], dtype=torch.long),
                "observation.language.attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }
        )
        buffer.add(first)
        buffer.add(second)

        batch = buffer.sample(batch_size=2, device=torch.device("cpu"))

        self.assertEqual(batch["curr_obs"]["observation.language.tokens"].shape, (2, 3))
        self.assertEqual(batch["curr_obs"]["observation.language.attention_mask"].shape, (2, 3))

    def test_sample_can_balance_positive_rewards_and_tasks(self):
        buffer = ChunkReplayBuffer(capacity=8, seed=123)
        for index in range(4):
            transition = self.make_transition(float(index))
            transition.chunk_reward = 1.0 if index < 2 else 0.0
            transition.curr_obs["task"] = [f"task-{index % 2}"]
            buffer.add(transition)

        batch = buffer.sample(
            batch_size=4,
            device=torch.device("cpu"),
            positive_fraction=0.5,
            task_balanced=True,
        )

        self.assertEqual(batch["batch_positive_reward_fraction"], 0.5)
        self.assertEqual(batch["batch_task_count"], 2.0)

    def test_sample_uses_all_transitions_from_successful_episodes(self):
        buffer = ChunkReplayBuffer(capacity=4, seed=123)
        prefix = self.make_transition(1.0)
        prefix.chunk_reward = 0.0
        prefix.episode_success = True
        ordinary = self.make_transition(2.0)
        ordinary.chunk_reward = 0.0
        buffer.add(prefix)
        buffer.add(ordinary)

        batch = buffer.sample(
            batch_size=2,
            device=torch.device("cpu"),
            positive_fraction=0.5,
            task_balanced=False,
        )

        self.assertEqual(batch["batch_positive_reward_fraction"], 0.0)
        self.assertEqual(batch["batch_positive_outcome_fraction"], 0.5)

    def test_sample_balances_task_and_intervention_groups(self):
        buffer = ChunkReplayBuffer(capacity=16, seed=123)
        value = 0
        for task in ("task-a", "task-b"):
            for intervention in (False, True):
                for _ in range(3):
                    transition = self.make_transition(float(value))
                    value += 1
                    transition.chunk_reward = 0.0
                    transition.curr_obs["task"] = [task]
                    transition.intervention_applied = intervention
                    transition.intervention_noise_l2 = 0.3 if intervention else 0.0
                    buffer.add(transition)

        batch = buffer.sample(
            batch_size=4,
            device=torch.device("cpu"),
            task_balanced=True,
            intervention_balanced=True,
        )

        self.assertEqual(batch["batch_task_count"], 2.0)
        self.assertEqual(batch["batch_intervention_fraction"], 0.5)
        self.assertEqual(batch["batch_intervention_labeled_fraction"], 1.0)
        self.assertEqual(int(batch["intervention_applied"].sum()), 2)

    def make_paired_transition(
        self,
        *,
        branch: str,
        success: bool,
        length: int,
        pair_id: str = "task=2:pair=0:episode=0",
    ) -> ChunkTransition:
        transition = self.make_transition(1.0 if branch == "clean" else 2.0)
        transition.actions = torch.tensor(
            [[1.0, 0.0]] if branch == "clean" else [[0.0, 1.0]],
            dtype=torch.float32,
        )
        transition.pair_id = pair_id
        transition.pair_branch = branch
        transition.pair_anchor = True
        transition.episode_success = success
        transition.episode_completed = True
        transition.episode_length = length
        return transition

    def test_sample_pairwise_prefers_success_over_failure_from_same_pair(self):
        buffer = ChunkReplayBuffer(capacity=4, seed=123)
        buffer.add(
            self.make_paired_transition(
                branch="clean",
                success=True,
                length=20,
            )
        )
        buffer.add(
            self.make_paired_transition(
                branch="intervention",
                success=False,
                length=10,
            )
        )

        batch = buffer.sample_pairwise(
            batch_size=2,
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(batch)
        self.assertEqual(batch["pair_available_count"], 1.0)
        self.assertEqual(batch["pair_count"], 2.0)
        torch.testing.assert_close(
            batch["pair_preferred_actions"],
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        )
        torch.testing.assert_close(
            batch["pair_rejected_actions"],
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        )

    def test_sample_pairwise_prefers_shorter_success_when_both_succeed(self):
        buffer = ChunkReplayBuffer(capacity=4, seed=123)
        buffer.add(
            self.make_paired_transition(
                branch="clean",
                success=True,
                length=30,
            )
        )
        buffer.add(
            self.make_paired_transition(
                branch="intervention",
                success=True,
                length=20,
            )
        )

        batch = buffer.sample_pairwise(
            batch_size=1,
            device=torch.device("cpu"),
            min_length_gap=5,
        )

        self.assertIsNotNone(batch)
        torch.testing.assert_close(
            batch["pair_preferred_actions"],
            torch.tensor([[0.0, 1.0]]),
        )
        torch.testing.assert_close(
            batch["pair_rejected_actions"],
            torch.tensor([[1.0, 0.0]]),
        )

    def test_sample_pairwise_discards_both_failed_or_ambiguous_success_pairs(self):
        buffer = ChunkReplayBuffer(capacity=8, seed=123)
        buffer.add(
            self.make_paired_transition(
                branch="clean",
                success=False,
                length=20,
                pair_id="task=2:episode=0",
            )
        )
        buffer.add(
            self.make_paired_transition(
                branch="intervention",
                success=False,
                length=20,
                pair_id="task=2:episode=0",
            )
        )
        buffer.add(
            self.make_paired_transition(
                branch="clean",
                success=True,
                length=22,
                pair_id="task=2:episode=1",
            )
        )
        buffer.add(
            self.make_paired_transition(
                branch="intervention",
                success=True,
                length=24,
                pair_id="task=2:episode=1",
            )
        )

        self.assertIsNone(
            buffer.sample_pairwise(
                batch_size=1,
                device=torch.device("cpu"),
                min_length_gap=5,
            )
        )

    def test_verified_pair_count_matches_pairwise_validity_rules(self):
        buffer = ChunkReplayBuffer(capacity=8, seed=123)
        buffer.add(self.make_paired_transition(branch="clean", success=True, length=10, pair_id="p0"))
        buffer.add(self.make_paired_transition(branch="intervention", success=False, length=20, pair_id="p0"))
        buffer.add(self.make_paired_transition(branch="clean", success=True, length=10, pair_id="p1"))
        buffer.add(self.make_paired_transition(branch="intervention", success=True, length=12, pair_id="p1"))
        self.assertEqual(buffer.verified_pair_count(min_length_gap=5), 1)


if __name__ == "__main__":
    unittest.main()
