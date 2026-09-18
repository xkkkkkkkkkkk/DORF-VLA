import unittest

import torch

from lerobot.rlinf_smolvla_libero.intervention import FixedSlotInterventionCollector
from lerobot.rlinf_smolvla_libero.libero_adapter import execute_batched_action_chunk


class RecordingVectorEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.actions = []

    def step(self, actions):
        self.actions.append(actions.clone())
        return (
            {"states": torch.zeros(self.num_envs, 2)},
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {"is_success": torch.zeros(self.num_envs, dtype=torch.bool)},
        )


class FixedSlotInterventionCollectorTest(unittest.TestCase):
    def test_builds_one_fixed_intervention_slot_per_two_slot_task(self):
        collector = FixedSlotInterventionCollector(
            num_tasks=2,
            slots_per_task=2,
            intervention_fraction=0.5,
            noise_std=0.3,
            seed=7,
            task_indices=(2, 3),
        )

        self.assertEqual(collector.task_indices, (2, 2, 3, 3))
        self.assertEqual(collector.slot_indices, (0, 1, 0, 1))
        for start in (0, 2):
            self.assertEqual(sum(collector.applied[start : start + 2]), 1)

    def test_only_changes_token_zero_for_intervention_slots(self):
        collector = FixedSlotInterventionCollector(
            num_tasks=2,
            slots_per_task=2,
            intervention_fraction=0.5,
            noise_std=0.3,
            seed=11,
        )
        raw_chunk = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        original = raw_chunk.clone()

        batch = collector.apply(raw_chunk)

        torch.testing.assert_close(raw_chunk, original)
        torch.testing.assert_close(batch.raw_chunk[:, 1:], original[:, 1:])
        for index, applied in enumerate(batch.applied):
            if applied:
                self.assertFalse(torch.equal(batch.raw_chunk[index, 0], original[index, 0]))
                self.assertAlmostEqual(
                    batch.noise_l2[index],
                    float((batch.raw_chunk[index, 0] - original[index, 0]).norm()),
                    places=6,
                )
            else:
                torch.testing.assert_close(batch.raw_chunk[index], original[index])
                self.assertEqual(batch.noise_l2[index], 0.0)

    def test_pairing_copies_clean_action_chunk_before_intervention(self):
        collector = FixedSlotInterventionCollector(
            num_tasks=1,
            slots_per_task=2,
            intervention_fraction=0.5,
            noise_std=0.3,
            seed=11,
            pair_actions=True,
        )
        raw_chunk = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]
        )

        batch = collector.apply(raw_chunk)

        self.assertFalse(batch.applied[0])
        self.assertTrue(batch.applied[1])
        torch.testing.assert_close(batch.raw_chunk[0, 1], raw_chunk[0, 1])
        torch.testing.assert_close(batch.raw_chunk[1, 1], raw_chunk[0, 1])
        self.assertFalse(torch.equal(batch.raw_chunk[1, 0], raw_chunk[0, 0]))

    def test_pairing_uses_each_even_slot_as_its_own_clean_anchor(self):
        collector = FixedSlotInterventionCollector(
            num_tasks=1,
            slots_per_task=4,
            intervention_fraction=0.5,
            noise_std=0.3,
            seed=11,
            pair_actions=True,
        )
        raw_chunk = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[21.0, 22.0], [23.0, 24.0]],
                [[29.0, 30.0], [31.0, 32.0]],
            ]
        )

        batch = collector.apply(raw_chunk)

        self.assertEqual(batch.applied, (False, True, False, True))
        torch.testing.assert_close(batch.raw_chunk[1, 1], raw_chunk[0, 1])
        torch.testing.assert_close(batch.raw_chunk[3, 1], raw_chunk[2, 1])
        self.assertFalse(torch.equal(batch.raw_chunk[1, 0], raw_chunk[0, 0]))
        self.assertFalse(torch.equal(batch.raw_chunk[3, 0], raw_chunk[2, 0]))

    def test_pairing_requires_half_intervention_on_even_slots(self):
        with self.assertRaisesRegex(ValueError, "even number of slots"):
            FixedSlotInterventionCollector(
                num_tasks=1,
                slots_per_task=3,
                intervention_fraction=0.5,
                noise_std=0.3,
                seed=11,
                pair_actions=True,
            )

    def test_executed_and_stored_actions_are_the_same_perturbed_actions(self):
        collector = FixedSlotInterventionCollector(
            num_tasks=2,
            slots_per_task=2,
            intervention_fraction=0.5,
            noise_std=0.3,
            seed=13,
            task_indices=(2, 3),
        )
        raw_chunk = torch.zeros(4, 1, 2)
        batch = collector.apply(raw_chunk)
        env = RecordingVectorEnv(num_envs=4)

        result = execute_batched_action_chunk(
            env=env,
            curr_obs={"states": torch.zeros(4, 2)},
            raw_chunk=batch.raw_chunk,
            gamma=0.96,
            max_chunk_steps=1,
            intervention_applied=batch.applied,
            intervention_noise_l2=batch.noise_l2,
            intervention_task_indices=batch.task_indices,
            intervention_slot_indices=batch.slot_indices,
        )

        torch.testing.assert_close(env.actions[0], batch.raw_chunk[:, 0])
        for index, rollout in enumerate(result.rollouts):
            torch.testing.assert_close(
                rollout.transition.actions,
                batch.raw_chunk[index : index + 1, 0],
            )
            self.assertEqual(
                rollout.transition.intervention_applied,
                batch.applied[index],
            )
            self.assertEqual(
                rollout.transition.intervention_task_index,
                batch.task_indices[index],
            )

    def test_rejects_partial_intervention_with_one_slot_per_task(self):
        with self.assertRaisesRegex(ValueError, "at least two vector slots"):
            FixedSlotInterventionCollector(
                num_tasks=2,
                slots_per_task=1,
                intervention_fraction=0.5,
                noise_std=0.3,
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
