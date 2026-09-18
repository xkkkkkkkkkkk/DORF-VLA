import unittest

import torch
import torch.nn as nn

from lerobot.rlinf_smolvla_libero.critic_diagnostics import (
    evaluate_actor_gradient_compatibility,
    evaluate_critic_action_gate,
    evaluate_critic_root_causes,
    evaluate_intervention_coverage,
    evaluate_pairwise_action_ranking,
)
from lerobot.rlinf_smolvla_libero.replay import ChunkTransition


class AnalyticActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def encode_obs(self, obs):
        return obs["states"]

    def sample_chunk(self, obs, train):
        features = self.encode_obs(obs)
        actions = self.weight * torch.ones(features.shape[0], 2)
        log_pi = torch.zeros(features.shape[0], 1)
        raw_chunk = actions.reshape(features.shape[0], 1, 2)
        return actions, log_pi, features, raw_chunk


class ActionSensitiveQ(nn.Module):
    def forward(self, features, actions):
        value = features[:, :1] + 2.0 * actions[:, :1]
        return torch.cat((value, value), dim=-1)


class SignConflictActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def sample_chunk(self, obs, train):
        features = obs["features"]
        actions = self.weight * torch.ones(features.shape[0], 1)
        log_pi = torch.zeros(features.shape[0], 1)
        raw_chunk = actions.reshape(features.shape[0], 1, 1)
        return actions, log_pi, features, raw_chunk


class SignConflictQ(nn.Module):
    def forward(self, features, actions):
        value = features[:, :1] * actions[:, :1]
        return torch.cat((value, value), dim=-1)


def make_transition(*, state, action, task):
    return ChunkTransition(
        curr_obs={"states": torch.tensor([[state, state + 1.0]]), "task": [task]},
        next_obs={"states": torch.tensor([[state + 1.0, state + 2.0]])},
        actions=torch.tensor([action], dtype=torch.float32),
        rewards=[0.0],
        done=False,
        horizon=1,
        discount=0.96,
        chunk_reward=0.0,
    )


class CriticDiagnosticsTest(unittest.TestCase):
    def test_intervention_coverage_uses_only_completed_episode_outcomes(self):
        clean = make_transition(state=0.0, action=[0.0, 0.0], task="task-a")
        clean.intervention_applied = False
        clean.intervention_task_index = 2
        clean.episode_completed = True
        clean.episode_success = True
        intervention = make_transition(state=1.0, action=[0.1, 0.1], task="task-a")
        intervention.intervention_applied = True
        intervention.intervention_noise_l2 = 0.4
        intervention.intervention_task_index = 2
        intervention.episode_completed = True
        intervention.episode_success = False
        continuing = make_transition(state=2.0, action=[0.2, 0.2], task="task-a")
        continuing.intervention_applied = True
        continuing.intervention_task_index = 2
        continuing.episode_success = True

        metrics = evaluate_intervention_coverage(
            [clean, intervention, continuing]
        )

        self.assertEqual(metrics["clean_episode_success_rate"], 1.0)
        self.assertEqual(metrics["intervention_episode_success_rate"], 0.0)
        self.assertEqual(metrics["intervention_completed_episode_count"], 1.0)
        self.assertEqual(metrics["intervention_minus_clean_success_rate"], -1.0)
        self.assertEqual(metrics["task_2_clean_completed_episode_count"], 1.0)

    def test_pairwise_action_ranking_prefers_verified_success(self):
        clean = make_transition(state=0.0, action=[1.0, 0.0], task="task-a")
        clean.pair_id = "task=2:episode=0"
        clean.pair_branch = "clean"
        clean.pair_anchor = True
        clean.episode_success = True
        clean.episode_length = 10
        intervention = make_transition(state=0.0, action=[0.0, 1.0], task="task-a")
        intervention.pair_id = "task=2:episode=0"
        intervention.pair_branch = "intervention"
        intervention.pair_anchor = True
        intervention.episode_success = False
        intervention.episode_length = 20

        metrics = evaluate_pairwise_action_ranking(
            actor=AnalyticActor(),
            q_network=ActionSensitiveQ(),
            transitions=[clean, intervention],
            device=torch.device("cpu"),
        )

        self.assertEqual(metrics["pair_available_count"], 1.0)
        self.assertEqual(metrics["pair_id_group_count"], 1.0)
        self.assertEqual(metrics["pairwise_ranking_accuracy"], 1.0)
        self.assertGreater(metrics["pairwise_q_gap_mean"], 0.0)
        self.assertEqual(metrics["pairwise_margin_violation_fraction"], 0.0)

    def test_pairwise_action_ranking_uses_shorter_success_when_both_succeed(self):
        clean = make_transition(state=0.0, action=[0.0, 1.0], task="task-a")
        clean.pair_id = "task=2:episode=0"
        clean.pair_branch = "clean"
        clean.pair_anchor = True
        clean.episode_success = True
        clean.episode_length = 20
        intervention = make_transition(state=0.0, action=[1.0, 0.0], task="task-a")
        intervention.pair_id = "task=2:episode=0"
        intervention.pair_branch = "intervention"
        intervention.pair_anchor = True
        intervention.episode_success = True
        intervention.episode_length = 10

        metrics = evaluate_pairwise_action_ranking(
            actor=AnalyticActor(),
            q_network=ActionSensitiveQ(),
            transitions=[clean, intervention],
            device=torch.device("cpu"),
        )

        self.assertEqual(metrics["pair_available_count"], 1.0)
        self.assertEqual(metrics["pairwise_ranking_accuracy"], 1.0)

    def test_pairwise_action_ranking_reports_rejected_pair_reasons(self):
        missing_branch = make_transition(state=0.0, action=[0.0, 0.0], task="task-a")
        missing_branch.pair_id = "task=2:seed=7:pair=0:episode=0"
        missing_branch.pair_branch = "clean"
        missing_branch.pair_anchor = True
        missing_branch.episode_success = True
        missing_branch.episode_length = 10

        same_outcome = []
        for branch in ("clean", "intervention"):
            transition = make_transition(state=1.0, action=[0.0, 0.0], task="task-a")
            transition.pair_id = "task=2:seed=7:pair=1:episode=0"
            transition.pair_branch = branch
            transition.pair_anchor = True
            transition.episode_success = False
            transition.episode_length = 10
            same_outcome.append(transition)

        short_gap = []
        for branch in ("clean", "intervention"):
            transition = make_transition(state=2.0, action=[0.0, 0.0], task="task-a")
            transition.pair_id = "task=2:seed=7:pair=2:episode=0"
            transition.pair_branch = branch
            transition.pair_anchor = True
            transition.episode_success = True
            transition.episode_length = 10 if branch == "clean" else 12
            short_gap.append(transition)

        metrics = evaluate_pairwise_action_ranking(
            actor=AnalyticActor(),
            q_network=ActionSensitiveQ(),
            transitions=[missing_branch, *same_outcome, *short_gap],
            device=torch.device("cpu"),
            min_length_gap=5,
        )

        self.assertEqual(metrics["pair_id_group_count"], 3.0)
        self.assertEqual(metrics["pair_missing_branch_count"], 1.0)
        self.assertEqual(metrics["pair_same_outcome_count"], 1.0)
        self.assertEqual(metrics["pair_length_gap_rejected_count"], 1.0)
        self.assertEqual(metrics["pair_available_count"], 0.0)

    def test_action_gate_bounds_long_replay_without_losing_late_successes(self):
        transitions = [
            make_transition(
                state=float(index),
                action=[0.1 * index, -0.1 * index],
                task="task-a",
            )
            for index in range(20)
        ]
        transitions[-2].episode_completed = True
        transitions[-2].done = True
        transitions[-2].chunk_reward = 1.0
        transitions[-1].episode_completed = True

        metrics = evaluate_critic_action_gate(
            actor=AnalyticActor(),
            q_network=ActionSensitiveQ(),
            transitions=transitions,
            device=torch.device("cpu"),
            max_transitions_per_task=8,
            seed=7,
        )

        self.assertEqual(metrics["replay_transition_count"], 8.0)
        self.assertGreaterEqual(metrics["success_transition_count"], 1.0)

    def test_root_cause_diagnostics_expose_coordinate_and_action_sensitivity(self):
        transitions = [
            make_transition(state=0.5, action=[2.0, -1.5], task="task-a"),
            make_transition(state=0.7, action=[1.5, -1.0], task="task-a"),
            make_transition(state=-0.5, action=[0.2, 0.4], task="task-b"),
            make_transition(state=-0.7, action=[0.3, 0.5], task="task-b"),
        ]

        metrics = evaluate_critic_root_causes(
            actor=AnalyticActor(),
            q_network=ActionSensitiveQ(),
            transitions=transitions,
            device=torch.device("cpu"),
            perturb_repeats=3,
            max_transitions_per_task=2,
            seed=7,
        )

        self.assertEqual(metrics["task_count"], 2.0)
        self.assertGreater(metrics["action_outside_unit_fraction"], 0.0)
        self.assertGreater(metrics["perturbation_clamp_fraction"], 0.0)
        self.assertGreater(metrics["q_action_grad_norm_mean"], 0.0)
        self.assertGreater(metrics["q_action_shuffle_abs_delta_mean"], 0.0)
        self.assertIn("task_0_replay_vs_unclamped_perturbed_fraction_mean", metrics)
        self.assertIn("task_1_replay_vs_unclamped_perturbed_fraction_mean", metrics)

    def test_actor_gradient_diagnostic_finds_task_conflict_without_updating_actor(self):
        actor = SignConflictActor()
        before = actor.weight.detach().clone()
        transitions = [
            ChunkTransition(
                curr_obs={"features": torch.tensor([[1.0]]), "task": ["task-a"]},
                next_obs={"features": torch.tensor([[1.0]])},
                actions=torch.tensor([[1.0]]),
                rewards=[0.0],
                done=False,
                horizon=1,
                discount=0.96,
                chunk_reward=0.0,
            ),
            ChunkTransition(
                curr_obs={"features": torch.tensor([[-1.0]]), "task": ["task-b"]},
                next_obs={"features": torch.tensor([[-1.0]])},
                actions=torch.tensor([[1.0]]),
                rewards=[0.0],
                done=False,
                horizon=1,
                discount=0.96,
                chunk_reward=0.0,
            ),
        ]

        metrics = evaluate_actor_gradient_compatibility(
            actor=actor,
            q_network=SignConflictQ(),
            transitions=transitions,
            device=torch.device("cpu"),
            include_kl=False,
            repeats=2,
            max_transitions_per_task=1,
        )

        self.assertEqual(metrics["task_count"], 2.0)
        self.assertAlmostEqual(metrics["q_pairwise_cosine_min"], -1.0)
        self.assertEqual(metrics["q_pairwise_negative_fraction"], 1.0)
        self.assertAlmostEqual(metrics["total_mean_gradient_to_task_norm_ratio"], 0.0)
        torch.testing.assert_close(actor.weight.detach(), before)


if __name__ == "__main__":
    unittest.main()
