import unittest


class FakeParameter:
    def __init__(self, numel=1):
        self.requires_grad = True
        self._numel = numel

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def numel(self):
        return self._numel


class FakePolicy:
    def __init__(self):
        self.params = {
            "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj.weight": FakeParameter(10),
            "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight": FakeParameter(20),
            "model.state_proj.weight": FakeParameter(3),
            "model.action_in_proj.weight": FakeParameter(4),
            "model.action_out_proj.bias": FakeParameter(5),
            "model.action_time_mlp_in.weight": FakeParameter(6),
            "model.action_time_mlp_out.weight": FakeParameter(7),
            "unrelated.weight": FakeParameter(8),
        }

    def named_parameters(self):
        return list(self.params.items())


class SACFlowTrainableScopeTest(unittest.TestCase):
    def test_action_path_trains_expert_and_action_projections_only(self):
        from lerobot.rlinf_smolvla_libero.trainable_scope import apply_actor_trainable_scope

        policy = FakePolicy()
        audit = apply_actor_trainable_scope(policy, scope="action_path")

        self.assertFalse(
            policy.params["model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj.weight"].requires_grad
        )
        self.assertFalse(policy.params["model.state_proj.weight"].requires_grad)
        self.assertFalse(policy.params["unrelated.weight"].requires_grad)
        self.assertTrue(policy.params["model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"].requires_grad)
        self.assertTrue(policy.params["model.action_in_proj.weight"].requires_grad)
        self.assertTrue(policy.params["model.action_out_proj.bias"].requires_grad)
        self.assertTrue(policy.params["model.action_time_mlp_in.weight"].requires_grad)
        self.assertTrue(policy.params["model.action_time_mlp_out.weight"].requires_grad)
        self.assertEqual(audit.total_params, 63)
        self.assertEqual(audit.trainable_params, 42)
        self.assertIn("model.state_proj.weight", audit.frozen_names)

    def test_unknown_trainable_scope_fails_clearly(self):
        from lerobot.rlinf_smolvla_libero.trainable_scope import apply_actor_trainable_scope

        with self.assertRaisesRegex(ValueError, "actor_train_scope"):
            apply_actor_trainable_scope(FakePolicy(), scope="state_proj")

    def test_iter_trainable_parameters_returns_only_enabled_parameters(self):
        from lerobot.rlinf_smolvla_libero.trainable_scope import (
            apply_actor_trainable_scope,
            iter_trainable_parameters,
        )

        policy = FakePolicy()
        apply_actor_trainable_scope(policy, scope="action_path")

        trainable = list(iter_trainable_parameters(policy))

        self.assertEqual(len(trainable), 5)
        self.assertTrue(all(parameter.requires_grad for parameter in trainable))


if __name__ == "__main__":
    unittest.main()
