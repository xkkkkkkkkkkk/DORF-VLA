import unittest
from pathlib import Path
from types import SimpleNamespace

from lerobot.rlinf_smolvla_libero.runtime_builder import build_smolvla_policy_runtime


class SACFlowRuntimeBuilderTest(unittest.TestCase):
    def make_cfg(self, *, pretrained_path=None, resume=False):
        policy = SimpleNamespace(
            pretrained_path=pretrained_path,
            input_features={"obs": object()},
            output_features={"action": object()},
            normalization_mapping={"obs": "mean_std", "action": "min_max"},
        )
        return SimpleNamespace(
            policy=policy,
            resume=resume,
            rename_map={"old": "new"},
            validate=lambda: None,
        )

    def test_builds_dataset_policy_and_processors_from_train_config(self):
        calls = []
        cfg = self.make_cfg(pretrained_path=Path("/tmp/checkpoint"))
        dataset_meta = SimpleNamespace(stats={"obs": "stats"})
        dataset = SimpleNamespace(meta=dataset_meta)
        policy_obj = SimpleNamespace(config=cfg.policy)
        preprocessor = object()
        postprocessor = object()

        def make_dataset(train_cfg):
            calls.append(("dataset", train_cfg))
            return dataset

        def make_policy(*, cfg, ds_meta, rename_map):
            calls.append(("policy", cfg, ds_meta, rename_map))
            return policy_obj

        def make_processors(*, policy_cfg, pretrained_path, **kwargs):
            calls.append(("processors", policy_cfg, pretrained_path, kwargs))
            return preprocessor, postprocessor

        runtime = build_smolvla_policy_runtime(
            cfg,
            make_dataset_fn=make_dataset,
            make_policy_fn=make_policy,
            make_processors_fn=make_processors,
        )

        self.assertIs(runtime.dataset, dataset)
        self.assertIs(runtime.dataset_meta, dataset_meta)
        self.assertIs(runtime.policy, policy_obj)
        self.assertIs(runtime.preprocessor, preprocessor)
        self.assertIs(runtime.postprocessor, postprocessor)
        self.assertEqual(calls[0], ("dataset", cfg))
        self.assertEqual(calls[1], ("policy", cfg.policy, dataset_meta, {"old": "new"}))
        processor_call = calls[2]
        self.assertEqual(processor_call[0], "processors")
        self.assertIs(processor_call[1], cfg.policy)
        self.assertEqual(processor_call[2], Path("/tmp/checkpoint"))
        self.assertIn("preprocessor_overrides", processor_call[3])
        self.assertIn("postprocessor_overrides", processor_call[3])

    def test_raises_when_policy_is_missing_after_validation(self):
        cfg = SimpleNamespace(policy=None, validate=lambda: None)
        with self.assertRaisesRegex(RuntimeError, "policy"):
            build_smolvla_policy_runtime(
                cfg,
                make_dataset_fn=lambda cfg: None,
                make_policy_fn=lambda **kwargs: None,
                make_processors_fn=lambda **kwargs: (None, None),
            )

    def test_can_skip_validation_for_prevalidated_config(self):
        cfg = self.make_cfg(pretrained_path=None)
        cfg.validate = lambda: self.fail("validate should not be called")
        dataset = SimpleNamespace(meta=SimpleNamespace(stats={"obs": "stats"}))

        runtime = build_smolvla_policy_runtime(
            cfg,
            validate_config=False,
            make_dataset_fn=lambda train_cfg: dataset,
            make_policy_fn=lambda **kwargs: "policy",
            make_processors_fn=lambda **kwargs: ("pre", "post"),
        )

        self.assertEqual(runtime.policy, "policy")
        self.assertEqual(runtime.preprocessor, "pre")
        self.assertEqual(runtime.postprocessor, "post")


if __name__ == "__main__":
    unittest.main()
