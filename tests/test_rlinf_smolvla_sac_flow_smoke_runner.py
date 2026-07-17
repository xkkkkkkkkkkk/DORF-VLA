import unittest
from pathlib import Path
from types import SimpleNamespace

from lerobot.rlinf_smolvla_libero.config import SACFlowConfig
from lerobot.rlinf_smolvla_libero.libero_adapter import execute_action_chunk
from lerobot.rlinf_smolvla_libero.smoke_runner import (
    SACFlowRunConfig,
    SACFlowSmokeConfig,
    configure_actor_trainable_scope,
    log_sac_flow_step_results,
    move_policy_to_device,
    postprocess_env_action,
    prepare_policy_observation,
    run_sac_flow_training_run,
    run_sac_flow_gpu_smoke,
    select_single_libero_vector_env,
)


class FakeVectorEnv:
    num_envs = 1

    def __init__(self):
        self.reset_calls = []

    def reset(self, seed=None):
        self.reset_calls.append(seed)
        return {"raw": "obs"}, {"info": True}

    def step(self, action):
        return {"raw": "next"}, [1.0], [False], [False], {}


class FakeRawChunk:
    ndim = 3
    shape = (1, 1, 2)

    def __iter__(self):
        return iter([[[0.1, 0.2]]])

    def __getitem__(self, item):
        if item == (0, 0):
            return FakeAction([0.1, 0.2])
        if isinstance(item, tuple) and item[0] == slice(None, None, None):
            return [[[0.1, 0.2]]]
        raise IndexError(item)


class FakeAction:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def reshape(self, batch, action_dim):
        if batch != 1 or action_dim != len(self.values):
            raise ValueError("bad reshape")
        return FakeBatchedAction([self.values])


class FakeBatchedAction:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))


class FakePolicyWithTo:
    def __init__(self):
        self.devices = []

    def to(self, device):
        moved = FakePolicyWithTo()
        moved.devices = [*self.devices, device]
        return moved


class FakeReplayBuffer:
    def __len__(self):
        return 3


class FakeWandBLogger:
    def __init__(self):
        self.logged = []

    def log(self, metrics, *, step=None):
        self.logged.append((metrics, step))


class SmokeRunnerTest(unittest.TestCase):
    def test_requires_explicit_confirmation_for_gpu_smoke(self):
        with self.assertRaisesRegex(RuntimeError, "confirm"):
            SACFlowSmokeConfig(device="cuda:0", confirm_gpu_smoke=False)

    def test_rejects_large_smoke_budget(self):
        with self.assertRaisesRegex(ValueError, "max_train_steps"):
            SACFlowSmokeConfig(device="cpu", confirm_gpu_smoke=True, max_train_steps=100)
        with self.assertRaisesRegex(ValueError, "num_updates_per_step"):
            SACFlowSmokeConfig(device="cpu", confirm_gpu_smoke=True, num_updates_per_step=8)

    def test_train_run_config_allows_short_run_budget(self):
        config = SACFlowRunConfig(
            device="cpu",
            max_train_steps=100,
            max_chunk_steps=1,
            num_updates_per_step=4,
            batch_size=2,
            min_buffer_size=2,
            replay_capacity=64,
        )

        self.assertEqual(config.max_train_steps, 100)
        self.assertEqual(config.num_updates_per_step, 4)

    def test_configures_actor_trainable_scope_from_sac_config(self):
        policy = object()
        events = []

        def apply_scope(policy_arg, *, scope):
            events.append((policy_arg, scope))
            return "audit"

        audit = configure_actor_trainable_scope(
            policy,
            SACFlowConfig(actor_train_scope="action_path"),
            apply_scope_fn=apply_scope,
        )

        self.assertEqual(audit, "audit")
        self.assertEqual(events, [(policy, "action_path")])

    def test_logs_rollout_and_update_metrics_to_wandb_logger(self):
        logger = FakeWandBLogger()
        step_result = SimpleNamespace(
            rollout=SimpleNamespace(
                raw_rewards=[1.0, 2.0],
                success=True,
                transition=SimpleNamespace(chunk_reward=2.5, horizon=2),
            ),
            update_metrics=[{"critic_loss": 1.0, "alpha": 0.5}],
        )

        log_sac_flow_step_results(
            [step_result],
            logger=logger,
            replay_buffer=FakeReplayBuffer(),
            start_step=10,
        )

        self.assertEqual(
            logger.logged,
            [
                (
                    {
                        "train/global_step": 10,
                        "env/reward": 3.0,
                        "env/discounted_return": 2.5,
                        "env/success": 1.0,
                        "env/chunk_steps": 2,
                        "train/replay_buffer/size": 3,
                        "train/sac/critic_loss": 1.0,
                        "train/sac/alpha": 0.5,
                    },
                    10,
                )
            ],
        )

    def test_training_run_uses_run_budget_and_logs_metrics(self):
        events = []
        vec = FakeVectorEnv()
        logger = FakeWandBLogger()

        def build_runtime(train_cfg, validate_config):
            events.append(("runtime", validate_config))
            return SimpleNamespace(
                train_cfg=train_cfg,
                policy="policy",
                preprocessor=lambda obs: {"policy": obs},
                postprocessor=lambda action: f"policy_post:{action}",
            )

        def make_env(env_cfg, n_envs, use_async_envs):
            events.append(("env", env_cfg, n_envs, use_async_envs))
            return {"libero_10": {0: vec}}

        def make_env_processors(env_cfg, policy_cfg):
            return (lambda obs: obs, lambda action: action)

        def build_loop_components(**kwargs):
            return {
                "actor": "actor",
                "replay_buffer": FakeReplayBuffer(),
                "trainer": "trainer",
                "q_network": "q",
                "target_q_network": "target_q",
                "temperature": "temperature",
            }

        class FakeLoop:
            def __init__(self, **kwargs):
                events.append(("loop", kwargs["max_chunk_steps"]))
                self.replay_buffer = kwargs["replay_buffer"]

            def run(self, initial_obs, *, num_steps):
                events.append(("run", num_steps))
                return [
                    SimpleNamespace(
                        rollout=SimpleNamespace(
                            raw_rewards=[1.0],
                            success=False,
                            transition=SimpleNamespace(chunk_reward=1.0, horizon=1),
                        ),
                        update_metrics=[{"critic_loss": 1.0}],
                    )
                    for _ in range(num_steps)
                ]

        def save_checkpoint(**kwargs):
            events.append(("save", kwargs["step"]))
            return Path("/tmp/out/checkpoint_000003")

        result = run_sac_flow_training_run(
            train_cfg=SimpleNamespace(policy="policy_cfg", env="env_cfg", output_dir=Path("/tmp/out")),
            run_cfg=SACFlowRunConfig(device="cpu", max_train_steps=3, max_chunk_steps=1),
            sac_config=SACFlowConfig(device="cpu", wandb_enable=True),
            logger=logger,
            build_runtime_fn=build_runtime,
            make_env_fn=make_env,
            make_env_processors_fn=make_env_processors,
            build_loop_components_fn=build_loop_components,
            loop_cls=FakeLoop,
            save_checkpoint_fn=save_checkpoint,
            close_envs_fn=lambda envs: events.append(("close", envs)),
            preprocess_observation_fn=lambda raw: raw,
            add_envs_task_fn=lambda env, obs: obs,
        )

        self.assertEqual(result.steps, 3)
        self.assertEqual(events[0], ("runtime", True))
        self.assertIn(("run", 3), events)
        self.assertIn(("save", 3), events)
        self.assertEqual(len(logger.logged), 3)
        self.assertEqual(logger.logged[0][0]["train/sac/critic_loss"], 1.0)

    def test_selects_single_vector_env_from_nested_libero_factory_result(self):
        vec = FakeVectorEnv()
        selected = select_single_libero_vector_env({"libero_10": {3: vec}})
        self.assertIs(selected, vec)

    def test_prepare_policy_observation_reuses_lerobot_eval_order(self):
        calls = []

        def preprocess(raw):
            calls.append(("preprocess", raw))
            return {"pre": raw}

        def add_task(env, obs):
            calls.append(("task", env, obs))
            obs = dict(obs)
            obs["task"] = ["do task"]
            return obs

        def env_preprocessor(obs):
            calls.append(("env_pre", obs))
            obs = dict(obs)
            obs["env_pre"] = True
            return obs

        def policy_preprocessor(obs):
            calls.append(("policy_pre", obs))
            obs = dict(obs)
            obs["policy_pre"] = True
            return obs

        env = FakeVectorEnv()
        prepared = prepare_policy_observation(
            raw_obs={"raw": "obs"},
            env=env,
            env_preprocessor=env_preprocessor,
            policy_preprocessor=policy_preprocessor,
            preprocess_observation_fn=preprocess,
            add_envs_task_fn=add_task,
        )

        self.assertEqual([item[0] for item in calls], ["preprocess", "task", "env_pre", "policy_pre"])
        self.assertTrue(prepared["policy_pre"])
        self.assertEqual(prepared["task"], ["do task"])

    def test_move_policy_to_device_does_not_mutate_frozen_runtime(self):
        policy = FakePolicyWithTo()
        runtime = SimpleNamespace(policy=policy)

        moved = move_policy_to_device(runtime.policy, "cuda:0")

        self.assertIs(runtime.policy, policy)
        self.assertIsNot(moved, policy)
        self.assertEqual(moved.devices, ["cuda:0"])

    def test_postprocess_env_action_matches_lerobot_eval_order(self):
        calls = []

        class TensorLikeAction:
            def __init__(self, value):
                self.value = value

            def to(self, device):
                calls.append(("to", device, self.value))
                return self

            def numpy(self):
                calls.append(("numpy", self.value))
                return f"np:{self.value}"

        def policy_postprocessor(action):
            calls.append(("policy_post", action))
            return TensorLikeAction("policy")

        def env_postprocessor(transition):
            calls.append(("env_post", transition))
            return {"action": TensorLikeAction("env")}

        result = postprocess_env_action(
            "raw",
            policy_postprocessor=policy_postprocessor,
            env_postprocessor=env_postprocessor,
        )

        self.assertEqual(result, "np:env")
        self.assertEqual([item[0] for item in calls], ["policy_post", "env_post", "to", "numpy"])

    def test_vector_action_postprocessor_receives_batched_action(self):
        seen_shapes = []

        def action_postprocessor(action):
            seen_shapes.append(action.shape)
            return action

        execute_action_chunk(
            env=FakeVectorEnv(),
            curr_obs={"obs": "ready"},
            raw_chunk=FakeRawChunk(),
            gamma=0.99,
            max_chunk_steps=1,
            action_postprocessor=action_postprocessor,
            observation_preparer=lambda raw: {"obs": raw["raw"]},
        )

        self.assertEqual(seen_shapes, [(1, 2)])

    def test_run_smoke_wires_runtime_env_loop_and_checkpoint(self):
        events = []
        vec = FakeVectorEnv()
        logger = FakeWandBLogger()
        runtime = SimpleNamespace(
            train_cfg=SimpleNamespace(env="env_cfg", output_dir=Path("/tmp/out")),
            policy="policy",
            preprocessor=lambda obs: {"ready": obs},
            postprocessor=lambda action: f"policy_post:{action}",
        )

        def build_runtime(train_cfg, validate_config):
            events.append(("runtime", train_cfg, validate_config))
            return runtime

        def make_env(env_cfg, n_envs, use_async_envs):
            events.append(("env", env_cfg, n_envs, use_async_envs))
            return {"libero_10": {0: vec}}

        def make_env_processors(env_cfg, policy_cfg):
            events.append(("env_processors", env_cfg, policy_cfg))
            return lambda obs: obs, lambda transition: {"action": f"env_post:{transition['action']}"}

        def build_loop_components(runtime, sac_config, initial_obs):
            events.append(("components", runtime.policy, sac_config.device, initial_obs))
            return {
                "actor": "actor",
                "replay_buffer": "replay",
                "trainer": "trainer",
                "q_network": "q",
                "target_q_network": "target_q",
                "temperature": "temperature",
            }

        class FakeLoop:
            def __init__(self, **kwargs):
                events.append(("loop", kwargs))

            def run(self, initial_obs, *, num_steps):
                events.append(("run", initial_obs, num_steps))
                return [
                    SimpleNamespace(
                        rollout=SimpleNamespace(
                            raw_rewards=[1.0],
                            success=False,
                            transition=SimpleNamespace(chunk_reward=1.0, horizon=1),
                        ),
                        update_metrics=[{"critic_loss": 1.0}],
                    )
                ]

        def save_checkpoint(**kwargs):
            events.append(("save", kwargs["step"], kwargs["output_dir"]))
            return Path("/tmp/out/checkpoint_000002")

        def close_envs(envs):
            events.append(("close", envs))

        result = run_sac_flow_gpu_smoke(
            train_cfg=SimpleNamespace(policy="policy_cfg", env="env_cfg", output_dir=Path("/tmp/out")),
            smoke_cfg=SACFlowSmokeConfig(device="cpu", confirm_gpu_smoke=True, max_train_steps=2),
            sac_config=SACFlowConfig(device="cpu", min_buffer_size=1, num_updates_per_step=1, batch_size=1),
            logger=logger,
            build_runtime_fn=build_runtime,
            make_env_fn=make_env,
            make_env_processors_fn=make_env_processors,
            build_loop_components_fn=build_loop_components,
            loop_cls=FakeLoop,
            save_checkpoint_fn=save_checkpoint,
            close_envs_fn=close_envs,
            preprocess_observation_fn=lambda raw: {"pre": raw},
            add_envs_task_fn=lambda env, obs: obs,
        )

        self.assertEqual(result.steps, 2)
        self.assertEqual(result.checkpoint_dir, Path("/tmp/out/checkpoint_000002"))
        self.assertEqual(events[0][0], "runtime")
        self.assertEqual(events[1], ("env", "env_cfg", 1, False))
        self.assertEqual(events[3][0], "components")
        self.assertEqual(events[4][0], "loop")
        self.assertEqual(events[4][1]["max_chunk_steps"], 1)
        self.assertEqual(events[4][1]["action_postprocessor"]("raw"), "env_post:policy_post:raw")
        self.assertEqual(events[-2][0], "save")
        self.assertEqual(events[-1], ("close", {"libero_10": {0: vec}}))
        self.assertEqual(len(logger.logged), 1)
        self.assertEqual(logger.logged[0][0]["train/sac/critic_loss"], 1.0)


if __name__ == "__main__":
    unittest.main()
