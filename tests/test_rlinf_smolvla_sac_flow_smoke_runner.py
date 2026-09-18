import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lerobot.rlinf_smolvla_libero.config import SACFlowConfig
from lerobot.rlinf_smolvla_libero.libero_adapter import execute_action_chunk
from lerobot.rlinf_smolvla_libero.smoke_runner import (
    SACFlowRunConfig,
    SACFlowSmokeConfig,
    configure_actor_trainable_scope,
    log_sac_flow_step_results,
    move_policy_to_device,
    MultiTaskVectorEnv,
    postprocess_env_action,
    prepare_policy_observation,
    _run_independent_heldout_check,
    run_sac_flow_training_run,
    run_sac_flow_gpu_smoke,
    select_libero_vector_env,
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
    def test_heldout_check_seeds_all_rngs_and_never_updates_training_state(self):
        transition = SimpleNamespace(
            curr_obs={"states": "curr"},
            next_obs={"states": "next"},
            actions="actions",
            rewards=[0.0],
            done=False,
            horizon=1,
            discount=0.96,
            chunk_reward=0.0,
            episode_completed=False,
        )

        class HeldoutLoop:
            def __init__(self):
                self.step_callback = "step"
                self.update_callback = "update"
                self.reset_calls = 0
                self.run_calls = []

            def reset_episode_tracking(self):
                self.reset_calls += 1

            def run(self, initial_obs, *, num_steps, store_in_replay, update):
                self.run_calls.append((initial_obs, num_steps, store_in_replay, update))
                return [
                    SimpleNamespace(
                        rollout=SimpleNamespace(
                            transition=transition,
                            raw_rewards=[0.0],
                            success=False,
                            truncated=False,
                            env_index=0,
                        ),
                        rollouts=(),
                        update_metrics=[],
                    )
                ]

        class HeldoutTrainer:
            def evaluate_bellman_error(self, batch, *, train):
                return {
                    "bellman_mse": 0.1,
                    "bellman_abs_error": 0.2,
                    "q_mean": 0.3,
                    "target_q_mean": 0.4,
                    "q_head_span": 0.5,
                    "q_head_span_max": 0.6,
                    "sample_count": 1.0,
                }

        loop = HeldoutLoop()
        env = FakeVectorEnv()
        with (
            patch(
                "lerobot.rlinf_smolvla_libero.smoke_runner._seed_sac_flow_rng"
            ) as seed_rng,
            patch(
                "lerobot.rlinf_smolvla_libero.smoke_runner.collate_transitions",
                return_value={"batch": True},
            ),
        ):
            metrics = _run_independent_heldout_check(
                loop=loop,
                env=env,
                seed=2000,
                num_steps=1,
                prepare_initial_obs=lambda raw: {"prepared": raw},
                trainer=HeldoutTrainer(),
                device="cpu",
            )

        seed_rng.assert_called_once_with(2000)
        self.assertEqual(env.reset_calls, [2000])
        self.assertEqual(loop.reset_calls, 1)
        self.assertEqual(loop.run_calls, [({"prepared": {"raw": "obs"}}, 1, False, False)])
        self.assertEqual(loop.step_callback, "step")
        self.assertEqual(loop.update_callback, "update")
        self.assertEqual(metrics["heldout/bellman_abs_error"], 0.2)

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
        self.assertEqual(config.heldout_num_steps, 0)
        self.assertFalse(config.save_checkpoint)

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
                        "env/transitions_collected": 1.0,
                        "env/episodes_completed": 0.0,
                        "env/episodes_successful": 0.0,
                        "env/episodes_truncated": 0.0,
                        "env/terminal_transitions": 0.0,
                        "env/positive_reward_transitions": 1.0,
                        "env/positive_reward_transition_fraction": 1.0,
                        "env/episode_success_rate": 0.0,
                        "env/mean_completed_episode_length": 0.0,
                        "env/raw_reward_sum": 3.0,
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
                self.step_callback = kwargs["step_callback"]

            def run(self, initial_obs, *, num_steps):
                events.append(("run", num_steps))
                results = [
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
                for step, result in enumerate(results):
                    self.step_callback(result, step)
                return results

        def save_checkpoint(**kwargs):
            events.append(("save", kwargs["step"]))
            return Path("/tmp/out/checkpoint_000003")

        result = run_sac_flow_training_run(
            train_cfg=SimpleNamespace(policy="policy_cfg", env="env_cfg", output_dir=Path("/tmp/out")),
            run_cfg=SACFlowRunConfig(
                device="cpu",
                max_train_steps=3,
                max_chunk_steps=1,
                save_checkpoint=True,
            ),
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
        self.assertEqual(logger.logged[0][0]["train/global_step"], 0)
        self.assertIsNone(logger.logged[0][1])

    def test_training_run_resume_keeps_global_step_continuous(self):
        events = []
        vec = FakeVectorEnv()
        logger = FakeWandBLogger()

        def build_runtime(train_cfg, validate_config):
            return SimpleNamespace(
                train_cfg=train_cfg,
                policy="resumed_policy",
                preprocessor=lambda obs: obs,
                postprocessor=lambda action: action,
            )

        def build_loop_components(**kwargs):
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
                events.append(("loop",))

            def run(self, initial_obs, *, num_steps):
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

        def load_checkpoint(**kwargs):
            events.append(("load", kwargs["checkpoint_dir"], kwargs["restore_rng"]))
            return {"step": 280}

        def save_checkpoint(**kwargs):
            events.append(("save", kwargs["step"], kwargs["extra_state"]))
            return Path("/tmp/out/checkpoint_000560")

        result = run_sac_flow_training_run(
            train_cfg=SimpleNamespace(policy="policy_cfg", env="env_cfg", output_dir=Path("/tmp/out")),
            run_cfg=SACFlowRunConfig(
                device="cpu",
                max_train_steps=280,
                max_chunk_steps=1,
                save_checkpoint=True,
            ),
            sac_config=SACFlowConfig(device="cpu", wandb_enable=True),
            logger=logger,
            build_runtime_fn=build_runtime,
            make_env_fn=lambda *args, **kwargs: {"libero_10": {0: vec}},
            make_env_processors_fn=lambda *args: (lambda obs: obs, lambda action: action),
            build_loop_components_fn=build_loop_components,
            loop_cls=FakeLoop,
            save_checkpoint_fn=save_checkpoint,
            load_checkpoint_fn=load_checkpoint,
            resume_checkpoint="/tmp/out/checkpoint_000280",
            close_envs_fn=lambda envs: None,
            preprocess_observation_fn=lambda raw: raw,
            add_envs_task_fn=lambda env, obs: obs,
        )

        self.assertEqual(result.steps, 560)
        self.assertIn(("load", "/tmp/out/checkpoint_000280", True), events)
        save_events = [event for event in events if event[0] == "save"]
        self.assertEqual(len(save_events), 1)
        _, saved_step, extra_state = save_events[0]
        self.assertEqual(saved_step, 560)
        self.assertEqual(extra_state["train_steps"], 280)
        self.assertEqual(extra_state["global_step"], 560)
        self.assertEqual(extra_state["resumed_from"], "/tmp/out/checkpoint_000280")
        self.assertIsInstance(extra_state["collection_stats"], dict)
        self.assertEqual(extra_state["heldout_metrics"], {})
        self.assertEqual(extra_state["critic_metrics"], {})
        self.assertEqual(logger.logged[0][0]["train/global_step"], 280)

    def test_selects_single_vector_env_from_nested_libero_factory_result(self):
        vec = FakeVectorEnv()
        selected = select_single_libero_vector_env({"libero_10": {3: vec}})
        self.assertIs(selected, vec)

    def test_combines_one_vector_env_per_task_for_training(self):
        first = SimpleNamespace(num_envs=2, reset=lambda **kwargs: None)
        second = SimpleNamespace(num_envs=2, reset=lambda **kwargs: None)

        selected = select_libero_vector_env(
            {"libero_10": {0: first, 1: second}},
            expected_num_envs_per_task=2,
        )

        self.assertEqual(selected.num_envs, 4)
        self.assertEqual(selected.vector_envs, [first, second])

    def test_multitask_vector_env_normalizes_heterogeneous_info_keys(self):
        class Child:
            num_envs = 1

            def __init__(self, info):
                self.info = info

            def step(self, action):
                return {"obs": [0]}, [0.0], [False], [False], self.info

            def reset(self, **kwargs):
                return {"obs": [0]}, {}

        env = MultiTaskVectorEnv(
            [
                Child({"is_success": [True], "task_only_a": [1]}),
                Child({"success": [False], "task_only_b": [2]}),
            ]
        )

        result = env.step([[0.0], [0.0]])

        self.assertEqual(result[4]["is_success"].tolist(), [True, False])

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
            smoke_cfg=SACFlowSmokeConfig(
                device="cpu",
                confirm_gpu_smoke=True,
                max_train_steps=2,
                save_checkpoint=True,
            ),
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
