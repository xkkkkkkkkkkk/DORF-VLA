# SmolVLA SAC-Flow Post-Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement explicit action-path training for SmolVLA SAC-Flow, WandB logging enabled by default for training scripts, and reproducible smoke/short-run scripts.

**Architecture:** Add a small trainable-scope module that freezes everything except the SmolVLA action path, then make the SAC trainer optimize only trainable actor parameters. Add a lightweight logging module used by the runner so existing smoke behavior remains testable and short-run training can report metrics to WandB. Reuse the existing GPU smoke runner by generalizing it with a non-smoke train config while preserving strict smoke limits.

**Tech Stack:** Python dataclasses, PyTorch optimizers, unittest/pytest, optional WandB import, shell scripts for AutoDL execution.

---

## File Structure

- Create `lerobot/rlinf_smolvla_libero/trainable_scope.py`: applies and audits `action_path` trainable scope.
- Create `lerobot/rlinf_smolvla_libero/wandb_logger.py`: optional WandB wrapper and metric flattening.
- Modify `lerobot/rlinf_smolvla_libero/config.py`: add `actor_train_scope` and WandB config fields.
- Modify `lerobot/rlinf_smolvla_libero/trainer.py`: build actor optimizer from trainable parameters only.
- Modify `lerobot/rlinf_smolvla_libero/smoke_runner.py`: apply trainable scope, emit audit, log loop metrics, and add a less restricted train-run config.
- Modify `lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py`: add train-run entry arguments and WandB overrides.
- Create `scripts/rlinf_smolvla_libero/gpu_smoke.sh`: task14 smoke wrapper with WandB disabled.
- Create `scripts/rlinf_smolvla_libero/short_run_wandb.sh`: short-run wrapper with WandB enabled by default.
- Create `scripts/rlinf_smolvla_libero/baseline_run_wandb.sh`: longer baseline wrapper with conservative defaults.
- Add tests in existing `tests/test_rlinf_smolvla_sac_flow_*.py` files and new focused tests for trainable scope and WandB.

## Task 1: Trainable Scope

**Files:**
- Create: `lerobot/rlinf_smolvla_libero/trainable_scope.py`
- Modify: `lerobot/rlinf_smolvla_libero/config.py`
- Test: `tests/test_rlinf_smolvla_sac_flow_trainable_scope.py`

- [ ] Write failing tests for `action_path`: VLM and `state_proj` frozen, `lm_expert` and action projections trainable.
- [ ] Run: `python -m pytest tests/test_rlinf_smolvla_sac_flow_trainable_scope.py -q`
- [ ] Implement `apply_actor_trainable_scope(policy, scope="action_path")`.
- [ ] Implement `audit_trainable_parameters(module)`.
- [ ] Add `actor_train_scope: str = "action_path"` to `SACFlowConfig`.
- [ ] Run the focused tests again until green.

## Task 2: Optimizer Uses Trainable Parameters Only

**Files:**
- Modify: `lerobot/rlinf_smolvla_libero/trainer.py`
- Test: `tests/test_rlinf_smolvla_sac_flow_trainer.py`

- [ ] Write a failing test where actor has one frozen parameter and one trainable parameter, then assert the default optimizer only receives the trainable one.
- [ ] Run: `python -m pytest tests/test_rlinf_smolvla_sac_flow_trainer.py::SACFlowTrainerTest::test_default_actor_optimizer_uses_only_trainable_parameters -q`
- [ ] Add a helper that filters parameters with `requires_grad=True`.
- [ ] Raise a clear error when no actor trainable parameters exist.
- [ ] Run the focused trainer test.

## Task 3: Apply Scope in Runtime Components

**Files:**
- Modify: `lerobot/rlinf_smolvla_libero/smoke_runner.py`
- Test: `tests/test_rlinf_smolvla_sac_flow_smoke_runner.py`

- [ ] Write a failing test proving `build_default_loop_components()` calls the trainable-scope function before constructing `SACFlowTrainer`.
- [ ] Run the focused smoke-runner test.
- [ ] Apply the scope after moving policy to device and before actor/trainer construction.
- [ ] Return the trainable audit in loop components.
- [ ] Run the focused smoke-runner test.

## Task 4: WandB Logger

**Files:**
- Create: `lerobot/rlinf_smolvla_libero/wandb_logger.py`
- Modify: `lerobot/rlinf_smolvla_libero/config.py`
- Test: `tests/test_rlinf_smolvla_sac_flow_wandb_logger.py`

- [ ] Write failing tests for disabled mode not importing WandB and enabled mode logging expected keys through a fake module.
- [ ] Run the focused WandB tests.
- [ ] Implement `SACFlowWandBConfig`, `SACFlowWandBLogger`, and metric helpers.
- [ ] Add config fields for `wandb_enable`, `wandb_project`, `wandb_run_name`, `wandb_mode`, and `wandb_tags`.
- [ ] Run the focused WandB tests.

## Task 5: Runner Logging and Train-Run Config

**Files:**
- Modify: `lerobot/rlinf_smolvla_libero/smoke_runner.py`
- Test: `tests/test_rlinf_smolvla_sac_flow_smoke_runner.py`

- [ ] Write failing tests that loop results are logged with `train/global_step`, replay size, env reward, and SAC metrics.
- [ ] Add a non-smoke `SACFlowRunConfig` without the hard smoke maxima.
- [ ] Keep `SACFlowSmokeConfig` strict and WandB-disabled by default.
- [ ] Add `run_sac_flow_training_run()` or generalize the runner without weakening smoke.
- [ ] Run the focused smoke-runner tests.

## Task 6: CLI Entry

**Files:**
- Modify: `lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py`
- Test: `tests/test_rlinf_smolvla_sac_flow_entry.py`

- [ ] Write failing tests for a `--train-run` path that calls the generalized runner and does not raise the old `NotImplementedError`.
- [ ] Write failing tests for `--sac-flow.actor-train-scope=action_path` and WandB override parsing.
- [ ] Implement CLI extraction for string/bool/list-like overrides needed by train scope and WandB.
- [ ] Preserve `--gpu-smoke` confirmation behavior.
- [ ] Run the focused entry tests.

## Task 7: Shell Scripts

**Files:**
- Create: `scripts/rlinf_smolvla_libero/gpu_smoke.sh`
- Create: `scripts/rlinf_smolvla_libero/short_run_wandb.sh`
- Create: `scripts/rlinf_smolvla_libero/baseline_run_wandb.sh`
- Test: `tests/test_rlinf_smolvla_sac_flow_scripts.py`

- [ ] Write tests that scripts contain the required env vars, dataset root, cache paths, train scope, and WandB defaults.
- [ ] Implement scripts with manual `WANDB_PROJECT` support and timestamp fallback.
- [ ] Ensure smoke script disables WandB and train scripts enable it.
- [ ] Run script tests.

## Task 8: Verification

**Files:**
- No new files.

- [ ] Run local focused tests that do not require torch if local torch is unavailable.
- [ ] Run all local available SAC-Flow tests.
- [ ] If local torch is unavailable, record that torch-dependent tests must run on AutoDL.
- [ ] Run `python -m py_compile` on modified Python files.
- [ ] Review `git diff`.

## Self-Review

Spec coverage:

- `action_path` scope: Tasks 1-3.
- `state_proj` frozen by default: Task 1.
- optimizer uses trainable parameters only: Task 2.
- WandB default for training scripts: Tasks 4, 6, 7.
- smoke keeps WandB disabled and strict budgets: Tasks 5-7.
- scripts: Task 7.
- tests and verification: Task 8.

No placeholder requirements remain for the implementation scope.
