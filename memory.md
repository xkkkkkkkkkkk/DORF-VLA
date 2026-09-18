# DORF-VLA Project Memory

Last updated: 2026-09-18

This file is the project-level source of truth for the current DORF-VLA
SmolVLA/LIBERO SAC-Flow post-training work. Read it before changing code or
starting an AutoDL experiment. Update it in the same task after any source,
configuration, test, server, or experiment change.

## 1. Project and Runtime Locations

- Local checkout: `/Users/yikai/Documents/DORF-VLA`
- Local branch: `codex/v2`
- AutoDL SSH alias: `autodl`
- Remote checkout: `/root/autodl-tmp/lerobot/src`
- Remote conda environment: `lerobot`
- Actual AutoFS mount: `/autodl-fs/data`
- Remote symlink used by scripts: `/root/autodl-fs -> /autodl-fs/data`
- Remote temporary/system experiment disk: `/root/autodl-tmp`
- Hugging Face model cache: `/root/.cache/huggingface`
- LIBERO dataset: `/root/autodl-fs/hf_libero_full`
- HF datasets cache: `/root/autodl-fs/hf_datasets_cache`
- Temporary files for experiments: `/root/autodl-fs/tmp`

Important disk distinction:

- `/root/autodl-tmp` is a 50 GB XFS volume and was at about 62% used on
  2026-09-18 after cleanup. The project `outputs` directory occupies about
  8.9 GB.
- `/autodl-fs/data` is a 200 GB AutoFS volume and was at about 51% used on
  2026-09-18 after cleanup. The large directories are `hf_libero_full` (about 33 GB),
  `hf_datasets_cache` (about 33 GB), `sac-flow-experiments` (about 27 GB),
  and `sac-flow-archive` (about 18 GB).
- Do not confuse the temporary disk warning with the AutoFS data disk. Always
  run `df -hT` before cleanup.

## 2. How to Start the Remote Environment

```bash
ssh autodl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /root/autodl-tmp/lerobot/src
```

The standard offline environment is:

```bash
export LEROBOT_LIBERO_ROOT=/root/autodl-fs/hf_libero_full
export HF_DATASETS_CACHE=/root/autodl-fs/hf_datasets_cache
export TMPDIR=/root/autodl-fs/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

The scripts already set these variables. For a new run, explicitly set an
output directory on AutoFS whenever the entry point supports it. Do not let a
large checkpoint land on `/root/autodl-tmp` by accident.

## 3. Common Commands

Run these from `/root/autodl-tmp/lerobot/src` after activating `lerobot`.

### GPU smoke

```bash
bash scripts/rlinf_smolvla_libero/gpu_smoke.sh
```

Purpose: two-step GPU and import/checkpoint-path smoke test. It does not save a
checkpoint by default. Set `WANDB_ENABLE=true` only when the smoke metrics are
needed in W&B.

### Short critic/actor-path run

```bash
bash scripts/rlinf_smolvla_libero/short_run_wandb.sh
```

Relevant overrides:

```bash
SAC_FLOW_ACTOR_UPDATES_ENABLED=false \
SAC_FLOW_SAVE_CHECKPOINT=false \
SAC_FLOW_ROOT_CAUSE_DIAGNOSTICS=true \
SAC_FLOW_RESUME_CHECKPOINT=/absolute/path/to/checkpoint_000280 \
bash scripts/rlinf_smolvla_libero/short_run_wandb.sh
```

Defaults include `task_ids=[0]`, `max_train_steps=280`, `max_chunk_steps=1`,
`num_envs=4`, `num_updates_per_step=4`, `batch_size=8`, `replay_capacity=2048`,
`actor_updates_enabled=false`, and `actor_lr=3e-6`.

### Baseline run

```bash
bash scripts/rlinf_smolvla_libero/baseline_run_wandb.sh
```

This is the unchanged-policy/action-path comparison path. It defaults to task
0, 500 collection steps, 4 environments, 16 critic updates per step, and no
checkpoint unless `SAC_FLOW_SAVE_CHECKPOINT=true` is supplied.

### Paired intervention critic-only run

```bash
SAC_FLOW_SEED=0 \
SAC_FLOW_OUTPUT_DIR=/root/autodl-fs/sac-flow-intervention-$(date +%Y%m%d-%H%M%S) \
bash scripts/rlinf_smolvla_libero/intervention_recheck.sh
```

The intervention entry point also accepts `SAC_FLOW_TASK_IDS` and
`SAC_FLOW_NUM_ENVS`, for example:

```bash
SAC_FLOW_TASK_IDS='[3,9]' SAC_FLOW_NUM_ENVS=6 SAC_FLOW_SEED=23 \
SAC_FLOW_OUTPUT_DIR=/autodl-fs/data/sac-flow-experiments/2026-09-18/task3-task9-seed23 \
bash scripts/rlinf_smolvla_libero/intervention_recheck.sh
```

This entry point is the current diagnostic path. Its important defaults are:

- `task_ids=[3,9]`
- `paired_init_states=true`
- `max_train_steps=400`
- `num_envs=6`
- `num_updates_per_step=4`
- `batch_size=8`
- `replay_capacity=8192`
- `actor_updates_enabled=false`
- `save_checkpoint=true`
- `heldout_num_steps=400`, `heldout_seed=2000`
- `critic_intervention_fraction=0.5`
- `critic_intervention_noise_std=0.3`
- `critic_intervention_noise_stds=0.15,0.3,0.45`
- `critic_intervention_pairing=true`
- `critic_pairwise_coef=1.0`
- `critic_pairwise_margin=0.05`
- `critic_pairwise_min_length_gap=5`
- `target_valid_pairs=4` (collection stops early only after four verified pairs)
- `entropy_regularization=false`
- `backup_entropy=false`

The collector compares clean and intervention branches from the same initial
state. A valid pair must be a verified, completed success/failure or otherwise
well-defined length-separated comparison. Pair IDs must include task, run or
collection seed, episode/anchor identity, and branch; a local episode number
alone is not globally unique.

## 4. Current Architecture and Validated Milestones

The current path is a rebuilt one-step action-semantic off-policy actor-critic
for SmolVLA. `max_chunk_steps=1` is intentional: environment execution,
replay, critic targets, and actor actions must all refer to the same 7D action.
The former full-chunk interpretation must not be reintroduced.

Current structural choices:

- train scope is `action_path`; VLM backbone, vision/language encoders, and
  `state_proj` remain frozen for the first baseline;
- entropy and alpha updates are disabled in the interim one-step path;
- critic uses double-Q style estimates and bounded diagnostics;
- trajectory-space KL is wired with a frozen phase-start reference policy,
  default coefficient `0.05`, and W&B metrics `kl_estimate` and `kl_penalty`;
- checkpoint writing was redesigned around temporary write, fsync, ZIP
  validation, atomic replacement, and directory fsync.

Milestones:

1. 2026-07-18: actor learning-rate experiments showed rapid policy
   degradation at `1e-5`; `3e-6` was the most stable short-range choice.
   Trajectory KL was added as an always-on constraint for later actor phases.
2. 2026-08-12: matched baseline/KL and holdout evaluation produced equal
   aggregate counts (`220/240` each across the recorded seed ranges) with
   paired `p=1.0`. KL is evidence for anti-drift/stability, not proof of a
   reproducible improvement over the unchanged baseline.
3. 2026-08-24: the unbounded conservative term
   `logsumexp(Q_random)-Q_replay` was replaced by the bounded hinge
   `relu(Q_random - Q_replay + margin)` with `critic_action_margin=0.1`.
   AutoDL Q scale improved from roughly `[-21, 0.55]` to `[-0.825, 1.186]`.
   This validated numerical stabilization only; it did not authorize actor
   updates.
4. 2026-09-02 to 2026-09-04: paired clean/intervention collection,
   timeout-failure labeling, replay pair filtering, pairwise critic loss, and
   final pairwise diagnostics were implemented. The collector can now produce
   real success/failure pairs. In the latest reported runs there were 1 to 5
   usable pairs depending on the run/checkpoint, pairwise success-transition
   ranking reached `1.0` on the tiny available set, but replay-vs-perturbed
   action ranking was only about `40.6%`. Positive reward coverage was about
   `14/2800 = 0.5%` in one reported run. These are diagnostic signals, not a
   general critic result.
5. 2026-09-18: fixed pair identity at the source. `SACFlowOnlineLoop` now
   receives the collection seed and writes IDs such as
   `task=3:seed=23:pair=0:episode=0`; the final pairwise diagnostic also reports
   pair groups rejected for missing branches, same outcomes, or insufficient
   length gap. Remote source hashes were matched to the local checkout and
   focused AutoDL tests passed: training loop 10, critic diagnostics 7, and
   scripts 4.

Known milestone checkpoints to preserve until a deliberate archive decision:

- `/root/autodl-tmp/lerobot/src/outputs/train/2026-07-24/22-03-58_libero_smolvla/checkpoint_000280`
- `/root/autodl-tmp/lerobot/src/outputs/train/2026-07-24/23-38-04_libero_smolvla/checkpoint_000281`
- `/root/autodl-tmp/lerobot/src/outputs/train/2026-08-12/16-36-15_libero_smolvla/checkpoint_000296`
- `/root/autodl-tmp/lerobot/src/outputs/train/2026-09-04/21-24-29_libero_smolvla/checkpoint_000280`

The final 2026-09-04 run directories are not all milestones. Keep only the
checkpoint whose pair diagnostics are being used, then remove duplicate runs
after checking that no process is active and that W&B or a small summary has
preserved the decision-relevant metrics.

## 5. Current Bottleneck

The critic is numerically stable but not yet demonstrated to be action-aware.
The main risks are:

- too few independent success/failure pairs;
- pair IDs can collide across collection rounds unless the seed/run identity is
  included;
- `success episode`, `positive reward transition`, and `completed episode`
  must remain separate metrics;
- replay action versus perturbed action ranking is currently weak;
- low TD/conservative loss or stable Q range is not an actor-readiness gate;
- actor updates would change the data distribution before the critic has been
  shown to rank actions reliably.

Therefore `actor_updates_enabled` must remain false until the action-ranking,
held-out Bellman, Q-head disagreement, positive-coverage, and cross-task
gradient gates are all reviewed together.

## 6. Assessment of the Proposed Weekly Plan

The previous plan is directionally correct but the order is too optimistic.
Continuing to collect more task-9 pairs is useful only after the pair identity
and episode accounting are corrected, and more of the same intervention data
will not by itself repair a weak action-sensitive critic.

Recommended correction:

1. Fix pair identity and reporting first. Add collection seed/run ID to every
   pair, and log completed episodes, successful episodes, positive-reward
   transitions, valid pairs, pair rejection reasons, and branch outcomes.
2. Run one bounded collector on task 3 and task 9. Task 3 is included because
   the 2026-09-04 run already produced a real clean-success/intervention-timeout
   failure pair there; task 9 remains the intended hard task. Stop after a
   small predefined budget or once enough independent pairs are obtained.
3. Re-run critic-only training with checkpoint saving and a held-out split.
   Evaluate success-vs-failure ranking, replay-vs-perturbed ranking,
   margin-violation fraction, Q-head span, held-out Bellman error, and
   positive coverage. Do not enable actor updates in this stage.
4. Only if the single-task action gate is stable, migrate to 2-3 tasks for
   critic transfer and cross-task actor-gradient direction checks.
5. Only after those gates pass, run a tiny actor update from a preserved
   checkpoint with KL enabled and rollback available. Then perform the final
   paired evaluation with independent LIBERO seeds for unchanged baseline,
   critic-guided actor, and the untouched baseline/control.

Suggested stop conditions for the next bounded decision run:

- fewer than 8-10 independent verified pairs across the selected tasks: do not
  interpret ranking as generalization;
- replay-vs-perturbed ranking remains near chance: stop collecting and repair
  intervention/action sensitivity rather than increasing training length;
- unstable held-out Bellman error, exploding Q-head span, or inconsistent
  cross-task gradients: keep actor disabled;
- all gates pass: authorize only a small actor update, not a long run.

Validated seed-aware collection on 2026-09-18:

- Checkpoint:
  `/autodl-fs/data/sac-flow-experiments/2026-09-18/task3-task9-seed23-032152/checkpoint_000280`
- Configuration: `task_ids=[3,9]`, `num_envs=6` per task, collection seed
  `23`, held-out seed `2000`, 280 collection steps, 3360 transitions,
  actor disabled, checkpoint enabled.
- Collection: 15 completed episodes, 6 successful, 9 truncated, 6 positive
  reward transitions. Clean success rate was `4/8=0.50`; intervention success
  rate was `2/7=0.286`.
- Pair accounting: 8 pair groups, 1 missing branch, 4 same-outcome groups,
  2 insufficient-length-gap groups, and 1 usable pair. All 3360 replay items
  contained `seed=23` in their pair ID; no old-format IDs remained.
- Critic gate: pairwise ranking accuracy `1.0` on the single usable pair, but
  replay-vs-perturbed ranking was only `0.5625`, with pairwise margin violation
  fraction `1.0`. This is not actor-ready evidence.
- Held-out Bellman MSE was `0.000837`, absolute error `0.01882`, and the
  two-task actor-gradient pairwise cosine was `0.615` with no negative pairs.
  These support numerical stability and lack of an obvious cross-task gradient
  conflict, but do not establish reliable action ranking.
- Checkpoint ZIP validation passed. Actor updates remain disabled.

Validated layered-noise calibration on 2026-09-18:

- Command: `SAC_FLOW_SEED=23 SAC_FLOW_TARGET_PAIRS=4 SAC_FLOW_MAX_TRAIN_STEPS=400 SAC_FLOW_HELDOUT_NUM_STEPS=400 SAC_FLOW_OUTPUT_DIR=/autodl-fs/data/sac-flow-experiments/2026-09-18/task3-task9-seed23-calibration bash scripts/rlinf_smolvla_libero/intervention_recheck.sh`
- Source changes: per-pair intervention noise (`0.15, 0.30, 0.45`), noise-aware
  pair IDs, verified-pair counting and bounded early stop, noise-bucket
  coverage metrics, CLI overrides, and focused tests.
- Corrected checkpoint: `/autodl-fs/data/sac-flow-experiments/2026-09-18/task3-task9-seed23-calibration/checkpoint_000400`
  (`sac_flow_state.pt` about 7.6 GB). The first launch used an incorrect
  `/root/autodl-fs/data` symlink-expanded path and old script; it was terminated
  before analysis and its output was removed. The approved configuration was
  then synchronized to the real checkout and verified from `/proc/<pid>/cmdline`.
- Configuration: `task_ids=[3,9]`, `num_envs=6`, collection seed `23`,
  `max_train_steps=400`, held-out steps `400`, `actor_updates_enabled=false`,
  `critic_pairwise_min_length_gap=5`, target valid pairs `4`.
- Training collection: 4,800 transitions, 14 completed episodes, 5 successes,
  9 truncated episodes, and 5 positive-reward transitions. Clean branches had
  8 completed/4 successful episodes; intervention branches had 6 completed/1
  successful episode. Noise buckets recorded 0.15: 4 completed/0 successful,
  0.30: 5/2, and 0.45: 5/3.
- Pair accounting: 8 pair groups, 2 missing branches, 4 same-outcome groups,
  1 insufficient-length-gap group, and 1 verified pair. The target of 4 was
  not reached, so the run used the 400-step ceiling.
- Critic gate: verified pairwise ranking `1.0` on one pair; replay-vs-perturbed
  ranking `0.5625`; replay-minus-perturbed Q gap `0.000275`; Q-head span mean
  `0.0366`, max `0.0597`; held-out Bellman MSE `0.002257`, absolute error
  `0.03325`, and held-out Q-head span `0.03799`. Cross-task actor-gradient
  cosine was `0.7375` with no negative task pair. The root-cause probe showed
  action-to-observation effect ratio `8.50`, but this did not translate into
  reliable replay-action ranking.
- Decision: `INCONCLUSIVE/FAIL for actor readiness`, not `PASS`. Numerical
  stability, held-out Bellman behavior, and gradient compatibility are useful
  evidence, but one valid pair and near-chance action ranking are insufficient.
  Keep actor updates disabled. The next intervention should improve verified
  pair yield and action-sensitive ranking; do not increase training length or
  enable actor updates from this checkpoint.
- Focused AutoDL tests passed after the correct sync: 8 intervention, 16 replay,
  7 diagnostics, 10 training-loop, and 21 entry tests. Local compile, shell
  syntax, and `git diff --check` passed; local Torch tests were unavailable
  because the host Python has no Torch.

## 7. Cleanup Protocol After Every Experiment

First inspect:

```bash
ssh autodl 'df -hT /root/autodl-tmp /autodl-fs/data; ps -eo pid,etime,cmd --sort=-%mem | head -40'
```

Then inspect large files:

```bash
ssh autodl 'du -xhd1 /root/autodl-tmp/lerobot/src /root/.cache /autodl-fs/data | sort -h | tail -40'
ssh autodl 'find /root/autodl-tmp/lerobot/src/outputs -type f -printf "%s %p\n" | sort -nr | head -40'
```

Cleanup rules:

- Never delete a live run. Confirm no training process uses the directory.
- Preserve only validated milestone checkpoints and the summary needed to
  reproduce the conclusion.
- Delete duplicate/non-milestone `outputs/train/*` directories from the 50 GB
  temporary disk after validation.
- Remove stale `/root/autodl-tmp/autodl-tmp/.cache`, `.ipynb_checkpoints`, and
  orphaned temporary files when they are not in use.
- Do not delete `/root/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero`
  or the SmolVLM model cache blindly; those are needed for offline startup.
- Do not delete `hf_libero_full`, `hf_datasets_cache`, or the two large
  AutoFS archive/experiment trees without an explicit inventory and archive
  decision. They are data, not automatically disposable cache.
- After cleanup, run `df -hT` again and record the before/after usage in this
  file.

## 8. Update Log

- 2026-09-18: reconstructed history from pinned tasks 0718-0904 and the
  0904 continuation; audited local checkout and AutoDL storage; recorded the
  current critic/action-ranking bottleneck and the corrected execution order.
- 2026-09-18: added project startup commands, offline variables, experiment
  entry points, checkpoint retention rules, and cleanup protocol.
- 2026-09-18: AutoDL cleanup completed after confirming no training process was
  active. Removed four non-milestone 2026-09-04 output directories totaling
  about 14.7 GB, plus stale `/root/autodl-tmp/autodl-tmp/.cache`, project
  `.ipynb_checkpoints`, W&B/pip cache directories, and stale HF lock files.
  Preserved the four milestone checkpoints listed above. `/root/autodl-tmp`
  fell from 91% to 62% used; `/autodl-fs/data` remained at 56% used.
- 2026-09-18: implemented source-level seed-aware pair IDs and rejection-reason
  diagnostics; synchronized and hash-verified the remote source; ran the
  approved task-3/task-9 bounded collection; validated the checkpoint; removed
  two pre-fix no-seed checkpoints totaling about 10.8 GB. AutoFS fell from 64%
  to 59% used and `/root/autodl-tmp` remained at 62%.
- 2026-09-18: implemented layered intervention noise, noise-aware pair IDs,
  verified-pair early stopping, and noise-bucket diagnostics. The approved
  calibration produced one verified pair and did not pass the action gate.
  Preserved its checkpoint, moved it out of the accidental nested
  `/autodl-fs/data/data` path, removed three confirmed non-milestone diagnostic
  checkpoints totaling about 19 GB plus stale notebook checkpoints/logs, and
  rechecked usage at `/root/autodl-tmp` 62% and `/autodl-fs/data` 51%.
