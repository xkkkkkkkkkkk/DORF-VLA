# DORF-VLA Project Instructions

Before any work in this repository, read `/Users/yikai/Documents/DORF-VLA/memory.md`.
Treat it as the current project state and historical decision record.

After any modification to source code, configuration, scripts, tests, remote
experiment state, or checkpoint state:

1. Update `memory.md` in the same task.
2. Record the date, files or remote paths changed, the exact command used,
   important parameters, validation results, experiment metrics, and any
   remaining blocker.
3. Keep the update factual. Separate validated evidence from hypotheses.
4. Preserve unrelated user changes in the worktree. Do not reset or discard
   them.
5. Before starting an AutoDL run, check both `/root/autodl-tmp` and
   `/autodl-fs/data` with `df -hT`, confirm no conflicting process is running,
   and choose an output path deliberately.
6. After every AutoDL run, clean only confirmed non-milestone checkpoints,
   logs, temporary files, and caches. Recheck disk usage and record it in
   `memory.md`.

For DORF-VLA experiments, keep the following gates explicit:

- actor updates remain disabled while critic action sensitivity is unverified;
- stable Q scale or low critic loss alone never authorizes actor updates;
- distinguish success episodes, positive-reward transitions, and completed
  episodes;
- pair IDs must include collection seed/run identity;
- use bounded, decision-relevant tests and preserve independent holdout seeds;
- interpret KL as an anti-drift constraint unless a matched independent
  comparison demonstrates improvement over the unchanged baseline.
