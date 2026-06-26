---
name: run-experiment
description: Run TabPFN football prediction experiments in this repo. Use when Codex needs to follow the repo-local experiment command after editing predict.py: commit the experiment, execute uv run predict.py --run-name, parse metrics, and update BACKTESTS.md.
---

# Run Experiment

Follow `.codex/commands/run-experiment.md` as the source of truth. Read that file before taking action, then execute its workflow exactly.

If `uv run` fails because of sandboxed cache access, rerun it with escalation.
