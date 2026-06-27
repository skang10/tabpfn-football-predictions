---
name: run-experiment
description: Run TabPFN football prediction experiments in this repo. Use when Codex needs to follow the repo-local experiment workflow after editing predict.py: create a branch, run uv run backtest.py --run-name, parse BT1/BT2/BT3 metrics, and update BACKTESTS.md.
---

# Run Experiment

Follow `.codex/commands/run-experiment.md` as the source of truth. Read that file before taking action, then execute its workflow exactly.

If `uv run` fails because of sandboxed cache access, rerun it with escalation.
