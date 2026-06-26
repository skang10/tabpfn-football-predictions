Commit the current changes to predict.py with a description of what changed, then run the backtest and prediction, and log results to BACKTESTS.md.

## Steps

1. Ask the user for:
   - `run_name`: short slug for this experiment (e.g. `crowd_v1`, `no_form10`)
   - `description`: one sentence describing what changed (e.g. "add crowd support for WC2026 venues")
   - `notes`: brief note for the BACKTESTS.md table describing what this experiment tests (e.g. "Continuous crowd factors for WC2026 venues")

2. Stage and commit predict.py:
   ```
   git add predict.py
   git commit -m "<description>"
   ```

3. Run prediction with the run name:
   ```
   uv run predict.py --run-name <run_name>
   ```

4. Parse the backtest output for accuracy, log-loss, and n_matches. Get the short commit hash from the run output or via `git rev-parse --short HEAD`.

5. Append a new row to the table in `BACKTESTS.md`:
   ```
   | <run_name> | <accuracy> | <log-loss> | <n_matches> | <commit> | <today's date> | <notes> |
   ```

6. Show the user the backtest results and confirm both `experiments.jsonl` and `BACKTESTS.md` were updated.
