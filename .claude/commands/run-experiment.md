Commit the current changes to predict.py on a dedicated experiment branch, run the backtest and prediction, log results to BACKTESTS.md, then return to main.

## Steps

1. Ask the user for:
   - `run_name`: short slug for this experiment (e.g. `draw_features`, `recency_filter`)
   - `parent_run`: which experiment this builds on — use the full run name from BACKTESTS.md (e.g. `baseline_20260626`). Default to the most recent row in BACKTESTS.md.
   - `description`: one sentence describing what changed vs the parent
   - `notes`: brief note for the BACKTESTS.md table

2. Look up the parent's accuracy and log-loss from BACKTESTS.md so you can compute deltas later.

3. Determine the branch to start from:
   - If `exp/<parent_run>` exists: `git checkout exp/<parent_run>`
   - Otherwise: `git checkout main`
   Then create the experiment branch:
   ```
   git checkout -b exp/<run_name>
   ```

4. Stage and commit predict.py:
   ```
   git add predict.py
   git commit -m "<description>"
   ```

5. Run prediction with the run name and parent:
   ```
   uv run predict.py --run-name <run_name> --parent <parent_run>
   ```

6. Parse the backtest output for accuracy, log-loss, and n_matches. Get the short commit hash via `git rev-parse --short HEAD`. Compute deltas vs parent:
   - `Δacc` = this accuracy − parent accuracy (e.g. `+2%` or `-4%`)
   - `Δloss` = this log-loss − parent log-loss (e.g. `+0.006` or `-0.008`)

7. Append a new row to the table in `BACKTESTS.md`:
   ```
   | <run_name>_<date> | <parent_run> | <accuracy> | <Δacc> | <log-loss> | <Δloss> | <n_matches> | <commit> | <today's date> | <notes> |
   ```

8. Commit BACKTESTS.md on the experiment branch:
   ```
   git add BACKTESTS.md
   git commit -m "Log backtest results for <run_name>"
   ```

9. Switch back to main and bring BACKTESTS.md across:
   ```
   git checkout main
   git checkout exp/<run_name> -- BACKTESTS.md
   git commit -m "Update BACKTESTS.md with results for <run_name>"
   ```

10. Show the user the backtest results and confirm:
    - Branch `exp/<run_name>` created and committed (branched from `exp/<parent_run>` or main)
    - `experiments.jsonl` logged with parent field
    - `BACKTESTS.md` updated on both the experiment branch and main

11. If this experiment beats the current best (highest accuracy or lowest log-loss), ask the user if they want to merge it into main:
    ```
    git merge exp/<run_name>
    ```
