Commit the current changes to predict.py on a dedicated experiment branch, run the backtest and prediction, log results to BACKTESTS.md, then return to main.

## Steps

1. Ask the user for:
   - `run_name`: short slug for this experiment (e.g. `draw_features`, `recency_filter`)
   - `description`: one sentence describing what changed
   - `notes`: brief note for the BACKTESTS.md table

2. Create and switch to an experiment branch from main:
   ```
   git checkout main
   git checkout -b exp/<run_name>
   ```

3. Stage and commit predict.py:
   ```
   git add predict.py
   git commit -m "<description>"
   ```

4. Run prediction with the run name:
   ```
   uv run predict.py --run-name <run_name>
   ```

5. Parse the backtest output for accuracy, log-loss, and n_matches. Get the short commit hash via `git rev-parse --short HEAD`.

6. Append a new row to the table in `BACKTESTS.md`:
   ```
   | <run_name>_<date> | <accuracy> | <log-loss> | <n_matches> | <commit> | <today's date> | <notes> |
   ```

7. Commit BACKTESTS.md on the experiment branch:
   ```
   git add BACKTESTS.md
   git commit -m "Log backtest results for <run_name>"
   ```

8. Switch back to main:
   ```
   git checkout main
   ```

9. Show the user the backtest results and confirm:
   - Branch `exp/<run_name>` created and committed
   - `experiments.jsonl` logged
   - `BACKTESTS.md` updated on both the experiment branch and main

10. If this experiment beats the current best, ask the user if they want to merge it into main:
    ```
    git merge exp/<run_name>
    ```
