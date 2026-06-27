Run a new experiment: create a branch, implement the model in predict.py, run all three backtests, and update BACKTESTS.md.

## Architecture

- `features.py` — shared data loading and feature engineering (do not modify per-experiment)
- `backtest.py` — model-agnostic evaluator; runs BT1/BT2/BT3 (do not modify per-experiment)
- `predict.py` — **the only file that changes per experiment**; must expose the standard interface:
  - `train(pool) → model`
  - `predict_proba(model, X) → DataFrame[p_home_win, p_draw, p_away_win]`
  - `FEATURES`, `TRAIN_START`, `build_features` (may override the defaults from features.py)

## Backtests

| ID | Test set | Matches | Training cutoff |
|----|----------|---------|-----------------|
| BT1 | WC 2022 group stage | 48 | 2022-11-20 |
| BT2 | WC 2022 knockout | 16 | 2022-11-20 |
| BT3 | WC 2026 group stage rounds 1–2 | 48 | 2026-06-11 |

Primary metric: **multi-class log-loss** (lower is better).

## Steps

1. Ask the user for:
   - `run_name`: short slug (e.g. `recency_2018`, `draw_boost`)
   - `parent_branch`: which branch to start from — default `main` (baseline); use an experiment branch if building on a prior experiment
   - `description`: one sentence describing what changed
   - `notes`: brief note for the BACKTESTS.md row

2. Create the experiment branch:
   ```
   git checkout <parent_branch>
   git checkout -b exp/<run_name>
   ```

3. If `features.py` or `backtest.py` are missing on this branch, bring them from main:
   ```
   git checkout main -- features.py backtest.py
   ```

4. Implement the model changes in `predict.py`. Ensure the standard interface is satisfied.

5. Commit:
   ```
   git add predict.py features.py backtest.py
   git commit -m "<description>"
   ```

6. Run all three backtests:
   ```
   uv run backtest.py --run-name <run_name>
   ```

7. Parse the output for BT1, BT2, BT3 log-loss and accuracy. Get the commit hash:
   ```
   git rev-parse --short HEAD
   ```

8. Update BACKTESTS.md: append one row to each of the three BT tables.
   - Use `**bold**` on the log-loss value if it is the best in that column.
   - Use `<u>underline</u>` on the accuracy value if it is the best in that column.

9. Commit BACKTESTS.md on the experiment branch:
   ```
   git add BACKTESTS.md
   git commit -m "Log backtest results for <run_name>"
   ```

10. Bring BACKTESTS.md back to main:
    ```
    git checkout main
    git checkout exp/<run_name> -- BACKTESTS.md
    git commit -m "BACKTESTS.md: add <run_name> results"
    ```

11. Report BT1/BT2/BT3 results to the user and confirm the branch and log entry were created.
