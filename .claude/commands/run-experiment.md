Commit the current changes to predict.py with a description of what changed, then run the backtest and prediction.

## Steps

1. Ask the user for:
   - `run_name`: short slug for this experiment (e.g. `crowd_v1`, `no_form10`)
   - `description`: one sentence describing what changed (e.g. "add crowd support for WC2026 venues")

2. Stage and commit predict.py:
   ```
   git add predict.py
   git commit -m "<description>"
   ```

3. Run prediction with the run name:
   ```
   uv run predict.py --run-name <run_name>
   ```

4. Show the user the backtest results printed to stdout and confirm the experiment was logged to `experiments.jsonl`.
