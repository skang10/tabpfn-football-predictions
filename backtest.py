"""Model-agnostic backtester.

Imports train() and predict_proba() from predict.py (whichever branch you are on),
then evaluates on three fixed test sets:
  BT1 — WC 2022 group stage   (48 matches, model trained on pre-WC2022 data)
  BT2 — WC 2022 knockout      (16 matches, same model as BT1)
  BT3 — WC 2026 group rounds 1-2 (48 matches, model trained on pre-WC2026 data)

Usage:
    uv run backtest.py [--refresh] [--run-name <label>]
"""
import argparse
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from features import (
    load_data,
    MAX_TRAIN,
    wc_matches, wc_group_stage, wc_knockout, wc_group_rounds,
)
# predict.py is the single source of truth per branch:
# it may override FEATURES, TRAIN_START, and build_features
from predict import train, predict_proba, FEATURES, TRAIN_START, build_features
from analyze_backtest import (
    print_summary, print_diagnostics,
    plot_logloss_grouped, plot_logloss_heatmap, plot_draw_distribution,
)

EXPERIMENTS_LOG = "experiments.jsonl"
WC2022_START    = pd.Timestamp("2022-11-20")
WC2026_START    = pd.Timestamp("2026-06-11")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_branch():
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        return branch.split("/")[-1]
    except Exception:
        return "unknown"


def _bt_entry(result):
    """Convert an evaluate() result tuple to a jsonl-compatible dict."""
    if result is None:
        return None
    ll, acc, pm = result
    return {
        "log_loss":  round(ll, 4),
        "accuracy":  round(acc, 4),
        "n_matches": len(pm),
        "per_match": (pm.assign(date=pm["date"].dt.strftime("%Y-%m-%d"))
                        .to_dict(orient="records")),
    }


def log_experiment(run_name, bt1, bt2, bt3):
    """Append BT1/BT2/BT3 results to experiments.jsonl."""
    entry = {
        "run":       run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit":    git_commit(),
        "features":  FEATURES,
        "bt1":       _bt_entry(bt1),
        "bt2":       _bt_entry(bt2),
        "bt3":       _bt_entry(bt3),
    }
    with open(EXPERIMENTS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Logged to {EXPERIMENTS_LOG} (run='{run_name}', commit={entry['commit']})")


def evaluate(label, test, model):
    """Evaluate model on test matches; print metrics and return (ll, acc, per_match_df)."""
    if not len(test):
        print(f"\n{label}: no matches found")
        return None

    X = test[FEATURES].values
    proba_df = predict_proba(model, X)
    proba_df = proba_df / proba_df.sum(axis=1).values[:, None]  # ensure sum-to-1

    # sklearn log_loss needs columns in lexicographic label order
    proba_lex = proba_df[["p_away_win", "p_draw", "p_home_win"]].values
    ll  = log_loss(test["outcome"], proba_lex, labels=["away_win", "draw", "home_win"])

    label_arr = np.array(["home_win", "draw", "away_win"])
    proba_hda = proba_df[["p_home_win", "p_draw", "p_away_win"]].values
    pred = label_arr[proba_hda.argmax(1)]
    acc  = accuracy_score(test["outcome"], pred)

    draws_actual = (test["outcome"] == "draw").sum()
    draws_pred   = (pred == "draw").sum()

    print(f"\n── {label} ({len(test)} matches) ──")
    print(f"  log-loss  {ll:.4f}")
    print(f"  accuracy  {acc:.1%}")
    print(f"  draws: {draws_actual} actual / {draws_pred} predicted")

    pm = test[["date", "home_team", "away_team", "outcome"]].copy()
    pm["predicted"]  = pred
    pm["p_home_win"] = proba_df["p_home_win"].round(3).values
    pm["p_draw"]     = proba_df["p_draw"].round(3).values
    pm["p_away_win"] = proba_df["p_away_win"].round(3).values
    pm["correct"]    = pm["outcome"] == pm["predicted"]
    return ll, acc, pm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh",  action="store_true", help="Re-download dataset")
    parser.add_argument("--run-name", default=None, help="Label for experiments.jsonl")
    args = parser.parse_args()

    df = load_data(refresh=args.refresh)
    print(f"Latest game in dataset: {df['date'].max().date()}")

    feats  = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]

    today_str = datetime.now().strftime("%Y%m%d")
    run_label = args.run_name or git_branch()
    slug      = f"{run_label}_{today_str}"

    bt1 = bt2 = bt3 = None

    # ── BT1 & BT2: WC 2022 ────────────────────────────────────────────────────
    wc22_all = wc_matches(feats, 2022)
    if len(wc22_all):
        pool22 = played[played["date"] < WC2022_START].tail(MAX_TRAIN)
        print(f"\nTraining on {len(pool22)} matches (pre-WC2022) ...")
        model22 = train(pool22)
        bt1 = evaluate("BT1 — WC 2022 group stage", wc_group_stage(wc22_all), model22)
        bt2 = evaluate("BT2 — WC 2022 knockout",    wc_knockout(wc22_all),    model22)
    else:
        print("\nNo WC2022 matches found (try --refresh).")

    # ── BT3: WC 2026 group rounds 1-2 ─────────────────────────────────────────
    wc26_all = wc_matches(feats, 2026)
    wc26_r12 = wc_group_rounds(wc26_all, max_round=2)
    if len(wc26_r12):
        pool26 = played[played["date"] < WC2026_START].tail(MAX_TRAIN)
        print(f"\nTraining on {len(pool26)} matches (pre-WC2026) ...")
        model26 = train(pool26)
        bt3 = evaluate("BT3 — WC 2026 group rounds 1-2", wc26_r12, model26)
    else:
        print("\nNo WC2026 group-stage results yet (try --refresh).")

    # ── Log ───────────────────────────────────────────────────────────────────
    if args.run_name:
        log_experiment(slug, bt1, bt2, bt3)
    else:
        print("\n(pass --run-name <label> to log this experiment)")

    # ── Analyze ───────────────────────────────────────────────────────────────
    exp = {
        "run":     slug,
        "commit":  git_commit(),
        "bt1":     _bt_entry(bt1),
        "bt2":     _bt_entry(bt2),
        "bt3":     _bt_entry(bt3),
    }
    print()
    print_summary([exp])
    print_diagnostics([exp])
    plot_logloss_grouped([exp])
    plot_logloss_heatmap([exp])
    plot_draw_distribution([exp])


if __name__ == "__main__":
    main()
