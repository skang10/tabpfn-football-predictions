"""Baseline model (3-class TabPFN classifier) + forward predictions.

Exposes the standard interface used by backtest.py:
    train(pool)          → fitted model
    predict_proba(model, X) → DataFrame[p_home_win, p_draw, p_away_win]

Run as a script to generate forward predictions for upcoming fixtures:
    uv run predict.py [--refresh]
"""
import argparse
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from tabpfn_client import TabPFNClassifier

from features import (
    load_data, build_features,
    FEATURES, TRAIN_START, MAX_TRAIN, TODAY,
    wc_matches,
)


def train(pool):
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    return clf


def predict_proba(model, X):
    """Return a DataFrame with columns p_home_win, p_draw, p_away_win (rows sum to 1)."""
    proba = model.predict_proba(X)
    proba = proba / proba.sum(axis=1, keepdims=True)
    classes = list(model.classes_)
    return pd.DataFrame({
        "p_home_win": proba[:, classes.index("home_win")],
        "p_draw":     proba[:, classes.index("draw")],
        "p_away_win": proba[:, classes.index("away_win")],
    })


def _round_probs_2dp(probs):
    """Round H/D/A probabilities to 2 decimals while preserving a 1.00 row sum."""
    rounded = []
    for row in probs:
        cents = np.floor(row * 100).astype(int)
        remainder = int(100 - cents.sum())
        fractions = row * 100 - cents
        for idx in np.argsort(-fractions)[:remainder]:
            cents[idx] += 1

        zero_idx = np.where(cents == 0)[0]
        for idx in zero_idx:
            donor = int(np.argmax(cents))
            if cents[donor] <= 1:
                break
            cents[idx] = 1
            cents[donor] -= 1

        rounded.append(cents / 100)
    return np.array(rounded)


def _git_branch():
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        return branch.replace("/", "_")
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh",    action="store_true", help="Re-download dataset")
    parser.add_argument("--date",       default=None,
                        help="Predict fixtures on or after this date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--draw-scale", type=float, default=1.0,
                        help="Multiply p_draw by this factor then renormalize (default 1.0). "
                             "Use 1.2 for knockout-stage submissions.")
    parser.add_argument("--output-dir", default="predictions",
                        help="Directory for generated submission CSVs (default: predictions).")
    args = parser.parse_args()

    from_date = pd.Timestamp(args.date) if args.date else TODAY

    df    = load_data(refresh=args.refresh)
    feats = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    # Keep the fixture order from results.csv/load_data instead of re-sorting here.
    future = feats[feats["home_score"].isna() & (feats["date"] >= from_date)]

    if not len(future):
        print("No upcoming fixtures — run with --refresh to fetch latest data.")
        return

    today_str  = datetime.now().strftime("%Y%m%d")
    scale_tag  = f"_ds{args.draw_scale:.1f}".replace(".", "") if args.draw_scale != 1.0 else ""
    filename   = os.path.join(args.output_dir, f"submission_{_git_branch()}{scale_tag}_{today_str}.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    model     = train(played.tail(MAX_TRAIN))
    proba_df  = predict_proba(model, future[FEATURES].values)

    # Apply draw scaling then renormalize
    hda = proba_df[["p_home_win", "p_draw", "p_away_win"]].values.copy()
    if args.draw_scale != 1.0:
        hda[:, 1] *= args.draw_scale
        hda /= hda.sum(axis=1, keepdims=True)

    label_arr = np.array(["home_win", "draw", "away_win"])
    predicted = label_arr[hda.argmax(1)]

    # Submission format: date, home_team, away_team, p_home_win, p_draw, p_away_win (2 dp, sums to 1)
    rounded_hda = _round_probs_2dp(hda)
    ph_raw, pd_raw, pa_raw = rounded_hda[:, 0], rounded_hda[:, 1], rounded_hda[:, 2]
    out = future[["date", "home_team", "away_team"]].copy()
    out["p_home_win"] = ph_raw
    out["p_draw"]     = pd_raw
    out["p_away_win"] = pa_raw

    out.to_csv(filename, index=False, float_format="%.2f")
    print(f"\n{len(out)} fixture predictions -> {filename}\n")
    for r, pred in zip(out.itertuples(), predicted):
        print(f"  {r.date.date()}  {r.home_team:>22} vs {r.away_team:<22}"
              f"  -> {pred:<9}  H {r.p_home_win:.0%} | D {r.p_draw:.0%} | A {r.p_away_win:.0%}")


if __name__ == "__main__":
    main()
