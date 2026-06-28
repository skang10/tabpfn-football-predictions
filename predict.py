"""Baseline model (3-class TabPFN classifier) + forward predictions.

Exposes the standard interface used by backtest.py:
    train(pool)          → fitted model
    predict_proba(model, X) → DataFrame[p_home_win, p_draw, p_away_win]

Run as a script to generate forward predictions for upcoming fixtures:
    uv run predict.py [--refresh]
"""
import argparse
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download dataset")
    parser.add_argument("--date", default=None,
                        help="Predict fixtures on or after this date (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()

    from_date = pd.Timestamp(args.date) if args.date else TODAY

    df    = load_data(refresh=args.refresh)
    feats = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    future = feats[feats["home_score"].isna() & (feats["date"] >= from_date)].sort_values("date")

    if not len(future):
        print("No upcoming fixtures — run with --refresh to fetch latest data.")
        return

    today_str = datetime.now().strftime("%Y%m%d")
    filename  = f"predictions/submission_{today_str}.csv"
    os.makedirs("predictions", exist_ok=True)

    model     = train(played.tail(MAX_TRAIN))
    proba_df  = predict_proba(model, future[FEATURES].values)
    label_arr = np.array(["home_win", "draw", "away_win"])
    predicted = label_arr[proba_df[["p_home_win", "p_draw", "p_away_win"]].values.argmax(1)]

    # Submission format: date, home_team, away_team, p_home_win, p_draw, p_away_win (2 dp, sums to 1)
    ph_raw = proba_df["p_home_win"].values
    pd_raw = proba_df["p_draw"].values
    out = future[["date", "home_team", "away_team"]].copy()
    out["p_home_win"] = ph_raw.round(2)
    out["p_draw"]     = pd_raw.round(2)
    out["p_away_win"] = (1.0 - ph_raw - pd_raw).round(2)

    out.to_csv(filename, index=False)
    print(f"\n{len(out)} fixture predictions -> {filename}\n")
    for r, pred in zip(out.itertuples(), predicted):
        print(f"  {r.date.date()}  {r.home_team:>22} vs {r.away_team:<22}"
              f"  -> {pred:<9}  H {r.p_home_win:.0%} | D {r.p_draw:.0%} | A {r.p_away_win:.0%}")


if __name__ == "__main__":
    main()
