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
    args = parser.parse_args()

    df    = load_data(refresh=args.refresh)
    feats = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    future = feats[feats["home_score"].isna() & (feats["date"] >= TODAY)].sort_values("date")

    if not len(future):
        print("No upcoming fixtures — run with --refresh to fetch latest data.")
        return

    today_str = datetime.now().strftime("%Y%m%d")
    filename  = f"predictions/baseline_{today_str}.csv"
    os.makedirs("predictions", exist_ok=True)

    model     = train(played.tail(MAX_TRAIN))
    proba_df  = predict_proba(model, future[FEATURES].values)
    label_arr = np.array(["home_win", "draw", "away_win"])

    out = future[["date", "home_team", "away_team"]].copy()
    out["predicted"]  = label_arr[proba_df[["p_home_win", "p_draw", "p_away_win"]].values.argmax(1)]
    out["p_home_win"] = proba_df["p_home_win"].round(3).values
    out["p_draw"]     = proba_df["p_draw"].round(3).values
    out["p_away_win"] = proba_df["p_away_win"].round(3).values

    out.to_csv(filename, index=False)
    print(f"\n{len(out)} fixture predictions -> {filename}\n")
    for r in out.itertuples():
        print(f"  {r.date.date()}  {r.home_team:>20} vs {r.away_team:<20}  "
              f"-> {r.predicted:<9}  H {r.p_home_win:4.0%} | D {r.p_draw:4.0%} | A {r.p_away_win:4.0%}")


if __name__ == "__main__":
    main()
