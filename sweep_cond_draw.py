"""One-off sweep: conditional draw multiplier draw_k = 1 + b*exp(-|elo_diff|/s).

Computes raw 3-class probabilities once per test set (no retrain per grid point),
then applies the conditional boost and reports log-loss on BT1/BT2/BT3.
BT2 (WC22 knockout) is the competition-relevant target.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from features import (
    load_data, wc_matches, wc_group_stage, wc_knockout, wc_group_rounds, MAX_TRAIN,
)
from predict import train, predict_proba, build_features, FEATURES, TRAIN_START

WC2022_START = pd.Timestamp("2022-11-20")
WC2026_START = pd.Timestamp("2026-06-11")


def raw_block(test, model):
    """Return (raw_hda [n,3], elo_diff [n], outcomes) for a test set."""
    proba = predict_proba(model, test[FEATURES].values)
    proba = proba / proba.sum(axis=1).values[:, None]
    hda = proba[["p_home_win", "p_draw", "p_away_win"]].values
    return hda, test["elo_diff"].values, test["outcome"].values


def ll_of(hda, outcomes):
    lex = hda[:, [2, 1, 0]]  # away, draw, home
    return log_loss(outcomes, lex, labels=["away_win", "draw", "home_win"])


def apply_cond(hda, elo_diff, b, s):
    k = 1.0 + b * np.exp(-np.abs(elo_diff) / s)
    out = hda.copy()
    out[:, 1] *= k
    out /= out.sum(axis=1, keepdims=True)
    return out


def apply_flat(hda, k):
    out = hda.copy()
    out[:, 1] *= k
    out /= out.sum(axis=1, keepdims=True)
    return out


def main():
    df = load_data()
    feats = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]

    wc22 = wc_matches(feats, 2022)
    model22 = train(played[played["date"] < WC2022_START].tail(MAX_TRAIN))
    bt1 = raw_block(wc_group_stage(wc22), model22)
    bt2 = raw_block(wc_knockout(wc22), model22)

    wc26 = wc_matches(feats, 2026)
    model26 = train(played[played["date"] < WC2026_START].tail(MAX_TRAIN))
    bt3 = raw_block(wc_group_rounds(wc26, max_round=2), model26)

    blocks = {"BT1": bt1, "BT2": bt2, "BT3": bt3}

    print("elo_diff |range| per test set:")
    for name, (hda, ed, oc) in blocks.items():
        print(f"  {name}: |elo_diff| min={np.abs(ed).min():.0f} "
              f"med={np.median(np.abs(ed)):.0f} max={np.abs(ed).max():.0f}")

    print("\nRaw (draw_k=1.0):")
    for name, (hda, ed, oc) in blocks.items():
        print(f"  {name} ll={ll_of(hda, oc):.4f}")

    print("\nFlat baseline draw_k=1.2 (current production):")
    for name, (hda, ed, oc) in blocks.items():
        print(f"  {name} ll={ll_of(apply_flat(hda, 1.2), oc):.4f}")

    print("\nConditional sweep  draw_k = 1 + b*exp(-|elo_diff|/s):")
    print(f"{'b':>5} {'s':>6} | {'BT1':>8} {'BT2':>8} {'BT3':>8}   (BT2 is target)")
    best = None
    for b in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        for s in [50, 100, 150, 200, 300, 400]:
            lls = {name: ll_of(apply_cond(hda, ed, b, s), oc)
                   for name, (hda, ed, oc) in blocks.items()}
            marker = ""
            if best is None or lls["BT2"] < best[0]:
                best = (lls["BT2"], b, s, lls)
                marker = "  <- best BT2"
            print(f"{b:>5.1f} {s:>6.0f} | {lls['BT1']:>8.4f} {lls['BT2']:>8.4f} "
                  f"{lls['BT3']:>8.4f}{marker}")

    print(f"\nBest BT2: b={best[1]}, s={best[2]}  "
          f"BT1={best[3]['BT1']:.4f} BT2={best[3]['BT2']:.4f} BT3={best[3]['BT3']:.4f}")


if __name__ == "__main__":
    main()
