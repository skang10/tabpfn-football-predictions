"""Analyze experiment results from experiments.jsonl (bt1/bt2/bt3 schema).

Console output:
  - Summary table: experiment × BT → log-loss, accuracy, draw recall
  - probability_diagnostics() per experiment per BT

Plots:
  01_logloss_grouped.png   — grouped bar chart, experiments × BT
  02_logloss_heatmap.png   — heatmap, experiments × BT
  03_draw_distribution.png — P(draw) KDE: actual draws vs non-draws per experiment per BT
  04_outcome_distribution.png — predicted vs actual outcome distribution
  05_calibration_curve.png — calibration curves per outcome per experiment per BT

Usage:
    uv run analyze_backtest.py [--runs <slug> ...]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

BACKTESTS = ["bt1", "bt2", "bt3"]
BT_LABELS = {
    "bt1": "BT1 WC22 Group (48)",
    "bt2": "BT2 WC22 KO (16)",
    "bt3": "BT3 WC26 R1-2 (48)",
}
OUTCOMES  = ["home_win", "draw", "away_win"]
PLOTS_DIR = "plots"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_experiments(path="experiments.jsonl"):
    exps, skipped = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if any(bt in e for bt in BACKTESTS):
                exps.append(e)
            else:
                skipped += 1
    if skipped:
        print(f"  (skipped {skipped} old-schema entries — re-run with new backtest.py to log them)")
    return exps


def bt_df(exp, bt):
    """Return per-match DataFrame for a backtest, or None if missing."""
    data = exp.get(bt)
    if not data or not data.get("per_match"):
        return None
    return pd.DataFrame(data["per_match"])


def short_name(run):
    return run.rsplit("_", 1)[0] if run[-8:].isdigit() else run


# ── Helpers ───────────────────────────────────────────────────────────────────

def probability_diagnostics(df):
    """Average predicted probabilities grouped by actual outcome."""
    cols = {"p_home_win", "p_draw", "p_away_win"}
    if not cols.issubset(df.columns):
        return None
    rows = {}
    for outcome in OUTCOMES:
        sub = df[df["outcome"] == outcome]
        rows[outcome] = {
            "n":          len(sub),
            "avg_p_home": sub["p_home_win"].mean() if len(sub) else np.nan,
            "avg_p_draw": sub["p_draw"].mean()     if len(sub) else np.nan,
            "avg_p_away": sub["p_away_win"].mean() if len(sub) else np.nan,
        }
    return pd.DataFrame(rows).T.round(3)


def save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Plot 1: grouped log-loss bar chart ───────────────────────────────────────

def plot_logloss_grouped(experiments):
    names  = [short_name(e["run"]) for e in experiments]
    x      = np.arange(len(names))
    w      = 0.25
    colors = {"bt1": "#4C72B0", "bt2": "#DD8452", "bt3": "#55A868"}

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(names)), 5))

    for i, bt in enumerate(BACKTESTS):
        vals = [e[bt]["log_loss"] if e.get(bt) else np.nan for e in experiments]
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      label=BT_LABELS[bt], color=colors[bt], alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(1.099, color="gray", linestyle="--", linewidth=0.8, label="uniform baseline")
    ax.set_title("Log-loss by experiment and backtest", fontweight="bold")
    ax.set_ylabel("Log-loss (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    fig.tight_layout()
    save(fig, "01_logloss_grouped.png")


# ── Plot 2: log-loss heatmap ──────────────────────────────────────────────────

def plot_logloss_heatmap(experiments):
    names = [short_name(e["run"]) for e in experiments]
    data  = [
        [e[bt]["log_loss"] if e.get(bt) else np.nan for bt in BACKTESTS]
        for e in experiments
    ]
    df = pd.DataFrame(data,
                      index=names,
                      columns=[BT_LABELS[bt] for bt in BACKTESTS])

    fig, ax = plt.subplots(figsize=(7, max(3, 0.8 * len(names))))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax,
                linewidths=0.5, cbar_kws={"label": "log-loss"})
    ax.set_title("Log-loss heatmap (lower = greener)", fontweight="bold")
    fig.tight_layout()
    save(fig, "02_logloss_heatmap.png")


# ── Plot 3: P(draw) distribution ─────────────────────────────────────────────

def plot_draw_distribution(experiments):
    n_exp = len(experiments)
    n_bt  = len(BACKTESTS)

    fig, axes = plt.subplots(n_exp, n_bt,
                             figsize=(4 * n_bt, 3 * n_exp),
                             squeeze=False)
    fig.suptitle("P(draw) distribution — actual draws vs non-draws",
                 fontweight="bold", fontsize=12)

    for i, exp in enumerate(experiments):
        for j, bt in enumerate(BACKTESTS):
            ax = axes[i][j]
            df = bt_df(exp, bt)

            if df is None or "p_draw" not in df.columns:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            else:
                draws     = df[df["outcome"] == "draw"]["p_draw"]
                non_draws = df[df["outcome"] != "draw"]["p_draw"]

                if len(draws) > 1:
                    try:
                        draws.plot.kde(ax=ax, color="#DD8452", linewidth=2,
                                       label=f"draw (n={len(draws)})")
                    except Exception:
                        ax.axvline(draws.mean(), color="#DD8452", linewidth=2,
                                   label=f"draw (n={len(draws)}, fixed)")
                if len(non_draws) > 1:
                    try:
                        non_draws.plot.kde(ax=ax, color="#4C72B0", linewidth=2,
                                           linestyle="--",
                                           label=f"non-draw (n={len(non_draws)})")
                    except Exception:
                        ax.axvline(non_draws.mean(), color="#4C72B0", linewidth=2,
                                   linestyle="--",
                                   label=f"non-draw (n={len(non_draws)}, fixed)")

                ax.set_xlim(0, 0.6)
                ax.set_ylim(bottom=0)

            ax.set_xlabel("P(draw)" if i == n_exp - 1 else "")

            if i == 0:
                ax.set_title(BT_LABELS[bt], fontsize=10)
            if j == 0:
                ax.set_ylabel(short_name(exp["run"]), fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "03_draw_distribution.png")


# ── Plot 4: predicted vs actual outcome distribution ─────────────────────────

def plot_outcome_distribution(experiments):
    """For each experiment × BT: grouped bars of predicted% vs actual% per class."""
    n_exp = len(experiments)
    n_bt  = len(BACKTESTS)

    outcome_colors = {
        "home_win": "#4C72B0",
        "draw":     "#DD8452",
        "away_win": "#55A868",
    }
    labels     = ["home_win", "draw", "away_win"]
    x          = np.arange(len(labels))
    w          = 0.35

    fig, axes = plt.subplots(n_exp, n_bt,
                             figsize=(4 * n_bt, 3 * n_exp),
                             squeeze=False)
    fig.suptitle("Predicted vs actual outcome distribution",
                 fontweight="bold", fontsize=12)

    for i, exp in enumerate(experiments):
        for j, bt in enumerate(BACKTESTS):
            ax = axes[i][j]
            df = bt_df(exp, bt)

            if df is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            else:
                actual_pct    = [( df["outcome"]   == o).mean() for o in labels]
                predicted_pct = [( df["predicted"] == o).mean() for o in labels] \
                                if "predicted" in df.columns else [np.nan] * 3

                bars_a = ax.bar(x - w / 2, actual_pct,    w, label="actual",
                                color=[outcome_colors[o] for o in labels], alpha=0.85)
                bars_p = ax.bar(x + w / 2, predicted_pct, w, label="predicted",
                                color=[outcome_colors[o] for o in labels], alpha=0.4,
                                hatch="//")

                for bar, v in zip(bars_a, actual_pct):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                            f"{v:.0%}", ha="center", va="bottom", fontsize=7)
                for bar, v in zip(bars_p, predicted_pct):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                                f"{v:.0%}", ha="center", va="bottom", fontsize=7)

                ax.set_xticks(x)
                ax.set_xticklabels(["home\nwin", "draw", "away\nwin"], fontsize=8)
                ax.set_ylim(0, 0.8)
                ax.yaxis.set_major_formatter(
                    plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

            if i == 0:
                ax.set_title(BT_LABELS[bt], fontsize=10)
            if j == 0:
                ax.set_ylabel(short_name(exp["run"]), fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "04_outcome_distribution.png")


# ── Plot 5: calibration curves ───────────────────────────────────────────────

def plot_calibration_curve(experiments):
    """For each experiment × BT: calibration curve per outcome (predicted prob vs actual freq)."""
    n_exp = len(experiments)
    n_bt  = len(BACKTESTS)

    outcome_colors = {"home_win": "#4C72B0", "draw": "#DD8452", "away_win": "#55A868"}
    outcome_labels = {"home_win": "home win", "draw": "draw", "away_win": "away win"}
    p_cols = {"home_win": "p_home_win", "draw": "p_draw", "away_win": "p_away_win"}

    fig, axes = plt.subplots(n_exp, n_bt,
                             figsize=(4 * n_bt, 3.5 * n_exp),
                             squeeze=False)
    fig.suptitle("Calibration curves — predicted probability vs actual frequency",
                 fontweight="bold", fontsize=12)

    for i, exp in enumerate(experiments):
        for j, bt in enumerate(BACKTESTS):
            ax  = axes[i][j]
            df  = bt_df(exp, bt)

            if df is None or not {"p_home_win", "p_draw", "p_away_win"}.issubset(df.columns):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            else:
                n_bins = 4 if len(df) < 30 else 5
                for outcome, col in p_cols.items():
                    y_true = (df["outcome"] == outcome).astype(int).values
                    y_prob = df[col].values
                    try:
                        frac, mean_p = calibration_curve(
                            y_true, y_prob, n_bins=n_bins, strategy="quantile")
                        ax.plot(mean_p, frac, marker="o", linewidth=1.5,
                                color=outcome_colors[outcome],
                                label=outcome_labels[outcome])
                    except ValueError:
                        pass

                ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            if i == n_exp - 1:
                ax.set_xlabel("Mean predicted probability")
            if j == 0:
                ax.set_ylabel(short_name(exp["run"]), fontsize=9)
            if i == 0:
                ax.set_title(BT_LABELS[bt], fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "05_calibration_curve.png")


# ── Console output ────────────────────────────────────────────────────────────

def print_summary(experiments):
    print(f"\n{'=' * 84}")
    print(f"  {'experiment':<28}  "
          f"{'BT1 ll':>7} {'BT2 ll':>7} {'BT3 ll':>7}  "
          f"{'BT1 acc':>7} {'BT2 acc':>7} {'BT3 acc':>7}  "
          f"{'BT1 dr':>7} {'BT2 dr':>7} {'BT3 dr':>7}")
    print(f"  {'-' * 80}")
    for e in experiments:
        name    = short_name(e["run"])
        ll_cols = "  ".join(
            f"{e[bt]['log_loss']:>7.4f}" if e.get(bt) else f"{'—':>7}"
            for bt in BACKTESTS)
        acc_cols = "  ".join(
            f"{e[bt]['accuracy']:>6.1%}" if e.get(bt) else f"{'—':>7}"
            for bt in BACKTESTS)
        dr_cols = "  ".join(
            (f"{e[bt]['draw_recall']:>6.1%}" if e[bt].get("draw_recall") is not None
             else f"{'—':>7}") if e.get(bt) else f"{'—':>7}"
            for bt in BACKTESTS)
        print(f"  {name:<28}  {ll_cols}  {acc_cols}  {dr_cols}")


def print_diagnostics(experiments):
    for e in experiments:
        print(f"\n{'=' * 70}\n  {short_name(e['run'])}  "
              f"(commit {e['commit']})\n{'=' * 70}")
        for bt in BACKTESTS:
            df = bt_df(e, bt)
            if df is None:
                continue
            d            = e[bt]
            draws_pred   = (df["predicted"] == "draw").sum() if "predicted" in df.columns else "?"
            draws_actual = (df["outcome"]   == "draw").sum()
            dr = d.get("draw_recall")
            dr_str = f"{dr:.1%}" if dr is not None else "—"
            mp = d.get("mean_p_draw")
            mp_str = f"{mp:.3f}" if mp is not None else "—"
            tp = d.get("true_outcome_avg_prob")
            tp_str = f"{tp:.3f}" if tp is not None else "—"
            print(f"\n  {BT_LABELS[bt]} — "
                  f"log-loss {d['log_loss']:.4f} | "
                  f"accuracy {d['accuracy']:.1%} | "
                  f"draw recall {dr_str} ({draws_actual} actual / {draws_pred} predicted)")
            print(f"    mean p(draw): {mp_str}   true outcome avg prob: {tp_str}")
            diag = probability_diagnostics(df)
            if diag is not None:
                print(diag.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+",
                        help="Filter to experiments whose run name contains any of these slugs")
    args = parser.parse_args()

    experiments = load_experiments()
    if not experiments:
        print("No new-schema experiments found in experiments.jsonl.")
        print("Run:  uv run backtest.py --run-name <label>")
        return

    if args.runs:
        experiments = [e for e in experiments
                       if any(r in e["run"] for r in args.runs)]
        if not experiments:
            print(f"No experiments matched: {args.runs}")
            return

    print_summary(experiments)
    print_diagnostics(experiments)

    print(f"\n{'=' * 70}\n  Generating plots\n{'=' * 70}")
    plot_logloss_grouped(experiments)
    plot_logloss_heatmap(experiments)
    plot_draw_distribution(experiments)
    plot_outcome_distribution(experiments)
    plot_calibration_curve(experiments)
    print(f"\nAll plots saved to ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
