"""Analyze and compare backtest results across experiments from experiments.jsonl."""
import json
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import log_loss

OUTCOMES = ["home_win", "draw", "away_win"]
OUTCOME_COLORS = {"home_win": "#4C72B0", "draw": "#DD8452", "away_win": "#55A868"}
PLOTS_DIR = "plots"
N_CAL_BINS = 10


def load_experiments(path="experiments.jsonl"):
    experiments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))
    return experiments


def per_match_df(exp):
    return pd.DataFrame(exp["per_match"])


def prediction_distribution(df):
    return df["predicted"].value_counts(normalize=True).reindex(OUTCOMES, fill_value=0).round(3)


def actual_distribution(df):
    return df["outcome"].value_counts(normalize=True).reindex(OUTCOMES, fill_value=0).round(3)


def accuracy_by_outcome(df):
    rows = {}
    for outcome in OUTCOMES:
        subset = df[df["outcome"] == outcome]
        if len(subset):
            rows[outcome] = {
                "n":        len(subset),
                "correct":  subset["correct"].sum(),
                "accuracy": round(subset["correct"].mean(), 3),
            }
        else:
            rows[outcome] = {"n": 0, "correct": 0, "accuracy": float("nan")}
    return pd.DataFrame(rows).T


def confusion_matrix(df):
    return pd.crosstab(
        df["outcome"], df["predicted"],
        rownames=["actual"], colnames=["predicted"]
    ).reindex(index=OUTCOMES, columns=OUTCOMES, fill_value=0)


def probability_diagnostics(df):
    if not {"p_home_win", "p_draw", "p_away_win"}.issubset(df.columns):
        return None
    rows = {}
    for outcome in OUTCOMES:
        subset = df[df["outcome"] == outcome]
        if len(subset):
            rows[outcome] = {
                "n": len(subset),
                "avg_p_home": subset["p_home_win"].mean(),
                "avg_p_draw": subset["p_draw"].mean(),
                "avg_p_away": subset["p_away_win"].mean(),
            }
        else:
            rows[outcome] = {
                "n": 0,
                "avg_p_home": float("nan"),
                "avg_p_draw": float("nan"),
                "avg_p_away": float("nan"),
            }
    return pd.DataFrame(rows).T.round(3)


def short_name(run):
    """Strip date suffix for cleaner axis labels."""
    return run.rsplit("_", 1)[0] if run[-8:].isdigit() else run


def has_probs(df):
    return {"p_home_win", "p_draw", "p_away_win"}.issubset(df.columns)


def per_class_logloss(df):
    """Compute log-loss restricted to matches where a given class is the true outcome."""
    if not has_probs(df):
        return None
    # sklearn log_loss requires labels in lexicographic order: away_win, draw, home_win
    proba_raw = df[["p_home_win", "p_draw", "p_away_win"]].values
    proba_raw = proba_raw / proba_raw.sum(axis=1, keepdims=True)
    proba = np.column_stack([proba_raw[:, 2], proba_raw[:, 1], proba_raw[:, 0]])  # away,draw,home
    labels_lex = ["away_win", "draw", "home_win"]
    results = {}
    for cls in OUTCOMES:
        mask = df["outcome"] == cls
        if mask.sum() < 2:
            results[cls] = float("nan")
            continue
        results[cls] = log_loss(df.loc[mask, "outcome"], proba[mask], labels=labels_lex)
    return results


def calibration_ece(df, n_bins=N_CAL_BINS):
    """Expected Calibration Error per class using equal-width bins."""
    if not has_probs(df):
        return None
    prob_cols = {"home_win": "p_home_win", "draw": "p_draw", "away_win": "p_away_win"}
    results = {}
    for cls, col in prob_cols.items():
        probs = df[col].values
        labels = (df["outcome"] == cls).astype(float).values
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            conf = probs[mask].mean()
            acc  = labels[mask].mean()
            ece += mask.sum() * abs(conf - acc)
        results[cls] = ece / len(probs)
    return results


def reliability_data(df, cls, n_bins=N_CAL_BINS):
    """Return (mean_confidence, fraction_positive, bin_count) arrays for a reliability diagram."""
    col = {"home_win": "p_home_win", "draw": "p_draw", "away_win": "p_away_win"}[cls]
    probs  = df[col].values
    labels = (df["outcome"] == cls).astype(float).values
    bins = np.linspace(0, 1, n_bins + 1)
    conf_vals, frac_pos, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        conf_vals.append(probs[mask].mean())
        frac_pos.append(labels[mask].mean())
        counts.append(mask.sum())
    return np.array(conf_vals), np.array(frac_pos), np.array(counts)


def group_knockout_split(df):
    """Split WC matches into group stage (>=3 matches per team) vs knockout."""
    if "date" not in df.columns:
        return None, None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    team_count = {}
    stage = []
    for r in df.itertuples():
        h, a = r.home_team, r.away_team
        th = team_count.get(h, 0)
        ta = team_count.get(a, 0)
        is_group = th < 3 and ta < 3
        stage.append("group" if is_group else "knockout")
        team_count[h] = th + 1
        team_count[a] = ta + 1
    df["stage"] = stage
    return df[df["stage"] == "group"], df[df["stage"] == "knockout"]


def save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Plot 1: accuracy + log-loss comparison ───────────────────────────────────
def plot_summary(experiments):
    names    = [short_name(e["run"]) for e in experiments]
    accuracy = [e["accuracy"] * 100 for e in experiments]
    logloss  = [e["log_loss"] for e in experiments]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Experiment comparison", fontsize=13, fontweight="bold")

    bars = ax1.bar(names, accuracy, color="#4C72B0", alpha=0.85)
    ax1.set_title("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.axhline(accuracy[0], color="gray", linestyle="--", linewidth=0.8, label="baseline")
    for bar, v in zip(bars, accuracy):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=9)
    ax1.tick_params(axis="x", rotation=20)

    bars2 = ax2.bar(names, logloss, color="#DD8452", alpha=0.85)
    ax2.set_title("Log-loss (lower is better)")
    ax2.axhline(logloss[0], color="gray", linestyle="--", linewidth=0.8, label="baseline")
    for bar, v in zip(bars2, logloss):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    save(fig, "01_summary.png")


# ── Plot 2: prediction distribution per experiment ───────────────────────────
def plot_prediction_distributions(experiments):
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Prediction vs actual distribution per experiment",
                 fontsize=12, fontweight="bold")

    actual = actual_distribution(per_match_df(experiments[0]))

    for ax, exp in zip(axes, experiments):
        df   = per_match_df(exp)
        pred = prediction_distribution(df)
        x    = np.arange(len(OUTCOMES))
        w    = 0.35
        bars_p = ax.bar(x - w/2, [pred[o] for o in OUTCOMES], w,
                        label="predicted", color=[OUTCOME_COLORS[o] for o in OUTCOMES], alpha=0.9)
        bars_a = ax.bar(x + w/2, [actual[o] for o in OUTCOMES], w,
                        label="actual", color=[OUTCOME_COLORS[o] for o in OUTCOMES],
                        alpha=0.4, hatch="//")
        ax.set_title(short_name(exp["run"]), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["home\nwin", "draw", "away\nwin"], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "02_prediction_distributions.png")


# ── Plot 3: accuracy by outcome class ────────────────────────────────────────
def plot_accuracy_by_outcome(experiments):
    names = [short_name(e["run"]) for e in experiments]
    x     = np.arange(len(OUTCOMES))
    w     = 0.8 / len(experiments)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Accuracy by actual outcome class", fontsize=12, fontweight="bold")

    for i, exp in enumerate(experiments):
        df  = per_match_df(exp)
        acc = accuracy_by_outcome(df)
        vals = [acc.loc[o, "accuracy"] if o in acc.index else 0 for o in OUTCOMES]
        offset = (i - len(experiments) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=short_name(exp["run"]), alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(OUTCOMES)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.15)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "03_accuracy_by_outcome.png")


# ── Plot 4: confusion matrices ────────────────────────────────────────────────
def plot_confusion_matrices(experiments):
    n   = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion matrices (actual → predicted)", fontsize=12, fontweight="bold")

    for ax, exp in zip(axes, experiments):
        df = per_match_df(exp)
        cm = confusion_matrix(df)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["home\nwin", "draw", "away\nwin"],
                    yticklabels=["home\nwin", "draw", "away\nwin"],
                    cbar=False, linewidths=0.5)
        ax.set_title(short_name(exp["run"]), fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")

    fig.tight_layout()
    save(fig, "04_confusion_matrices.png")


# ── Plot 5: log-loss trend ────────────────────────────────────────────────────
def plot_logloss_trend(experiments):
    names   = [short_name(e["run"]) for e in experiments]
    logloss = [e["log_loss"] for e in experiments]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(names, logloss, marker="o", linewidth=2, color="#4C72B0")
    ax.axhline(logloss[0], color="gray", linestyle="--", linewidth=0.8, label="baseline")
    for i, (name, v) in enumerate(zip(names, logloss)):
        ax.annotate(f"{v:.3f}", (name, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title("Log-loss trend across experiments", fontsize=12, fontweight="bold")
    ax.set_ylabel("Log-loss")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    save(fig, "05_logloss_trend.png")


# ── Plot 6: reliability diagrams ─────────────────────────────────────────────
def plot_reliability(experiments):
    exps_with_probs = [e for e in experiments if has_probs(per_match_df(e))]
    if not exps_with_probs:
        return
    n = len(exps_with_probs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle("Reliability diagrams (calibration) — perfect = diagonal",
                 fontsize=12, fontweight="bold")
    for row, exp in enumerate(exps_with_probs):
        df = per_match_df(exp)
        for col, cls in enumerate(OUTCOMES):
            ax = axes[row][col]
            conf, frac, counts = reliability_data(df, cls)
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
            sc = ax.scatter(conf, frac, s=counts * 4, color=OUTCOME_COLORS[cls],
                            alpha=0.8, zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted P"); ax.set_ylabel("Fraction positive")
            ax.set_title(f"{short_name(exp['run'])} — {cls}", fontsize=9)
            ece = calibration_ece(df) or {}
            ax.text(0.05, 0.9, f"ECE={ece.get(cls, float('nan')):.3f}",
                    transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    save(fig, "06_reliability.png")


# ── Plot 7: log-loss by class ─────────────────────────────────────────────────
def plot_logloss_by_class(experiments):
    exps_with_probs = [e for e in experiments if has_probs(per_match_df(e))]
    if not exps_with_probs:
        return
    names = [short_name(e["run"]) for e in exps_with_probs]
    x = np.arange(len(OUTCOMES))
    w = 0.8 / len(exps_with_probs)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Log-loss by actual outcome class", fontsize=12, fontweight="bold")
    for i, exp in enumerate(exps_with_probs):
        df = per_match_df(exp)
        cls_ll = per_class_logloss(df) or {}
        vals = [cls_ll.get(o, float("nan")) for o in OUTCOMES]
        offset = (i - len(exps_with_probs) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=short_name(exp["run"]),
                      color=[OUTCOME_COLORS[o] for o in OUTCOMES], alpha=0.8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(OUTCOMES)
    ax.set_ylabel("Log-loss"); ax.legend(fontsize=8, loc="upper right")
    ax.axhline(np.log(3), color="gray", linestyle="--", linewidth=0.8, label="uniform=1.099")
    fig.tight_layout()
    save(fig, "07_logloss_by_class.png")


# ── Console output ────────────────────────────────────────────────────────────
def print_separator(title=""):
    width = 60
    if title:
        print(f"\n{'=' * width}\n  {title}\n{'=' * width}")
    else:
        print("-" * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", help="Filter to specific run name substrings")
    args = parser.parse_args()

    all_experiments = load_experiments()
    if not all_experiments:
        print("No experiments found in experiments.jsonl")
        return

    experiments = all_experiments
    if args.runs:
        experiments = [e for e in all_experiments
                       if any(r in e["run"] for r in args.runs)]
        print(f"Filtered to {len(experiments)} experiments matching {args.runs}")

    if not experiments:
        print("No matching experiments")
        return

    print_separator("Summary")
    summary_rows = []
    for exp in experiments:
        df = per_match_df(exp)
        pred_dist = prediction_distribution(df)
        p_draw_actual_draw = "n/a"
        p_draw_non_draw = "n/a"
        if "p_draw" in df.columns:
            draws = df[df["outcome"] == "draw"]
            non_draws = df[df["outcome"] != "draw"]
            if len(draws):
                p_draw_actual_draw = f"{draws['p_draw'].mean():.1%}"
            if len(non_draws):
                p_draw_non_draw = f"{non_draws['p_draw'].mean():.1%}"
        summary_rows.append({
            "run":        exp["run"],
            "commit":     exp["commit"],
            "accuracy":   f"{exp['accuracy']:.1%}",
            "log_loss":   exp["log_loss"],
            "n":          exp["n_matches"],
            "pred_home%": f"{pred_dist['home_win']:.0%}",
            "pred_draw%": f"{pred_dist['draw']:.0%}",
            "pred_away%": f"{pred_dist['away_win']:.0%}",
            "p_draw(actual_draw)": p_draw_actual_draw,
            "p_draw(non_draw)":    p_draw_non_draw,
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))

    for exp in experiments:
        print_separator(f"Experiment: {exp['run']}  (commit {exp['commit']})")
        df = per_match_df(exp)
        print(f"\nFeatures ({len(exp['features'])}):")
        print("  " + ", ".join(exp["features"]))
        dist = pd.DataFrame({
            "predicted": prediction_distribution(df),
            "actual":    actual_distribution(df),
        })
        dist["gap"] = (dist["predicted"] - dist["actual"]).round(3)
        print("\nPrediction vs actual distribution:")
        print(dist.to_string())
        print("\nAccuracy by actual outcome:")
        print(accuracy_by_outcome(df).to_string())
        print("\nConfusion matrix (actual → predicted):")
        print(confusion_matrix(df).to_string())
        probs = probability_diagnostics(df)
        if probs is not None:
            print("\nAverage predicted probabilities by actual outcome:")
            print(probs.to_string())

        cls_ll = per_class_logloss(df)
        if cls_ll:
            print("\nLog-loss by actual outcome class:")
            for cls, ll in cls_ll.items():
                n_cls = (df["outcome"] == cls).sum()
                print(f"  {cls:<10}  {ll:.4f}  (n={n_cls})")

        ece = calibration_ece(df)
        if ece:
            print("\nExpected Calibration Error (ECE) per class:")
            for cls, e in ece.items():
                print(f"  {cls:<10}  {e:.4f}")

        group_df, ko_df = group_knockout_split(df)
        if group_df is not None and has_probs(df):
            print(f"\nGroup stage ({len(group_df)} matches) vs Knockout ({len(ko_df)} matches):")
            for stage_label, stage_df in [("group", group_df), ("knockout", ko_df)]:
                if len(stage_df) < 2:
                    continue
                proba_raw = stage_df[["p_home_win", "p_draw", "p_away_win"]].values
                proba_raw = proba_raw / proba_raw.sum(axis=1, keepdims=True)
                proba_lex = np.column_stack([proba_raw[:, 2], proba_raw[:, 1], proba_raw[:, 0]])
                ll = log_loss(stage_df["outcome"], proba_lex,
                              labels=["away_win", "draw", "home_win"])
                acc = (stage_df["outcome"] == stage_df["predicted"]).mean()
                draws_actual = (stage_df["outcome"] == "draw").sum()
                print(f"  {stage_label:<8}  log-loss={ll:.4f}  acc={acc:.1%}  "
                      f"draws={draws_actual}/{len(stage_df)}")

    print_separator("Generating plots")
    plot_summary(experiments)
    plot_prediction_distributions(experiments)
    plot_accuracy_by_outcome(experiments)
    plot_confusion_matrices(experiments)
    plot_logloss_trend(experiments)
    plot_reliability(experiments)
    plot_logloss_by_class(experiments)
    print(f"\nAll plots saved to ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
