"""Calibration / reliability analysis of walk-forward predictions.

For each outcome (home win, draw, away win) we group matches into bins by
predicted probability and compare with how often the outcome actually happened.
A calibrated model sits on the diagonal. Error bars show sampling noise
(95%), because with ~1,200 matches the bins are small.

Models: Elo (entry priors), Dixon-Coles (half-life 365 days), bookmaker.
Saves calibration.png and prints a summary table.

Run: python calibration_analysis.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dixon_coles import walk_forward_dc
from elo_v2 import load_entrants
from predictor import book_probs, load_data
from ratings_experiment import PRIORS, run

OUTCOMES = ["Home win", "Draw", "Away win"]


def reliability(p, hit, n_bins=6):
    """Quantile bins -> (mean predicted, observed frequency, count) per bin."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    xs, fs, ns = [], [], []
    for k in range(n_bins):
        m = b == k
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        fs.append(hit[m].mean())
        ns.append(m.sum())
    return np.array(xs), np.array(fs), np.array(ns)


def ece(xs, fs, ns):
    """Expected calibration error: bin-size-weighted average |observed - predicted|."""
    return float((ns * np.abs(fs - xs)).sum() / ns.sum())


def analyse(df, entrants, out_png="calibration.png"):
    preds = {}
    _, p_elo, y = run(df, entrants, offsets=PRIORS, boost=1.0)
    preds["Elo (entry priors)"] = p_elo
    preds["Dixon-Coles (hl=365)"] = walk_forward_dc(df, half_life=365)
    preds["Bookmaker"] = book_probs(df)

    mask = np.ones(len(df), dtype=bool)
    for p in preds.values():
        mask &= ~np.isnan(p).any(axis=1)
    yt = y.values[mask]
    hits = [(yt == k).astype(float) for k in range(3)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    stats = {}
    for k, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        top = 0.0
        for name, p in preds.items():
            xs, fs, ns = reliability(p[mask][:, k], hits[k])
            err = 1.96 * np.sqrt(np.clip(fs * (1 - fs), 1e-9, None) / ns)
            ax.errorbar(xs, fs, yerr=err, marker="o", ms=4, lw=1.2, capsize=2, label=name)
            stats[(name, k)] = (ece(xs, fs, ns), p[mask][:, k].mean())
            top = max(top, xs.max(), (fs + err).max())
        ax.set_title(OUTCOMES[k])
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        lim = (0, min(1.0, top + 0.05))  # zoom to where the data is
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Reliability diagrams, walk-forward predictions ({mask.sum()} matches)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}\n")

    print(f"Actual frequencies: " + "  ".join(
        f"{OUTCOMES[k]} {hits[k].mean():.3f}" for k in range(3)))
    print("\nAverage predicted probability (compare with actual above)")
    print(f"{'Model':<24}" + "".join(f"{o:>11}" for o in OUTCOMES))
    for name in preds:
        print(f"{name:<24}" + "".join(f"{stats[(name, k)][1]:>11.3f}" for k in range(3)))
    print("\nExpected calibration error (lower is better)")
    print(f"{'Model':<24}" + "".join(f"{o:>11}" for o in OUTCOMES) + f"{'Mean':>9}")
    for name in preds:
        e = [stats[(name, k)][0] for k in range(3)]
        print(f"{name:<24}" + "".join(f"{v:>11.3f}" for v in e) + f"{np.mean(e):>9.3f}")


def main():
    df = load_data()
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    analyse(df, load_entrants())


if __name__ == "__main__":
    main()
