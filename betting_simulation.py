"""Betting simulation on walk-forward predictions (research exercise, not advice).

Rule: for each match, compute the expected profit per unit stake for each
outcome, EV = model_probability * decimal_odds - 1. If the best outcome has
EV above a threshold, stake 1 unit on it. We report ROI with a bootstrap 95%
interval. Thresholds are fixed in advance (below), and ALL of them are shown,
to avoid cherry-picking.

Baselines: always back the favourite, always back the home side. Those show
the cost of the bookmaker margin. Bet365 odds from football-data.co.uk.

Caveats: odds are not closing odds, no stake limits, no odds movement, no
account restrictions. Real-world results would be worse.

Saves betting.png and prints a summary. Run: python betting_simulation.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dixon_coles import walk_forward_dc
from elo_v2 import load_entrants
from predictor import CLASSES, book_probs, load_data
from ratings_experiment import PRIORS, run

THRESHOLDS = [0.0, 0.03, 0.05, 0.10]
PRIMARY = 0.05  # chosen before looking at results; used for the chart


def simulate(p, odds, y, threshold):
    """Flat 1-unit stakes on the outcome with the highest EV, if EV > threshold.
    Returns (bet mask, profit per match if it were bet)."""
    ev = p * odds - 1
    pick = ev.argmax(axis=1)
    rows = np.arange(len(pick))
    bet = ev[rows, pick] > threshold
    profit = np.where(y == pick, odds[rows, pick] - 1, -1.0)
    return bet, profit


def baseline_profit(odds, y, pick):
    rows = np.arange(len(pick))
    return np.where(y == pick, odds[rows, pick] - 1, -1.0)


def roi_with_ci(profit, n_boot=2000, seed=0):
    """ROI (mean profit per unit staked) and 95% bootstrap interval."""
    if len(profit) == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(profit, len(profit)).mean() for _ in range(n_boot)])
    return profit.mean(), tuple(np.percentile(means, [2.5, 97.5]))


def analyse(df, entrants, out_png="betting.png"):
    odds = df[["B365H", "B365D", "B365A"]].values.astype(float)
    y = df.FTR.map(CLASSES).values.astype(int)

    preds = {}
    _, p_elo, _ = run(df, entrants, offsets=PRIORS, boost=1.0)
    preds["Elo (entry priors)"] = p_elo
    preds["Dixon-Coles"] = walk_forward_dc(df, half_life=365)
    preds["Blend"] = 0.5 * preds["Elo (entry priors)"] + 0.5 * preds["Dixon-Coles"]

    mask = ~np.isnan(odds).any(axis=1)
    for p in preds.values():
        mask &= ~np.isnan(p).any(axis=1)
    odds_m, y_m = odds[mask], y[mask]
    margin = (1 / odds_m).sum(axis=1).mean() - 1
    print(f"{mask.sum()} matches with predictions and odds. Average bookmaker margin: {margin:.1%}\n")

    print("Baselines (all matches, 1 unit each)")
    base = {"Always back favourite": baseline_profit(odds_m, y_m, odds_m.argmin(axis=1)),
            "Always back home": baseline_profit(odds_m, y_m, np.zeros(len(y_m), dtype=int))}
    for name, prof in base.items():
        roi, (lo, hi) = roi_with_ci(prof)
        print(f"  {name:<24} bets={len(prof):>5}  ROI {roi:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")

    print("\nModel value betting: EV threshold -> bets, ROI, 95% CI")
    curves = {}
    for name, p in preds.items():
        print(f"\n{name}")
        for t in THRESHOLDS:
            bet, profit = simulate(p[mask], odds_m, y_m, t)
            roi, (lo, hi) = roi_with_ci(profit[bet])
            tag = "  <- primary" if t == PRIMARY else ""
            if bet.sum() == 0:
                print(f"  EV > {t:>4.0%}  no bets")
                continue
            print(f"  EV > {t:>4.0%}  bets={bet.sum():>4}  ROI {roi:+6.1%}  "
                  f"95% CI [{lo:+.1%}, {hi:+.1%}]{tag}")
            if t == PRIMARY:
                curves[name] = np.cumsum(profit[bet])

    fav = base["Always back favourite"]
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.cumsum(fav), color="grey", ls="--", label="Always back favourite")
    for name, c in curves.items():
        plt.plot(c, label=f"{name} (EV > {PRIMARY:.0%})")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("Bet number (chronological; the baseline has one bet per match)")
    plt.ylabel("Cumulative profit (units)")
    plt.title("Flat-stake cumulative profit, walk-forward predictions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")


def main():
    df = load_data()
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    analyse(df, load_entrants())


if __name__ == "__main__":
    main()
