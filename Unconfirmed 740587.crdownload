"""Walk-forward comparison: Dixon-Coles (3 half-lives) vs Elo (entry priors) vs bookmaker.

Also tests a 50/50 blend of Elo and the best Dixon-Coles model.

Run: python dixon_coles_experiment.py   (takes a few minutes)
"""
import numpy as np
from sklearn.metrics import log_loss

from dixon_coles import walk_forward_dc
from elo_v2 import load_entrants
from predictor import CLASSES, book_probs, brier, load_data
from ratings_experiment import PRIORS, run
from walkforward import bootstrap_diff, per_row_logloss

HALF_LIVES = [180, 365, 730]


def main():
    df = load_data()
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    entrants = load_entrants()

    preds = {}
    _, p_elo, y = run(df, entrants, offsets=PRIORS, boost=1.0)
    preds["Elo (entry priors)"] = p_elo
    for hl in HALF_LIVES:
        print(f"Fitting Dixon-Coles, half-life {hl} days...", flush=True)
        preds[f"Dixon-Coles hl={hl}"] = walk_forward_dc(df, half_life=hl)
    preds["Bookmaker"] = book_probs(df)

    mask = np.ones(len(df), dtype=bool)
    for p in preds.values():
        mask &= ~np.isnan(p).any(axis=1)

    yt = y.values[mask]

    # blend of Elo and the best Dixon-Coles variant
    dc_names = [k for k in preds if k.startswith("Dixon")]
    best = min(dc_names, key=lambda k: log_loss(yt, preds[k][mask], labels=[0, 1, 2]))
    blend = 0.5 * preds["Elo (entry priors)"] + 0.5 * preds[best]
    preds[f"Blend: Elo + {best.split(' ')[1]}"] = blend

    print(f"\nEvaluated on {mask.sum()} matches (same matches for every model)\n")
    print(f"{'Model':<26}{'LogLoss':>9}{'Brier':>8}")
    for name, p in preds.items():
        pm = p[mask]
        print(f"{name:<26}{log_loss(yt, pm, labels=[0, 1, 2]):>9.4f}{brier(yt, pm):>8.4f}")

    print("\nDifference in log loss (95% bootstrap CI; negative = first model better)")
    elo_loss = per_row_logloss(yt, preds["Elo (entry priors)"][mask])
    book_loss = per_row_logloss(yt, preds["Bookmaker"][mask])
    for name, p in preds.items():
        if name in ("Elo (entry priors)", "Bookmaker"):
            continue
        loss = per_row_logloss(yt, p[mask])
        lo, hi = bootstrap_diff(loss, elo_loss)
        lo2, hi2 = bootstrap_diff(loss, book_loss)
        print(f"{name:<26} vs Elo [{lo:+.4f}, {hi:+.4f}]   vs Bookmaker [{lo2:+.4f}, {hi2:+.4f}]")


if __name__ == "__main__":
    main()
