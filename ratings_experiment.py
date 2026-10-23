"""Compare Elo variants (entry priors, new-team K boost) with walk-forward validation.

Each variant feeds an Elo-only logistic regression, evaluated monthly out-of-sample.
We report log loss overall, on matches involving a team in its first 10 games
after entering the league (the weak spot), and on all other matches.

Run: python ratings_experiment.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from elo_v2 import add_elo_v2, load_entrants
from predictor import CLASSES, load_data
from walkforward import bootstrap_diff, per_row_logloss, walk_forward

BOOST_GAMES = 10
FLAT = {"PL": -50, "L1": -50, "UNK": -50}
PRIORS = {"PL": 0, "L1": -60, "UNK": -50}

VARIANTS = {
    "Baseline (flat entry)": dict(offsets=FLAT, boost=1.0),
    "New-team K boost": dict(offsets=FLAT, boost=1.5),
    "Entry priors": dict(offsets=PRIORS, boost=1.0),
    "Priors + K boost": dict(offsets=PRIORS, boost=1.5),
}


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def run(df0, entrants, **kw):
    d, _ = add_elo_v2(df0, entrants=entrants, boost_games=BOOST_GAMES, **kw)
    X = pd.DataFrame({"elo_diff": d.home_elo - d.away_elo}, index=d.index)
    y = d.FTR.map(CLASSES).astype(int)
    return d, walk_forward(d, X, y, make_model, ["elo_diff"]), y


def main():
    df0 = load_data()
    df0 = df0.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    entrants = load_entrants()

    preds, flags, y = {}, None, None
    for name, kw in VARIANTS.items():
        d, p, y = run(df0, entrants, **kw)
        preds[name] = p
        if flags is None:
            flags = (d.home_new | d.away_new).values

    first = next(iter(preds))
    mask = ~np.isnan(preds[first]).any(axis=1)
    yt = y.values
    print(f"Evaluated on {mask.sum()} matches; {(mask & flags).sum()} involve a team "
          f"in its first {BOOST_GAMES} games after entering.\n")

    base_loss = per_row_logloss(yt[mask], preds[first][mask])
    print(f"{'Variant':<24}{'All':>8}{'New-team':>10}{'Other':>8}   vs baseline (95% CI)")
    for name, p in preds.items():
        def ll(m):
            return log_loss(yt[m], p[m], labels=[0, 1, 2])
        lo, hi = bootstrap_diff(per_row_logloss(yt[mask], p[mask]), base_loss)
        ci = "" if name == first else f"[{lo:+.4f}, {hi:+.4f}]"
        print(f"{name:<24}{ll(mask):>8.4f}{ll(mask & flags):>10.4f}{ll(mask & ~flags):>8.4f}   {ci}")
    print("\nNegative CI = variant beats the baseline.")


if __name__ == "__main__":
    main()
