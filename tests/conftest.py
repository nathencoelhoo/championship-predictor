"""Shared test fixtures: a small synthetic league with known structure."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# make the repo root importable (predictor.py, elo_v2.py, ... live there)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def make_synthetic(seed=0, n_seasons=5, n_rounds=30, n_teams=24):
    """A league where teams have real strengths, two new teams arrive each season,
    and every team plays exactly once per weekly round."""
    rng = np.random.default_rng(seed)
    n_all = n_teams + 2 * n_seasons
    att = rng.normal(0, 0.25, n_all)
    dfn = rng.normal(0, 0.2, n_all)
    rows = []
    for si in range(n_seasons):
        pool = list(range(si * 2, si * 2 + n_teams))
        d = pd.Timestamp(2021 + si, 8, 6)
        for _ in range(n_rounds):
            order = rng.permutation(pool)
            for j in range(0, n_teams, 2):
                h, a = order[j], order[j + 1]
                hg = int(rng.poisson(np.exp(0.15 + att[h] + dfn[a] + 0.25)))
                ag = int(rng.poisson(np.exp(0.15 + att[a] + dfn[h])))
                rows.append(dict(
                    season=f"{(21 + si) % 100:02d}{(22 + si) % 100:02d}",
                    Date=d, HomeTeam=f"T{h}", AwayTeam=f"T{a}",
                    FTHG=hg, FTAG=ag,
                    FTR="H" if hg > ag else "A" if ag > hg else "D",
                    B365H=2.4, B365D=3.4, B365A=3.0))
            d += pd.Timedelta(days=7)
    return pd.DataFrame(rows).sort_values("Date", kind="stable").reset_index(drop=True)


def change_result(df, i):
    """Copy of df where match i has a different (home-win) result."""
    d = df.copy()
    d.loc[i, "FTHG"] = d.loc[i, "FTHG"] + 4
    d.loc[i, "FTR"] = "H"
    return d


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic()
