"""Data-leakage tests.

The core idea: if changing the result of match i changes anything the model
could have known BEFORE match i, there is leakage.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import change_result
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dixon_coles import walk_forward_dc
from elo_v2 import add_elo_v2
from predictor import CLASSES, add_elo, add_form, load_data
from walkforward import walk_forward

FORM_COLS = ["home_pts_roll", "home_gf_roll", "home_ga_roll",
             "away_pts_roll", "away_gf_roll", "away_ga_roll"]


def _elo_v1(d):
    return add_elo(d)


def _elo_v2(d):
    return add_elo_v2(d)


@pytest.mark.parametrize("elo_fn", [_elo_v1, _elo_v2])
def test_elo_is_pre_match(synthetic_df, elo_fn):
    """A match's own result must not affect its own (or any earlier) Elo ratings."""
    df, i = synthetic_df, len(synthetic_df) // 2
    base, _ = elo_fn(df)
    changed, _ = elo_fn(change_result(df, i))
    cols = ["home_elo", "away_elo"]
    pd.testing.assert_frame_equal(base.loc[:i, cols], changed.loc[:i, cols])
    # sanity check: later ratings DO react to the changed result
    assert not np.allclose(base.loc[i + 1:, cols].values, changed.loc[i + 1:, cols].values)


def test_form_excludes_current_match(synthetic_df):
    """Rolling form at match i must not include match i itself."""
    df, i = synthetic_df, len(synthetic_df) // 2
    base, _ = add_form(df)
    changed, _ = add_form(change_result(df, i))
    pd.testing.assert_frame_equal(base.loc[:i, FORM_COLS], changed.loc[:i, FORM_COLS])
    assert not np.allclose(base.loc[i + 1:, FORM_COLS].values,
                           changed.loc[i + 1:, FORM_COLS].values, equal_nan=True)


def test_first_match_of_a_team_has_no_form(synthetic_df):
    out, _ = add_form(synthetic_df)
    for team in synthetic_df.HomeTeam.unique()[:10]:
        first_home = synthetic_df.index[synthetic_df.HomeTeam == team].min()
        first_away = synthetic_df.index[synthetic_df.AwayTeam == team].min()
        if first_home < first_away:
            assert np.isnan(out.loc[first_home, "home_pts_roll"])
        else:
            assert np.isnan(out.loc[first_away, "away_pts_roll"])


class _Recorder:
    """Stand-in model that records which rows it was trained and tested on."""
    calls = []

    def fit(self, X, y):
        self.train_idx = X.index
        return self

    def predict_proba(self, X):
        _Recorder.calls.append((self.train_idx, X.index))
        return np.full((len(X), 3), 1 / 3)


def test_walk_forward_trains_only_on_the_past(synthetic_df):
    df = synthetic_df
    X = pd.DataFrame({"x": np.arange(len(df), dtype=float)}, index=df.index)
    y = df.FTR.map(CLASSES).astype(int)
    _Recorder.calls = []
    walk_forward(df, X, y, _Recorder, ["x"])
    assert len(_Recorder.calls) > 0
    for train_idx, test_idx in _Recorder.calls:
        assert df.Date[train_idx].max() < df.Date[test_idx].min()


def _make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))


def _months_up_to(df, i):
    period = df.Date.dt.to_period("M")
    return (period <= period[i]).values


def test_walk_forward_ignores_future_results(synthetic_df):
    """Changing a result must not alter predictions for that month or earlier."""
    df, i = synthetic_df, int(len(synthetic_df) * 0.7)

    def predictions(d):
        e, _ = add_elo(d)
        X = pd.DataFrame({"elo_diff": e.home_elo - e.away_elo}, index=e.index)
        y = e.FTR.map(CLASSES).astype(int)
        return walk_forward(e, X, y, _make_lr, ["elo_diff"])

    p1, p2 = predictions(df), predictions(change_result(df, i))
    # Rows after match i in the same month legitimately see different Elo features,
    # so compare only rows up to and including match i.
    early = np.arange(len(df)) <= i
    assert np.isfinite(p1[early]).any(axis=1).any()  # the check is not vacuous
    assert np.allclose(p1[early], p2[early], equal_nan=True)
    assert not np.allclose(p1[~early], p2[~early], equal_nan=True)


def test_dixon_coles_ignores_future_results(synthetic_df):
    df, i = synthetic_df, int(len(synthetic_df) * 0.7)
    p1 = walk_forward_dc(df, half_life=365)
    p2 = walk_forward_dc(change_result(df, i), half_life=365)
    early = _months_up_to(df, i)
    assert np.isfinite(p1[early]).any(axis=1).any()
    assert np.allclose(p1[early], p2[early], equal_nan=True)
    assert not np.allclose(p1[~early], p2[~early], equal_nan=True)


def test_load_data_drops_duplicate_fixtures(tmp_path):
    csv = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "06/08/2021,Leeds,Hull,2,1,H,2.0,3.4,3.8\n"
        "06/08/2021,Leeds,Hull,2,1,H,2.0,3.4,3.8\n"
        "07/08/2021,Derby,Hull,0,0,D,2.5,3.2,2.9\n"
    )
    (tmp_path / "E1_2122.csv").write_text(csv)
    out = load_data(str(tmp_path / "E1_*.csv"))
    assert not out.duplicated(["Date", "HomeTeam", "AwayTeam"]).any()
    assert len(out) == 2
