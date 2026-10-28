"""Sanity tests: probabilities are valid and the entry priors behave as intended."""
import numpy as np

from dixon_coles import walk_forward_dc
from elo_v2 import add_elo_v2
from predictor import book_probs


def test_bookmaker_probabilities_sum_to_one(synthetic_df):
    p = book_probs(synthetic_df)
    assert p.shape == (len(synthetic_df), 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()


def test_dixon_coles_probabilities_are_valid(synthetic_df):
    p = walk_forward_dc(synthetic_df, half_life=365)
    predicted = ~np.isnan(p).any(axis=1)
    assert predicted.sum() > 100
    assert np.allclose(p[predicted].sum(axis=1), 1.0)
    assert ((p[predicted] >= 0) & (p[predicted] <= 1)).all()


def _first_rating(out, team):
    """Pre-match rating at a team's first appearance."""
    home = out.index[out.HomeTeam == team].min()
    away = out.index[out.AwayTeam == team].min()
    if home < away:
        return out.loc[home, "home_elo"]
    return out.loc[away, "away_elo"]


def test_entry_priors_rank_relegated_above_promoted(synthetic_df):
    """In the synthetic data, T24 and T25 are new in the second season (start year 2022)."""
    entrants = {("T24", 2022): "PL", ("T25", 2022): "L1"}
    out, _ = add_elo_v2(synthetic_df, entrants=entrants)
    assert _first_rating(out, "T24") > _first_rating(out, "T25")


def test_unknown_entrants_use_default_offset(synthetic_df):
    out, _ = add_elo_v2(synthetic_df, entrants={("T24", 2022): "PL"})
    pl, unknown = _first_rating(out, "T24"), _first_rating(out, "T25")
    assert pl > unknown  # PL prior (0) beats the unknown default (-50)


def test_boost_of_one_is_a_no_op(synthetic_df):
    """A K boost of 1.0 must not change ratings, whatever boost_games is."""
    a, _ = add_elo_v2(synthetic_df, boost=1.0)
    b, _ = add_elo_v2(synthetic_df, boost=1.0, boost_games=0)
    assert np.allclose(a.home_elo, b.home_elo)
    assert np.allclose(a.away_elo, b.away_elo)
