"""Tests for the betting simulation's arithmetic."""
import numpy as np

from betting_simulation import baseline_profit, roi_with_ci, simulate
from predictor import book_probs


def test_simulate_hand_computed_example():
    p = np.array([[0.6, 0.2, 0.2],
                  [0.6, 0.2, 0.2],
                  [0.3, 0.4, 0.3]])
    odds = np.array([[2.0, 4.0, 4.0],
                     [2.0, 4.0, 4.0],
                     [3.0, 2.0, 3.0]])
    y = np.array([0, 1, 1])
    bet, profit = simulate(p, odds, y, threshold=0.1)
    # EVs: row0 best=0.6*2-1=+0.2 (bet), row1 same (bet), row2 best=0.3*3-1=-0.1 (no bet)
    assert bet.tolist() == [True, True, False]
    assert profit[0] == 1.0    # home won at odds 2.0 -> +1 unit
    assert profit[1] == -1.0   # home lost -> -1 unit


def test_no_value_bets_when_model_equals_margined_bookmaker(synthetic_df):
    """If our probabilities are the bookmaker's own (margin removed), every bet has
    negative expected value, so a threshold of 0 should place no bets."""
    odds = synthetic_df[["B365H", "B365D", "B365A"]].values
    y = np.zeros(len(synthetic_df), dtype=int)
    bet, _ = simulate(book_probs(synthetic_df), odds, y, threshold=0.0)
    assert not bet.any()


def test_baseline_profit():
    odds = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    y = np.array([0, 2])
    assert baseline_profit(odds, y, np.array([0, 0])).tolist() == [1.0, -1.0]


def test_roi_ci_brackets_the_mean():
    profit = np.array([1.0, -1.0, -1.0, 1.5, -1.0, -1.0] * 20)
    roi, (lo, hi) = roi_with_ci(profit)
    assert lo <= roi <= hi


def test_roi_with_no_bets_is_nan():
    roi, (lo, hi) = roi_with_ci(np.array([]))
    assert np.isnan(roi) and np.isnan(lo) and np.isnan(hi)
