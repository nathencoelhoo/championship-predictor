# EFL Championship Match Predictor

**Live demo:** https://championship-predictor.streamlit.app/

Predicts Home / Draw / Away probabilities for Championship matches and
benchmarks them against bookmaker odds.

## Setup
```
pip install pandas numpy scikit-learn matplotlib
```

## Data

Results and bookmaker odds come from [football-data.co.uk](https://www.football-data.co.uk/englandm.php) (Championship, 2021-22 to 2026-27, about 2,900 matches, current season up to 17 Oct 2026). Files live in `data/` as `E1_<label>.csv`; the label in the filename is only a season tag, and the model reads real match dates from inside each file. Duplicate fixtures are dropped on load.

To refresh, replace the current season's CSV in `data/` with the latest download and redeploy.

Run the evaluation and the app:
```
python predictor.py
streamlit run app.py
```

## Method
- **Elo ratings** with home advantage, goal-difference multiplier, between-season regression to the mean, and entry priors (teams relegated from the Premier League start higher than teams promoted from League One).
- **Rolling form** (points, goals for/against over the last 5 matches) and rest days,
  all shifted so a match never sees its own result.
- **Models**: Elo-only logistic regression, logistic regression + form,
  gradient boosting.
- **Evaluation**: walk-forward validation. Every month, retrain on past matches only and predict the next month, so all predictions are out-of-sample.
  Log loss and multiclass Brier score, with Bet365 odds (margin removed) as the benchmark, not a feature.
  Bootstrap confidence intervals compare each model with the bookmaker.

## Results
Walk-forward validation (retrained monthly, 1,219 matches from 2024-25 to 2026-27). Lower is better; uniform guessing scores 1.0986 log loss.

| Model | Log loss | Brier |
|---|---|---|
| Elo only | 1.0526 | 0.6350 |
| Logreg + form | 1.0560 | 0.6374 |
| Gradient boosting | 1.1169 | 0.6718 |
| Bookmaker (Bet365, margin removed) | 1.0364 | 0.6243 |

**Takeaways:** Elo captures most of the available signal and clearly beats naive guessing, but every model trails the bookmakers (95% CI for Elo minus bookmaker log loss: +0.007 to +0.025). Rolling form added nothing over Elo, and gradient boosting overfit the small dataset. Early-season error is highest because newly promoted teams start at a flat rating.

**Entry priors:** teams relegated from the Premier League start higher than teams promoted from League One. This lowered Elo-only log loss from 1.0546 to 1.0529 (95% CI of the difference: -0.0030 to -0.0004), with a larger gain on matches involving newly arrived teams (1.0803 to 1.0742). A K-factor boost for new teams added nothing and was dropped. The live app uses these ratings.
**Dixon-Coles:** a time-decayed Dixon-Coles scoreline model (attack/defence strengths, Poisson goals, low-score correction) performed on par with Elo (log loss 1.0558 vs 1.0529; 95% CI of the difference includes zero) and was insensitive to the decay half-life (180 to 730 days). A 50/50 blend of Elo and Dixon-Coles scored 1.0502, a small gain that is not statistically significant. All models remain behind the bookmaker (1.0374).

## Roadmap

Done:
- Walk-forward validation with bootstrap confidence intervals
- Live Streamlit app with Elo history, recent form and a bookmaker comparison
- Entry priors for promoted and relegated teams (small but significant gain)
- - Dixon-Coles scoreline model, deriving H/D/A from predicted goals

Next:

- Probability calibration and reliability plots
- Rolling xG features (needs a Championship xG data source)
- Betting simulation with ROI and uncertainty (expected to be near or below zero)
- Tests for data leakage and a cleaner module structure

## Files

- `predictor.py`: data loading, Elo, rolling form, models
- `elo_v2.py`: Elo with entry priors for promoted and relegated teams
- `walkforward.py`: monthly walk-forward evaluation against the bookmaker
- `ratings_experiment.py`: compares Elo variants (entry priors, new-team K boost)
- `app.py`: Streamlit app
- `data/`: season CSVs from football-data.co.uk and `entrants.csv` (promoted and relegated teams by season)
- - `dixon_coles.py`: Dixon-Coles scoreline model with time decay
- `dixon_coles_experiment.py`: walk-forward comparison of Elo, Dixon-Coles and bookmaker
