# archer

Archer is a Python research system for volatility-aware multi-asset portfolio allocation. 
It implements volatility forecasting, risk-based allocation, transaction-cost-aware backtesting, and portfolio risk analytics.

## Volatility forecasting

Archer compares four forecasts of the S&P 500’s next 21 trading days of average daily variance:

* `naive_ma22`: assumes the next month will look like the trailing 22 trading days
* `har`: an OLS HAR-RV model using daily, weekly, and monthly realized variance
* `garch`: a GARCH(1,1) model fitted to daily returns
* `vix_implied`: the VIX converted from annualized 30-calendar-day volatility into mean daily variance over 21 trading days

Daily realized variance is estimated with `gk_total`, which combines overnight squared returns with the intraday Garman–Klass estimator.

### Evaluation setup

The evaluation uses an expanding-window walk-forward design.

* First fit cutoff: December 31, 2013
* Out-of-sample period: January 2, 2014 to June 11, 2026
* Forecast horizon: 21 trading days
* Refit frequency: every 21 forecast origins
* Number of model refits: 149
* Number of out-of-sample forecasts: 3,129

Training rows are purged using the end date of each target window. A row is eligible for training only when its complete 21-day future target is known by the fit cutoff.

All models are evaluated on the same dates and in the same daily variance units.

### Overall results

Lower loss is better.

| Model       | Mean QLIKE | Improvement vs. naive | Mean MSE | Improvement vs. naive |
| ----------- | ---------: | --------------------: | -------: | --------------------: |
| HAR         |      0.318 |                 25.7% | 2.17e-08 |                 38.0% |
| GARCH       |      0.334 |                 22.1% | 3.18e-08 |                  9.1% |
| VIX implied |      0.394 |                  8.1% | 2.82e-08 |                 19.4% |
| Naive MA22  |      0.428 |                     — | 3.50e-08 |                     — |

HAR produced the lowest average QLIKE and MSE over the complete out-of-sample period.

The naive forecast had the lowest median QLIKE, which means it was often competitive during ordinary observations. Its average loss was worse because it made larger errors during important volatility transitions.

### Performance across market regimes

The models did not behave the same way in every year.

HAR performed particularly well in 2014, 2015, 2019, 2021, 2024, and 2026. It struggled relative to the naive forecast during calm periods such as 2017 and 2023.

GARCH was especially competitive during more turbulent periods. It had the lowest annual QLIKE in 2018 and 2022 and performed strongly during the 2020 volatility shock.

The VIX forecast was also most useful during stress. It had the lowest QLIKE in 2020 and remained competitive in 2018 and 2022, but performed poorly during several low-volatility years.

The results suggest that:

* HAR is the strongest general-purpose forecast in the current lineup
* GARCH reacts well to abrupt return-driven volatility changes
* VIX contains useful forward-looking information but is not a consistently calibrated point forecast of future realized variance
* the trailing-month benchmark remains difficult to beat during stable regimes

### Calibration

Mincer–Zarnowitz regressions were estimated as:

```text
realized variance = intercept + slope × forecast + error
```

HAC standard errors use 20 lags because adjacent 21-day targets overlap.

| Model       | Intercept | Slope |    R² | Joint calibration p-value |
| ----------- | --------: | ----: | ----: | ------------------------: |
| Naive MA22  |  0.000057 | 0.346 | 0.117 |                    <0.001 |
| HAR         |  0.000021 | 0.869 | 0.202 |                     0.115 |
| GARCH       |  0.000038 | 0.435 | 0.236 |                    <0.001 |
| VIX implied |  0.000008 | 0.543 | 0.270 |                    <0.001 |

HAR was the only model for which the joint null of zero intercept and unit slope was not rejected at the 5% level.

VIX had the highest regression R², showing that it contains substantial information about future variance. However, its calibration restriction was strongly rejected. This is consistent with VIX being informative while generally pricing variance above what is subsequently realized.

### Pairwise forecast comparison

Diebold–Mariano tests use QLIKE loss differentials, HAC long-run variance with 20 lags, and the Harvey–Leybourne–Newbold finite-sample correction.

The HAR improvement over the naive benchmark was statistically significant:

```text
HAR vs. naive MA22
Mean QLIKE difference: -0.110
p-value: 0.0039
```

GARCH also had lower average QLIKE than the naive benchmark, but the difference was not significant at the 5% level:

```text
GARCH vs. naive MA22
Mean QLIKE difference: -0.095
p-value: 0.0889
```

HAR had lower average QLIKE than GARCH and VIX, but those differences were not statistically significant:

```text
HAR vs. GARCH: p = 0.629
HAR vs. VIX:   p = 0.135
```

GARCH significantly outperformed the VIX-implied forecast under QLIKE:

```text
GARCH vs. VIX
Mean QLIKE difference: -0.060
p-value: 0.0060
```

The MSE differences were not statistically significant for any model pair. This is partly because MSE is heavily influenced by a small number of extreme observations, especially during 2020.

### Main finding

The strongest result is not simply that HAR ranked first.

A leakage-safe, expanding-window HAR forecast improved average QLIKE by 25.7% over a trailing-month benchmark, and the improvement was statistically significant. HAR also showed the best calibration of the four models.

GARCH and VIX added useful information during stress regimes, but neither dominated across the complete sample. This leaves room for later work combining forecasts or changing model weights according to the volatility regime.

### Reproducing the evaluation

```bash
uv run pytest -q
uv run python scripts/run_forecast_eval.py
```

The evaluation writes:

```text
data/evals/forecast_panel.parquet
data/evals/forecast_panel.manifest.json
data/evals/overall_qlike.csv
data/evals/overall_mse.csv
data/evals/qlike_by_year.csv
data/evals/mse_by_year.csv
data/evals/mz_calibration.csv
data/evals/dm_qlike.csv
data/evals/dm_mse.csv
data/evals/forecasts_vs_realized.png
data/evals/vix_variance_spread.png
```

### Current limitations

* The study evaluates statistical forecasts, not a tradable strategy.
* VIX is used as an implied-variance benchmark; VIX itself is not directly investable.
* Trading costs, instrument mechanics, leverage, term structure, and tail risk are deferred to the strategy phase.
* GARCH currently uses a Gaussian innovation distribution.
* The current evaluation uses an expanding window and a fixed 21-day refit frequency. Rolling-window and alternative-refit experiments remain future ablations.

