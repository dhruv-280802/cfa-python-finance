# CFA Python — From Basics to Portfolio Models

One repo that teaches Python for finance the way the CFA Program
teaches it — basic to advanced, in the order a candidate would actually
learn it — with every example run on real market data.

**Status:** complete and verified. `python run.py` runs all 53 modules
in order (5,261 lines of Python) on 5 years of real Yahoo Finance data
(SPY, AAPL, MSFT, TLT) and writes every headline number to
`results/summary.txt`.

## The idea (in plain English)

The CFA Program now requires every candidate to complete a Practical
Skills Module (PSM) at each level before receiving exam results — and
the Python modules are the ones that matter for a markets career. This
repo is a self-study version of those modules, plus the quantitative
methods and portfolio applications they point at:

- **Part A — CFA Level I PSM: "Python Programming Fundamentals".**
  Python basics the CFA way: variables, lists/dicts, control flow,
  functions, NumPy, pandas, reading real financial data, visualization,
  plus the habits that keep analysis code honest (error handling, dates,
  classes).
- **Part B — CFA Level II PSM: "Python, Data Science & AI".**
  Data cleaning, descriptive statistics, hypothesis testing, regression,
  time series, Monte Carlo, a full DCF financial model, an ML prediction
  workflow, sentiment analysis, and honest backtesting.
- **Part D — the CFA Level II PSM's four units, taught in full.**
  Unit 1 (Python fundamentals review) maps to Part A, not repeated;
  Unit 2 (data collection, wrangling, feature engineering) is D01–D03:
  yfinance end to end, financial statements to features, missing
  values and one-hot encoding; Unit 3 (financial forecasting with
  machine and deep learning) is D04–D08: least squares, the
  scikit-learn workflow, feedforward neural networks, random forests
  and XGBoost, hyperparameter tuning; Unit 4 (NLP) is D09–D11: text
  cleaning and tokenisation, a from-scratch NumPy LSTM for sentiment,
  and a TF-IDF baseline. D12–D13 are the tool modules: scipy +
  statsmodels, and Matplotlib for finance.
- **Part E — the wider Python-for-Finance canon** (Hilpisch's
  *Python for Finance* and the standard quant curriculum): the full
  OLS diagnostics battery, event studies, Fama-French 3-factor,
  GARCH volatility, options Greeks and implied volatility,
  yield-curve bootstrapping, pairs trading / statistical arbitrage,
  backtesting with transaction costs, risk parity, Black-Litterman,
  performance attribution (Brinson), and drawdown analysis.
- **Part C — Finance applications.** Time value of money, portfolio
  theory and CAPM, VaR/Expected Shortfall, Black-Scholes options,
  fixed income and duration, performance measurement.

Two facts worth knowing: the CFA exams themselves do **not** test Python
— the PSMs are a completion requirement, not exam content. And the
official PSMs are taught in Jupyter with videos and quizzes; this repo
is the same material as plain, runnable `.py` files you can read,
modify, and show a recruiter.

## Results (5 years of daily data, 2021-11-15 to 2026-11-13)

Headline numbers, all reproduced from `results/summary.txt`:

| Module | Headline result |
|--------|-----------------|
| A01 Basics | $10,000 in SPY → **$17,313** (+73.1%, CAGR 11.6%) |
| A04 Functions | AAPL +146.0% total (CAGR 19.8%); TLT −34.7% |
| A12 Portfolio basics | Equal-weight 4-asset portfolio: +10.2%/yr, vol 16.8%, Sharpe 0.37; diversification cuts vol 25% |
| B02 Descriptive stats | SPY daily returns: skew +0.28, excess kurtosis +8.82 — Jarque-Bera **rejects normality** |
| B03 Hypothesis testing | SPY mean daily return +0.05%: t = 1.61, p = 0.107 — **cannot reject zero** |
| B04 Regression | AAPL beta **1.15** vs SPY (t = 35.5), R² = 0.50; adding TLT adds nothing (R² 0.502 → 0.502) |
| B05 Time series | SPY prices non-stationary (ADF p = 0.98), returns stationary; squared-return autocorr +0.18 = volatility clustering |
| B06 Monte Carlo | $100k equal-weight, 1 year, 10,000 paths: median **$114,929**, 5th pct $81,088, P(loss) 26.1% |
| B07 DCF model | AAPL intrinsic value **$171.49** vs market $359.76 (−52.3%) at 6% growth / 9% WACC |
| B08 ML prediction | Next-day AAPL direction: test accuracy **53.3% = baseline** — the model does not beat "always guess up" |
| B09 Sentiment | Lexicon scores agreed with the market's sign on **9 of 12** illustrative headlines (corr +0.73) |
| B10 Backtesting | SPY 50/200 golden cross: +66.9%, Sharpe **0.87** vs buy-and-hold +73.1%, Sharpe 0.72 |
| C01 TVM | $300k mortgage at 6%: **$1,798.65**/month; project NPV $13,724, IRR 15.2% → accept |
| C02 CAPM | Min-vol portfolio vol 12.2%; max-Sharpe 0.64; MSFT beta **1.13**, alpha −0.03%/yr |
| C03 VaR/ES | $1m portfolio, 1-day 95%: historical VaR **$21,386**, ES $30,117; VaR breached 63 of 1,254 days (expected ~63) |
| C04 Options | SPY ATM 1-month call **$14.13**, delta 0.54; binomial tree $14.12; put-call parity gap ~0 |
| C05 Fixed income | 10y Treasury **5.16%**; 10y 5% bond $986.10, modified duration 7.75, convexity 73.1 |
| C06 Performance | Equal-weight vs SPY: +16.1% vs +12.5%/yr, alpha **+2.73%/yr**, info ratio 0.35; beat the benchmark 3 of 6 years |
| D01 yfinance | AAPL: **$5.18tn** market cap, P/E 39.7, FY2026 revenue **$478.8bn**, 23 option expiries, 93 dividends on record |
| D02 Features | AAPL FY2026: revenue +15.1%, gross margin 48.8%, ROE 117.9%, FCF **$136.1bn**; MSFT FY2026: revenue +17.8%, ROE 30.2% |
| D03 Missing data | **51 genuinely missing cells** in the AAPL/MSFT statements; one-hot encoding of 4 ticker sectors |
| D04 Regression | AAPL beta **1.15** (t = 35.5); ADF: prices non-stationary (p = 0.98), returns stationary (p ≈ 0); max VIF 1.92 (clean); White p = 1.6e-06 (heteroskedastic); RESET p = 0.002 (linear form rejected) |
| D05 sklearn | Ridge, next-day AAPL return: test R² **−0.008** — does not beat predicting the mean |
| D06 ANN | MLP (16, 8): test R² **−0.161** — the network does worse than Ridge; no non-linear edge in daily returns |
| D07 Trees | Random forest test R² −0.028, XGBoost −0.116; top feature: price vs 50-day MA |
| D08 Tuning | GridSearchCV: best Ridge alpha 1000 (test R² −0.001); best forest depth 3 — shallow trees win |
| D09 NLP pipeline | 40-sentence corpus: 120-word vocabulary, padded to 9 tokens; word cloud saved |
| D10 LSTM | From-scratch NumPy LSTM: loss 0.67 → 0.004, train 100%, test **62.5%** (8 sentences — a sanity check, not a claim) |
| D11 TF-IDF | Logistic regression baseline: test accuracy **50.0%** — a coin flip on 40 sentences |
| D12 Stats | AAPL beta 1.15, 95% CI [1.09, 1.21]; JB/Shapiro/KS all reject normality; Pearson 0.71, Spearman 0.65 |
| D13 Charts | 4 charts saved; 1.1% of days move >3σ (normal curve predicts 0.3%); max-Sharpe portfolio 0.63 |
| E01 Diagnostics | DW 1.79, Breusch-Godfrey p = 0.0008 (serial correlation); robust SEs change no conclusion |
| E02 Event study | 8 AAPL earnings: average CAR **+1.02%**, t = 0.43 — not significant; surprise/CAR correlation +0.30 |
| E03 Fama-French | SMB loading **−0.24** (large-cap behaviour), HML **−0.18** (growth behaviour), alpha p = 0.42 (zero); R² 0.500 → 0.514 |
| E04 GARCH | GARCH(1,1): alpha 0.08, beta 0.88, persistence 0.96, vol-shock half-life **17 days**; 10-day vol forecast 21.6%/yr |
| E05 Greeks/IV | SPY ATM call delta 0.54, theta −$0.60/day, IV **10.1%** (Yahoo's own: 8.7%); IV smile 6.9% → 15.5% |
| E06 Yield curve | 3m 4.12%, 10y 5.15%, 30y 5.48%; slope (10y−3m) **+1.03%** (normal curve); implied 5y rate in 5y: 5.40% |
| E07 Pairs | MS/GS cointegrated (p = 0.027), hedge ratio 5.50, half-life **29.4 days**, current z-score −1.10 |
| E08 Costs | 50/200 crossover: 3 trades in 5 years, cost drag **0.03 pp/yr**; net Sharpe 0.53 vs buy-and-hold 0.44 |
| E09 Risk parity | Weights SPY 25% / AAPL 16% / MSFT 17% / TLT 42%; each asset contributes 25% of risk; vol 14.0% vs 16.8% equal-weight |
| E10 Black-Litterman | View "AAPL beats MSFT by 2%/yr" moves AAPL 10.0% → 10.4%, MSFT 9.9% → 9.4%; max weight change 13% |
| E11 Attribution | Equal-weight vs 60/40: active **+28.4%** = allocation +15.5% + selection +10.3% + interaction +2.6% |
| E12 Drawdowns | Equal-weight max DD **−28.2%** (756 days to recover), SPY −24.5%; underwater 93% of the time; Calmar 0.33 |

The honest findings are the point: the ML model does not beat its
baseline, the DCF says Apple is expensive at the stated assumptions,
and the golden cross wins on Sharpe but loses on total return. A repo
that only reports wins is a repo nobody should trust.

## How to run

```bash
cd cfa-python-finance
pip install -r requirements.txt
python run.py        # all 53 modules, in order -> results/
```

Outputs land in `results/`: `summary.txt` (every headline number),
`efficient_frontier.csv` (5,000 random portfolios),
`dcf_sensitivity.csv` (WACC x growth), `yearly_excess_returns.csv`,
`d02_financial_features.csv` (AAPL + MSFT statement features), and
five PNG charts (`d09_wordcloud.png`, `d13_price_ma.png`,
`d13_returns_hist.png`, `d13_frontier.png`, `d13_candlestick.png`).
Charts are local files — the GitHub push path takes text only, so
PNGs are not pushed; the numbers behind every chart are printed in
the console and in `summary.txt`.

Data: Yahoo Finance daily closes (SPY, AAPL, MSFT, TLT), cached in
`data/prices.csv` after the first run; `data/sample_prices.csv` is a
bundled 60-month sample of the same series. Fama-French factors are
downloaded once from Kenneth French's data library (Dartmouth) and
cached in `data/ff_factors_daily.csv`; MS/GS prices in
`data/ms_gs_prices.csv`; AAPL earnings dates in
`data/aapl_earnings_dates.csv`. If Yahoo is unreachable,
the loader falls back to simulated correlated prices — and says so
loudly in the console and in `summary.txt`
(`Data source: SIMULATED ...`). This run used **real** Yahoo data.

## Project layout

```
run.py                    # runs all 53 modules in order, writes results/
requirements.txt          # numpy, pandas, yfinance, scipy, statsmodels,
                          # scikit-learn, matplotlib, wordcloud, xgboost
data/
  sample_prices.csv       # 60 months of real closes (Dec 2021 - Nov 2026)
  prices.csv              # cached daily closes (created on first run)
  ff_factors_daily.csv    # Fama-French daily factors, cached (E03)
  ms_gs_prices.csv        # MS/GS closes for the pairs module (E07)
  aapl_earnings_dates.csv # AAPL earnings dates + EPS surprises (E02)
src/
  common.py               # data loader (Yahoo + labelled simulated fallback)
  a01_basics.py           # variables & data types
  a02_collections.py      # lists, dicts, sets, comprehensions
  a03_control_flow.py     # if/elif/else, for, while
  a04_functions.py        # functions, HPR & CAGR
  a05_numpy.py            # arrays, vectorization, correlation matrix
  a06_pandas.py           # Series/DataFrames, resample, rolling vol
  a07_reading_data.py     # CSV files + Yahoo Finance API
  a08_visualization.py    # Matplotlib + text bar charts
  a09_error_handling.py   # try/except, validation, safe Sharpe
  a10_dates_time.py       # calendars, day-counts, worst/best days
  a11_oop_basics.py       # Asset & Portfolio classes
  a12_portfolio_basics.py # portfolio return/risk/Sharpe, diversification
  b01_data_cleaning.py    # missing values, duplicates, winsorizing
  b02_descriptive_stats.py# mean/median/skew/kurtosis, Jarque-Bera
  b03_hypothesis_testing.py # t-tests, confidence intervals
  b04_regression.py       # simple + multiple OLS (statsmodels, NumPy fallback)
  b05_time_series.py      # ADF stationarity, autocorrelation, AR(1)
  b06_monte_carlo.py      # GBM simulation, 10,000 paths
  b07_financial_model.py  # AAPL DCF end to end + sensitivity table
  b08_ml_prediction.py    # logistic regression, train/test, baseline check
  b09_sentiment_text.py   # lexicon sentiment on illustrative headlines
  b10_backtesting.py      # golden-cross backtest vs buy-and-hold
  c01_tvm.py              # PV/FV, annuities, mortgage, NPV/IRR, bond price
  c02_portfolio_capm.py   # efficient frontier, CAPM beta/alpha
  c03_var_es.py           # historical & parametric VaR/ES, breach check
  c04_options_basics.py   # Black-Scholes, Greeks, binomial cross-check
  c05_fixed_income.py     # Treasury curve, duration, convexity
  c06_performance.py      # Sharpe/Sortino/Calmar, alpha, information ratio
  d01_yfinance_end_to_end.py # Ticker: prices, dividends, splits, info,
                          #   statements, live options chain
  d02_financial_statements_features.py # statements -> engineered features
  d03_missing_values_encoding.py # real gaps, fill/drop costs, one-hot encoding
  d04_regression_least_squares.py # OLS + full diagnostics: JB, Shapiro,
                          #   ADF, VIF, Breusch-Pagan, White, RESET, DW
  d05_sklearn_workflow.py # Pipeline, time-ordered split, RMSE/R², baselines
  d06_neural_network_ann.py # MLPRegressor, backprop theory, Ridge comparison
  d07_tree_forest_xgboost.py # random forest + XGBoost, feature importances
  d08_hyperparameter_tuning.py # GridSearchCV, TimeSeriesSplit
  d09_text_cleaning_tokenization.py # cleaning, tokenisation, vocab, padding,
                          #   word cloud
  d10_lstm_from_scratch.py # NumPy LSTM + BPTT + Adam, sentiment on 40 sentences
  d11_sentiment_baseline_tfidf.py # TF-IDF + logistic regression; FinBERT note
  d12_stats_libraries.py  # scipy + statsmodels: distributions, tests, full OLS
  d13_matplotlib_finance.py # price/MA, histogram, frontier, candlesticks
  e01_ols_diagnostics.py  # the full battery + what to do when each test fails
  e02_event_study.py      # AAPL earnings: market model, CAR, t-test
  e03_fama_french.py      # Fama-French 3-factor (French data library)
  e04_garch_volatility.py # GARCH(1,1) MLE from scratch, vol forecast
  e05_options_greeks_iv.py # all five Greeks + Newton-Raphson IV + IV smile
  e06_yield_curve_bootstrap.py # par yields -> zero rates, slope, forwards
  e07_pairs_statarb.py    # Engle-Granger, hedge ratio, half-life, z-score
  e08_backtest_costs.py   # 50/200 crossover, gross vs net (5 bps/trade)
  e09_risk_parity.py      # equal risk contribution vs equal weight
  e10_black_litterman.py  # equilibrium returns + one view, posterior weights
  e11_performance_attribution.py # Brinson: allocation / selection / interaction
  e12_drawdown_analysis.py # max DD, recovery, underwater time, Calmar
results/                  # generated outputs (text + 5 PNG charts)
```

## How to interpret results (read this first)

Every number in this repo carries its interpretation next to it, in
the code comments and in the printed output. The pattern is always
the same: **what the test checks → the null hypothesis in plain
words → how to read the statistic (high vs low) → good or bad → what
to do if it fails.** The full glossary lives in `d04`, `d12`, and
`e01`; the essentials:

- **p-value:** the probability of seeing a result this extreme if
  nothing is really going on (the null is true). Below 0.05 → "this
  is probably real, not chance". Above 0.05 → "we cannot rule out
  chance — do NOT claim an effect". A p-value is never the
  probability that the null is true.
- **R² vs adjusted R²:** R² is the share of variance the model
  explains; adjusted R² penalises extra predictors, so it can fall
  when you add a useless variable. R² near 1 on daily returns is
  almost always overfitting or leakage — in this repo the best
  honest models score near zero, and that is the finding.
- **t-statistic:** coefficient ÷ standard error. |t| > 2 ≈
  significant at 5%. A large t on a variable that makes no economic
  sense is still a red flag, not a discovery.
- **Jarque-Bera / Shapiro-Wilk (normality):** null = "residuals are
  normal". Reject → fat tails: VaR from a normal curve understates
  risk; use historical VaR or a t-distribution.
- **Durbin-Watson:** near 2 = no autocorrelation (good); near 0 =
  positive autocorrelation (standard errors too small — use
  Newey-West or add lags); near 4 = negative autocorrelation.
- **Breusch-Pagan / White (heteroskedasticity):** null =
  "variance is constant". Reject → standard errors are wrong (often
  too small); use robust (HC1) standard errors, or model the
  variance (GARCH).
- **ADF (stationarity):** null = "there is a unit root" (the series
  wanders). Reject (p < 0.05) → stationary, safe to regress.
  Regressing two non-stationary series on each other gives
  beautiful, meaningless R².
- **VIF (multicollinearity):** under 5 = fine; over 10 = the
  predictors carry the same information — coefficients are
  unstable; drop one or combine them.
- **RESET:** null = "the linear form is right". Reject → the
  relationship is curved or a variable is missing; add squared
  terms or an interaction.
- **Sharpe / Sortino / Calmar:** return per unit of risk (total,
  downside, and worst-loss risk). Above 1 is good, above 2 is
  suspiciously good on real money, and anything on a backtest is
  optimistic until transaction costs and out-of-sample testing are
  in.
- **Beta / alpha:** beta is the sensitivity to the market; alpha is
  what is left after the market is accounted for. An alpha with
  p > 0.05 is indistinguishable from zero — never quote it as
  skill.

Covered, with runnable code: everything in the module map above —
including the L1 PSM's portfolio optimization and Monte Carlo, the
L2 PSM's regression-based prediction, data wrangling, and valuation
modelling, all four of its units (D01–D11), and the wider
Python-for-Finance canon (E01–E12).

Not covered (and why): the official modules also teach Jupyter
Notebooks (this repo is plain `.py` by design), EDGAR filing
ingestion and sentiment analysis via AWS Comprehend (needs an AWS
account — B09 and D11 teach the same techniques on labelled example
headlines instead), and SimFin's API for financial statements (needs
an API key — D01/D02 use Yahoo Finance through yfinance, which is
free and needs no key; the SimFin option is explained in the D01
comments). Those are extensions, not gaps to hide.

## Honest limitations (know these before any interview)

- **Assumptions are stated, not hidden**: risk-free rate 4%, DCF growth
  6% / WACC 9% / terminal growth 2.5%, Monte Carlo drift/vol from the
  trailing 5 years. Change any of them and the answers move — the DCF
  sensitivity table exists to prove it.
- **In-sample almost everywhere**: betas, volatilities and frontier
  portfolios are fit on the same 5 years they describe. B08 is the
  exception — it holds out the last 30% — and it finds nothing.
- **Simplified costs and frictions**: backtests use flat 5 bps/trade,
  no borrow fees, no market impact, fills at the close.
- **The machine-learning modules find no signal — and that is the
  honest result**: Ridge, the MLP, random forest and XGBoost all
  score R² ≈ 0 or worse out of sample on daily returns (D05–D08).
  Daily returns are close to noise; a model that "predicted" them
  with R² 0.9 would be leakage, not skill. The lesson is the
  comparison — and why a finance student should distrust any
  backtest that does not show its out-of-sample score.
- **The NLP demo is small (D09–D11)**: 40 illustrative sentences, not a
  news feed. The LSTM's 62.5% test accuracy is on 8 sentences; the
  TF-IDF baseline's 50% is a coin flip. The value is the pipeline
  (clean → tokenise → encode → pad → train → evaluate), which is
  identical on a real corpus.
- **Small samples in the canon**: the event study has 8 earnings
  events (E02); the Fama-French test is on 5 years of daily factors
  (E03); the pairs module tracks one pair (E07). Treat the
  conclusions as "how the method works", not as market truths.
- **yfinance quirks are handled, not hidden**: statement columns
  arrive newest-first (sorted in D02), AAPL's FY2022 revenue is
  missing from Yahoo's table (dropped and labelled in D02),
  dividend yield is reported in percent by this yfinance version
  (computed from dividendRate ÷ price in D01), and the options
  spot is fetched fresh rather than taken from the intraday cache
  (E05). Market-cap data for E10's Black-Litterman falls back to
  equal weights when the info endpoint is unavailable.
- **The B09 sentiment demo is small**: 12 illustrative headlines, not a
  news feed. The +0.73 correlation is a teaching illustration, not
  evidence of a signal.
- **Simulated fallback**: if Yahoo blocks the download, every number
  becomes simulated and is labelled as such — never presented as real.

## What to learn next

1. Re-run with a different sample (edit `period="5y"` in `common.py`)
   and watch which results are stable (beta ~1.15) and which are not
   (the DCF answer).
2. Add a train/test split to B04's regression and B10's backtest.
3. Replace D09's 40-sentence corpus with a real headlines feed and
   test the LSTM and TF-IDF baselines on 1,000+ days.
4. Turn A11's `Portfolio` class into a monthly-rebalanced backtester.
5. Swap the TF-IDF baseline (D11) for a pre-trained transformer
   (FinBERT) and compare its sentiment accuracy on the same test
   set.
6. Add the `arch` package's GARCH to E04 and compare its fit with
   the from-scratch MLE.

## Resume bullet

"Built a 5,200-line Python finance library (53 modules, NumPy/pandas/
statsmodels/scikit-learn/yfinance) covering the CFA Institute Python
Practical Skills Modules — Level I Python Programming Fundamentals and
Level II Python, Data Science & AI (all four units) — plus the wider
Python-for-Finance canon: OLS diagnostics (VIF, Durbin-Watson,
Breusch-Pagan, White, RESET, ADF, Jarque-Bera), event studies,
Fama-French 3-factor models, GARCH volatility, options Greeks and
implied volatility, yield-curve bootstrapping, pairs trading,
risk-parity and Black-Litterman portfolios, performance attribution,
and drawdown analysis; every result computed from 5 years of real
market data and reported with its interpretation and limitations."
