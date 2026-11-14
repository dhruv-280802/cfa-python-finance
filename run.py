"""Run every module in order: Part A (Python fundamentals, CFA L1 PSM),
Part B (data analysis & modelling, CFA L2 PSM + quant methods),
Part C (finance applications), Part D (CFA L2 PSM screenshot units:
data wrangling, ML forecasting, NLP, plus yfinance/stats/charts
tool modules), Part E (the wider Python-for-Finance canon:
diagnostics, event studies, factor models, GARCH, Greeks, yield
curve, pairs, backtesting with costs, risk parity, Black-Litterman,
attribution, drawdowns). Writes results/summary.txt plus the CSVs
listed in DATAFRAME_EXPORTS.

Usage:
    pip install -r requirements.txt
    python run.py
"""
import importlib
import os
import sys
import warnings
from datetime import datetime

# Silence statsmodels/pandas FutureWarnings (deprecation notices about
# return types in a future release) so the teaching output stays
# readable — they change nothing about the numbers.
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from common import load_data, RESULTS_DIR

# (section header, module name) — order is the learning path.
MODULES = [
    ("PART A — CFA Level I PSM: Python Programming Fundamentals", [
        ("A01  Variables & data types", "a01_basics"),
        ("A02  Lists, dicts, sets", "a02_collections"),
        ("A03  Control flow", "a03_control_flow"),
        ("A04  Functions & modules", "a04_functions"),
        ("A05  NumPy arrays", "a05_numpy"),
        ("A06  pandas Series & DataFrames", "a06_pandas"),
        ("A07  Reading financial data", "a07_reading_data"),
        ("A08  Visualization", "a08_visualization"),
        ("A09  Errors & defensive coding", "a09_error_handling"),
        ("A10  Dates, time & calendars", "a10_dates_time"),
        ("A11  Classes & objects", "a11_oop_basics"),
        ("A12  Portfolio returns, risk & Sharpe", "a12_portfolio_basics"),
    ]),
    ("PART B — CFA Level II PSM: Python, Data Science & AI + quant methods", [
        ("B01  Data cleaning", "b01_data_cleaning"),
        ("B02  Descriptive statistics", "b02_descriptive_stats"),
        ("B03  Hypothesis testing", "b03_hypothesis_testing"),
        ("B04  Regression (OLS)", "b04_regression"),
        ("B05  Time-series basics", "b05_time_series"),
        ("B06  Monte Carlo simulation", "b06_monte_carlo"),
        ("B07  Financial model (AAPL DCF)", "b07_financial_model"),
        ("B08  ML prediction workflow", "b08_ml_prediction"),
        ("B09  Text & sentiment analysis", "b09_sentiment_text"),
        ("B10  Backtesting a rule", "b10_backtesting"),
    ]),
    ("PART C — Finance applications (TVM, CAPM, risk, options, bonds)", [
        ("C01  Time value of money", "c01_tvm"),
        ("C02  Portfolio theory & CAPM", "c02_portfolio_capm"),
        ("C03  VaR & Expected Shortfall", "c03_var_es"),
        ("C04  Options (Black-Scholes)", "c04_options_basics"),
        ("C05  Fixed income & duration", "c05_fixed_income"),
        ("C06  Performance measurement", "c06_performance"),
    ]),
    ("PART D — CFA Level II PSM screenshot units + tool modules", [
        ("D01  yfinance end to end", "d01_yfinance_end_to_end"),
        ("D02  Statements -> features", "d02_financial_statements_features"),
        ("D03  Missing values & encoding", "d03_missing_values_encoding"),
        ("D04  Regression & least squares", "d04_regression_least_squares"),
        ("D05  scikit-learn workflow", "d05_sklearn_workflow"),
        ("D06  Feedforward neural network", "d06_neural_network_ann"),
        ("D07  Random forest + XGBoost", "d07_tree_forest_xgboost"),
        ("D08  Hyperparameter tuning", "d08_hyperparameter_tuning"),
        ("D09  Text cleaning & tokenisation", "d09_text_cleaning_tokenization"),
        ("D10  LSTM from scratch (NumPy)", "d10_lstm_from_scratch"),
        ("D11  Sentiment: TF-IDF baseline", "d11_sentiment_baseline_tfidf"),
        ("D12  Stats libraries (scipy/statsmodels)", "d12_stats_libraries"),
        ("D13  Matplotlib for finance", "d13_matplotlib_finance"),
    ]),
    ("PART E — The wider Python-for-Finance canon", [
        ("E01  OLS diagnostics (full battery)", "e01_ols_diagnostics"),
        ("E02  Event study (earnings)", "e02_event_study"),
        ("E03  Fama-French 3-factor", "e03_fama_french"),
        ("E04  GARCH(1,1) volatility", "e04_garch_volatility"),
        ("E05  Options Greeks + implied vol", "e05_options_greeks_iv"),
        ("E06  Yield-curve bootstrapping", "e06_yield_curve_bootstrap"),
        ("E07  Pairs trading / stat arb", "e07_pairs_statarb"),
        ("E08  Backtest with costs", "e08_backtest_costs"),
        ("E09  Risk parity", "e09_risk_parity"),
        ("E10  Black-Litterman", "e10_black_litterman"),
        ("E11  Performance attribution", "e11_performance_attribution"),
        ("E12  Drawdown analysis", "e12_drawdown_analysis"),
    ]),
]

# DataFrame results saved to CSV instead of the text summary.
DATAFRAME_EXPORTS = {
    "dcf_table": "dcf_sensitivity.csv",
    "frontier": "efficient_frontier.csv",
    "yearly": "yearly_excess_returns.csv",
    "features_table": "d02_financial_features.csv",
}


def _fmt(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main():
    prices, source = load_data()
    print("=" * 68)
    print("CFA PYTHON — from basics to portfolio models")
    print(f"Data source: {source}")
    print(f"Sample: {prices.index[0].date()} to {prices.index[-1].date()} "
          f"({len(prices)} trading days: SPY, AAPL, MSFT, TLT)")
    print("=" * 68)

    collected = {}
    for section, modules in MODULES:
        print(f"\n{'=' * 68}\n{section}\n{'=' * 68}")
        for label, name in modules:
            print(f"\n--- {label} ---")
            module = importlib.import_module(name)
            result = module.run(prices, source)
            collected[label] = result or {}

    # ---- save results ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("CFA Python — from basics to portfolio models\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Data source: {source}\n")
        f.write(f"Sample: {prices.index[0].date()} to "
                f"{prices.index[-1].date()} ({len(prices)} trading days)\n")
        f.write("Assets: SPY, AAPL, MSFT, TLT (daily adjusted closes)\n\n")
        for label, result in collected.items():
            f.write(f"[{label}]\n")
            for key, value in result.items():
                if hasattr(value, "to_csv"):  # a DataFrame -> separate file
                    path = DATAFRAME_EXPORTS.get(key, f"{key}.csv")
                    value.to_csv(os.path.join(RESULTS_DIR, path))
                    f.write(f"{key}: see {path} "
                            f"({value.shape[0]} rows x {value.shape[1]} cols)\n")
                else:
                    f.write(f"{key}: {_fmt(value)}\n")
            f.write("\n")
    print(f"\nSaved summary + CSVs -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
