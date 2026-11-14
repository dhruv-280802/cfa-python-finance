"""C06 — Performance measurement and attribution (CFA L2/L3 Portfolio
Management).

CFA idea: a return number without risk is a story, not a measurement.
The standard toolkit: Sharpe (excess return per unit of total risk),
Sortino (same, but only downside volatility counts), Calmar (return per
unit of worst drawdown), and benchmark-relative stats — alpha, beta,
information ratio, tracking error. A manager who beats the market with
twice the drawdown has not necessarily done a good job.

Worked example: the equal-weight SPY/AAPL/MSFT portfolio vs SPY as
benchmark — full-sample and worst-year performance, and a year-by-year
table of excess returns so you can see consistency, not just the average.
"""
import numpy as np
import pandas as pd

RISK_FREE = 0.04


def sortino(returns, risk_free_annual=RISK_FREE, target=0.0):
    """Sortino ratio: excess return per unit of downside deviation
    (volatility of returns below the target only)."""
    excess = returns - risk_free_annual / 252
    downside = returns[returns < target]
    dd = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
    return float(excess.mean() * 252 / dd) if dd > 0 else np.nan


def calmar(returns):
    """Calmar ratio: annualised return / |max drawdown|."""
    equity = (1 + returns).cumprod()
    mdd = float((equity / equity.cummax() - 1).min())
    ann = float(returns.mean() * 252)
    return ann / abs(mdd) if mdd < 0 else np.nan, mdd


def information_ratio(port_rets, bench_rets):
    """Active return / tracking error (std of active returns)."""
    active = port_rets - bench_rets
    te = float(active.std() * np.sqrt(252))
    return float(active.mean() * 252 / te) if te > 0 else np.nan, te


def run(prices, source):
    tickers = ["SPY", "AAPL", "MSFT"]
    rets = prices[tickers].pct_change().dropna()
    port = rets.mean(axis=1)
    bench = rets["SPY"]

    def sharpe(r):
        ex = r - RISK_FREE / 252
        return float(ex.mean() / ex.std() * np.sqrt(252))

    port_ann = float(port.mean() * 252)
    bench_ann = float(bench.mean() * 252)
    port_sharpe, bench_sharpe = sharpe(port), sharpe(bench)
    port_sortino = sortino(port)
    bench_sortino = sortino(bench)
    port_calmar, port_mdd = calmar(port)
    bench_calmar, bench_mdd = calmar(bench)
    ir, te = information_ratio(port, bench)
    # CAPM alpha of the portfolio vs SPY (from C02's method).
    y = port - RISK_FREE / 252
    x = bench - RISK_FREE / 252
    X = np.column_stack([np.ones(len(x)), x])
    alpha_d, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = float(alpha_d * 252)

    yearly = pd.DataFrame({
        "portfolio": port.resample("A").apply(lambda d: (1 + d).prod() - 1),
        "benchmark": bench.resample("A").apply(lambda d: (1 + d).prod() - 1),
    })
    yearly["excess"] = yearly["portfolio"] - yearly["benchmark"]
    n_years_won = int((yearly["excess"] > 0).sum())

    print("C06  Performance measurement (equal-weight vs SPY)")
    print(f"  Annualised return: portfolio {port_ann:+.2%}, "
          f"benchmark {bench_ann:+.2%}")
    print(f"  Sharpe: portfolio {port_sharpe:.2f}, benchmark {bench_sharpe:.2f}")
    print(f"  Sortino: portfolio {port_sortino:.2f}, benchmark {bench_sortino:.2f}")
    print(f"  Calmar: portfolio {port_calmar:.2f} (max DD {port_mdd:.1%}), "
          f"benchmark {bench_calmar:.2f} (max DD {bench_mdd:.1%})")
    print(f"  vs benchmark: alpha {alpha:+.2%}/yr, beta {float(beta):.2f}, "
          f"information ratio {ir:.2f}, tracking error {te:.2%}")
    print(f"  Years beating the benchmark: {n_years_won} of {len(yearly)}")
    print("  Excess return by year: " + ", ".join(
        f"{d.year} {v:+.1%}" for d, v in yearly["excess"].items()))
    return {"port_ann_return": port_ann, "bench_ann_return": bench_ann,
            "port_sharpe": port_sharpe, "bench_sharpe": bench_sharpe,
            "port_sortino": port_sortino, "bench_sortino": bench_sortino,
            "port_calmar": float(port_calmar),
            "bench_calmar": float(bench_calmar),
            "port_max_dd": port_mdd, "bench_max_dd": bench_mdd,
            "alpha": alpha, "beta": float(beta), "info_ratio": float(ir),
            "tracking_error": te, "years_won": n_years_won,
            "yearly": yearly}
