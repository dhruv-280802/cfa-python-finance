"""A12 — Portfolio returns, risk and Sharpe (CFA L1 PSM: "calculate
portfolio returns, risks, and Sharpe ratios in Python").

CFA idea: a portfolio's return is the weighted average of its assets'
returns, but its risk is NOT the weighted average of their risks —
diversification cuts risk whenever assets are less than perfectly
correlated. That gap is the whole reason portfolios exist.

Worked examples: (1) equal-weight portfolio of all four assets — return,
vol, Sharpe, max drawdown; (2) the same assets held alone, so you can
see the diversification benefit in numbers; (3) monthly rebalancing vs
buy-and-hold — does the discipline add anything? (4) the rolling 1-year
Sharpe, to show risk-adjusted performance is not constant.
"""
import numpy as np
import pandas as pd

RISK_FREE = 0.04


def max_drawdown(equity):
    """Largest peak-to-trough fall of an equity curve, as a decimal."""
    return float((equity / equity.cummax() - 1).min())


def run(prices, source):
    rets = prices.pct_change().dropna()
    n = len(rets.columns)
    weights = np.repeat(1 / n, n)
    port_rets = rets @ weights
    equity = (1 + port_rets).cumprod()
    mu = float(port_rets.mean() * 252)
    vol = float(port_rets.std() * np.sqrt(252))
    sharpe = (mu - RISK_FREE) / vol
    mdd = max_drawdown(equity)

    # Each asset alone, for the diversification comparison.
    solo_vol = {t: float(rets[t].std() * np.sqrt(252))
                for t in rets.columns}
    avg_solo_vol = float(np.mean(list(solo_vol.values())))
    diversification_gain = avg_solo_vol - vol

    # Monthly rebalanced vs buy-and-hold: rebalance to equal weights
    # at each month-end, then compare terminal wealth.
    monthly = rets.resample("M").apply(lambda d: (1 + d).prod() - 1)
    w = np.repeat(1 / n, n)
    rb_equity = 1.0
    for _, row in monthly.iterrows():
        rb_equity *= float(1 + row.to_numpy() @ w)
    bh_equity = float(equity.iloc[-1])
    rb_total = rb_equity - 1

    # Rolling 1-year Sharpe, latest value.
    roll_sharpe = (port_rets.rolling(252).mean() * 252 - RISK_FREE) / \
        (port_rets.rolling(252).std() * np.sqrt(252))
    latest_roll_sharpe = float(roll_sharpe.iloc[-1])

    print("A12  Portfolio returns, risk & Sharpe")
    print(f"  Equal-weight {n}-asset portfolio: return {mu:+.2%}/yr, "
          f"vol {vol:.2%}, Sharpe {sharpe:.2f}, max drawdown {mdd:.1%}")
    print(f"  Average single-asset vol: {avg_solo_vol:.2%} -> "
          f"diversification cuts vol by {diversification_gain:.2%} "
          f"({diversification_gain / avg_solo_vol:.0%} of it)")
    print(f"  Buy-and-hold terminal wealth (per $1): ${bh_equity:.3f}; "
          f"monthly-rebalanced: ${rb_equity:.3f} "
          f"({rb_total:+.1%} vs {bh_equity - 1:+.1%})")
    print(f"  Latest rolling 1-year Sharpe: {latest_roll_sharpe:.2f}")
    return {"port_return": mu, "port_vol": vol, "port_sharpe": float(sharpe),
            "port_max_dd": mdd, "avg_solo_vol": avg_solo_vol,
            "buy_hold_wealth": bh_equity, "rebalanced_wealth": float(rb_equity),
            "rolling_sharpe_latest": latest_roll_sharpe}
