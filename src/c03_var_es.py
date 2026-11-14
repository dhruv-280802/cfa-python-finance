"""C03 — Risk: Value at Risk and Expected Shortfall (CFA L2 Risk
Management; also the core of the sibling var-engine project, here in
tutorial form).

CFA idea: VaR answers "how much could I lose on a bad day?" — the loss
that is exceeded only 5% (or 1%) of the time. Expected Shortfall answers
the better question: "WHEN I exceed it, how bad is it on average?"
VaR tells you the fence; ES tells you what is beyond the fence.

Worked example: a $1,000,000 equal-weight SPY/AAPL/MSFT portfolio.
Historical VaR/ES (read off the actual return distribution), parametric
VaR/ES (normal curve), and a Kupiec-style check: how many times did the
1-day loss actually exceed the 95% VaR? If the model is right, about 5%
of days.
"""
import numpy as np
import pandas as pd
from scipy import stats

PORTFOLIO_VALUE = 1_000_000.0


def historical_var_es(returns, alpha=0.05):
    """Historical VaR and ES from the empirical distribution.
    VaR is the alpha-quantile of returns (a negative number); ES is the
    mean return on days worse than VaR. Reported as positive losses."""
    var_ret = float(np.quantile(returns, alpha))
    es_ret = float(returns[returns <= var_ret].mean())
    return -var_ret, -es_ret


def parametric_var_es(returns, alpha=0.05):
    """Parametric (normal) VaR and ES from mean and std only."""
    mu, sigma = float(returns.mean()), float(returns.std(ddof=1))
    z = stats.norm.ppf(alpha)
    var_ret = mu + sigma * z
    es_ret = mu - sigma * stats.norm.pdf(z) / alpha
    return -var_ret, -es_ret


def run(prices, source):
    tickers = ["SPY", "AAPL", "MSFT"]
    rets = prices[tickers].pct_change().dropna()
    port = rets.mean(axis=1)  # equal-weight daily returns
    out = {}
    print("C03  VaR & Expected Shortfall ($1,000,000 portfolio, 1-day)")
    for alpha, label in [(0.05, "95%"), (0.01, "99%")]:
        h_var, h_es = historical_var_es(port, alpha)
        p_var, p_es = parametric_var_es(port, alpha)
        print(f"  {label} historical: VaR ${h_var * PORTFOLIO_VALUE:,.0f}, "
              f"ES ${h_es * PORTFOLIO_VALUE:,.0f}")
        print(f"  {label} parametric (normal): VaR "
              f"${p_var * PORTFOLIO_VALUE:,.0f}, ES "
              f"${p_es * PORTFOLIO_VALUE:,.0f}")
        out[f"hist_var_{label}"] = h_var * PORTFOLIO_VALUE
        out[f"hist_es_{label}"] = h_es * PORTFOLIO_VALUE
        out[f"param_var_{label}"] = p_var * PORTFOLIO_VALUE
        out[f"param_es_{label}"] = p_es * PORTFOLIO_VALUE
        if alpha == 0.05:
            # Kupiec-style check at 95%: breaches of the historical VaR.
            breaches = int((port < -h_var).sum())
            expected = 0.05 * len(port)
            print(f"  Breach check: the 95% VaR was exceeded on "
                  f"{breaches} of {len(port)} days "
                  f"(expected ~{expected:.0f} if the model is right)")
            out["var95_breaches"] = breaches
            out["var95_expected_breaches"] = float(expected)
    print("  Reading: ES is always larger than VaR — it averages the "
          "tail, where the normal curve understates real losses "
          "(see B02: returns are fat-tailed)")
    return out
