"""B02 — Descriptive statistics and distributions (CFA L2 Quant Methods).

CFA idea: before you model returns, describe them. Mean and median tell
you the centre, standard deviation the spread, skewness whether the
tails lean one way, kurtosis whether the tails are fat. Then test whether
"looks normal" survives a formal check — it rarely does for daily
returns, which is why risk models need more than a normal curve.

Worked example: daily returns of all four assets, plus a Jarque-Bera
normality test on SPY (H0: returns are normal; p < 0.05 rejects it).
"""
import numpy as np
from scipy import stats


def run(prices, source):
    rets = prices.pct_change().dropna()
    print("B02  Descriptive statistics (daily returns)")
    out = {}
    for t in rets.columns:
        r = rets[t]
        out[f"{t}_mean"] = float(r.mean())
        out[f"{t}_median"] = float(r.median())
        out[f"{t}_std"] = float(r.std())
        out[f"{t}_skew"] = float(r.skew())
        out[f"{t}_kurt"] = float(r.kurtosis())
        out[f"{t}_q05"] = float(r.quantile(0.05))
        out[f"{t}_q95"] = float(r.quantile(0.95))
        print(f"  {t}: mean {r.mean():+.4%}, median {r.median():+.4%}, "
              f"std {r.std():.2%}, skew {r.skew():+.2f}, "
              f"excess kurtosis {r.kurtosis():+.2f}, "
              f"5th pct {r.quantile(0.05):+.2%}, 95th pct {r.quantile(0.95):+.2%}")
    jb_result = stats.jarque_bera(rets["SPY"])
    jb_stat, jb_p = float(jb_result[0]), float(jb_result[1])
    print(f"  Jarque-Bera on SPY: statistic {jb_stat:,.0f}, p-value {jb_p:.2e} "
          f"-> {'REJECT normality' if jb_p < 0.05 else 'cannot reject normality'} "
          f"(fat tails + skew: use historical VaR, not just normal VaR)")
    out["spy_jarque_bera_p"] = float(jb_p)
    return out
