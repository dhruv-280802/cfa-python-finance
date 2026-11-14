"""A05 — NumPy arrays and vectorization (CFA L1 PSM).

CFA idea: a NumPy array is a fast, typed list. The finance payoff is
*vectorization*: one array operation replaces a loop, so a calculation
over 1,255 days runs in microseconds. If you can write it as an array
expression, do — never loop over rows to compute returns.

Worked example: daily returns for all four assets as arrays — means,
volatilities, and the correlation matrix, in four lines of code.
"""
import time

import numpy as np


def run(prices, source):
    rets = prices.pct_change().dropna().to_numpy()  # (days x assets)
    tickers = list(prices.columns)
    means = rets.mean(axis=0) * 252
    vols = rets.std(axis=0, ddof=1) * np.sqrt(252)
    corr = np.corrcoef(rets, rowvar=False)

    # Vectorized vs loop: same answer, very different speed.
    t0 = time.perf_counter()
    loop_mean = np.array([sum(rets[:, j]) / len(rets) for j in range(4)])
    t_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    vec_mean = rets.mean(axis=0)
    t_vec = time.perf_counter() - t0

    print("A05  NumPy arrays")
    for j, t in enumerate(tickers):
        print(f"  {t}: annualised mean {means[j]:+.2%}, vol {vols[j]:.2%}")
    print("  Correlation matrix (rows/cols: " + ", ".join(tickers) + "):")
    for j, t in enumerate(tickers):
        print("    " + t + ": " + "  ".join(f"{corr[j, k]:+.2f}"
                                            for k in range(4)))
    print(f"  Loop mean vs vectorized mean identical: "
          f"{np.allclose(loop_mean, vec_mean)} "
          f"(loop {t_loop * 1e3:.2f} ms, vectorized {t_vec * 1e3:.4f} ms)")
    return {"spy_ann_mean": float(means[0]), "spy_ann_vol": float(vols[0]),
            "corr_aapl_msft": float(corr[1, 2]),
            "corr_spy_tlt": float(corr[0, 3])}
