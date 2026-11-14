"""B01 — Data cleaning and analysis with pandas (CFA L2 PSM: "financial
data scraping, wrangling, cleaning").

CFA idea: real data is dirty. Before any statistic or regression, you
handle the three usual problems — missing values, duplicates, and
outliers — and you do it in a way you can explain and reverse.

Worked example: take clean SPY prices, deliberately dirty a copy (drop
5 random days, duplicate 3 rows, spike one price 20%), then clean it
back and count what was fixed. The cleaning steps are the same ones you
would run on scraped fundamentals or an EDGAR filing table.
"""
import numpy as np
import pandas as pd


def run(prices, source):
    rng = np.random.default_rng(7)
    clean = prices[["SPY"]].copy()
    dirty = clean.copy()
    # 1. Missing values: blank out 5 random days.
    missing_idx = rng.choice(dirty.index, size=5, replace=False)
    dirty.loc[missing_idx, "SPY"] = np.nan
    # 2. Duplicates: repeat 3 rows.
    dirty = pd.concat([dirty, dirty.iloc[100:103]]).sort_index()
    # 3. Outlier: one price spiked +20%.
    outlier_date = dirty.index[500]
    dirty.loc[outlier_date, "SPY"] *= 1.20

    n_missing_before = int(dirty["SPY"].isna().sum())
    n_dupes_before = int(dirty.index.duplicated().sum())
    # Clean: forward-fill the gaps, drop the duplicated rows.
    fixed = dirty.ffill().drop_duplicates()
    n_missing_after = int(fixed["SPY"].isna().sum())
    # Winsorize: clip daily returns to the 1st/99th percentile.
    rets = fixed["SPY"].pct_change()
    lo, hi = rets.quantile(0.01), rets.quantile(0.99)
    n_clipped = int(((rets < lo) | (rets > hi)).sum())
    clipped = rets.clip(lo, hi)

    print("B01  Data cleaning")
    print(f"  Dirty copy: {n_missing_before} missing values, "
          f"{n_dupes_before} duplicated rows, 1 spiked price (+20%)")
    print(f"  After forward-fill + drop_duplicates: {n_missing_after} "
          f"missing values left, {len(fixed)} rows")
    print(f"  Winsorized {n_clipped} daily returns outside "
          f"[{lo:.2%}, {hi:.2%}] (1st/99th percentiles)")
    print(f"  Max daily move after cleaning: {clipped.abs().max():.2%} "
          f"(was {rets.abs().max():.2%} with the spike)")
    return {"missing_before": n_missing_before,
            "missing_after": n_missing_after,
            "duplicates_removed": n_dupes_before,
            "returns_winsorized": n_clipped}
