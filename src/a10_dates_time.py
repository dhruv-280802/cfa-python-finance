"""A10 — Dates, time and calendars (CFA L1 PSM: working with financial
time series).

CFA idea: finance runs on calendars, not on plain dates. Returns are
measured between trading days, interest accrues by day-count, and a
"month" in a price file is usually month-end, not the 30th. pandas
handles all of this — if you know which tool to reach for.

Worked examples: (1) the sample's span in calendar days vs trading
days; (2) trading days per year; (3) month-end and quarter-end dates;
(4) day-count fractions (ACT/360 vs 30/360) on a real holding period;
(5) the worst single-day move and the date it happened.
"""
import numpy as np
import pandas as pd


def day_count_act(start, end, basis=360):
    """ACT/basis day-count fraction: actual calendar days / basis."""
    return (end - start).days / basis


def day_count_30_360(start, end):
    """30/360 day-count fraction (US bond convention): every month is
    30 days, every year is 360 days."""
    d1, d2 = min(start.day, 30), min(end.day, 30) if start.day < 30 else 30
    m1, m2 = start.month, end.month
    y1, y2 = start.year, end.year
    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360


def run(prices, source):
    idx = prices.index
    first, last = idx[0], idx[-1]
    cal_days = (last - first).days
    trading_days = len(idx)
    per_year = idx.to_series().groupby(idx.year).size()

    rets = prices["SPY"].pct_change().dropna()
    worst_day = rets.idxmin()
    worst_ret = float(rets.min())
    best_day = rets.idxmax()
    best_ret = float(rets.max())

    # A real holding period: bought SPY at the start, sold at the end.
    holding_act = day_count_act(first, last, 360)
    holding_30360 = day_count_30_360(first, last)
    months = prices["SPY"].resample("M").last()
    quarters = prices["SPY"].resample("Q").last()

    print("A10  Dates, time & calendars")
    print(f"  Sample: {first.date()} to {last.date()} = {cal_days} "
          f"calendar days, {trading_days} trading days")
    print("  Trading days per year: " + ", ".join(
        f"{y}: {n}" for y, n in per_year.items()))
    print(f"  Worst SPY day: {worst_ret:.2%} on {worst_day.date()}; "
          f"best: {best_ret:.2%} on {best_day.date()}")
    print(f"  Holding-period year fraction: ACT/360 = {holding_act:.4f}, "
          f"30/360 = {holding_30360:.4f} (bond accrued interest uses these)")
    print(f"  Month-end closes: {len(months)}; quarter-end closes: "
          f"{len(quarters)}; last quarter-end SPY ${quarters.iloc[-1]:.2f}")
    return {"calendar_days": int(cal_days),
            "trading_days": int(trading_days),
            "worst_day_return": worst_ret,
            "best_day_return": best_ret,
            "holding_act360": float(holding_act),
            "holding_30360": float(holding_30360)}
