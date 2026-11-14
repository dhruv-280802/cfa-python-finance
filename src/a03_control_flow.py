"""A03 — Control flow: if/elif/else, for, while (CFA L1 PSM).

CFA idea: code makes decisions with if/elif/else and repeats work with
loops. In finance that is how you classify days, scan a series for
streaks, or stop a simulation when a condition is met.

Worked example: classify every SPY trading day as up / down / flat, then
walk the series once to find the longest run of consecutive up days.
"""
import numpy as np


def run(prices, source):
    rets = prices["SPY"].pct_change().dropna()
    up = int((rets > 0.001).sum())
    down = int((rets < -0.001).sum())
    flat = int(len(rets) - up - down)

    # Longest winning streak: a single for loop with two counters.
    best_streak, cur_streak = 0, 0
    for r in rets:
        if r > 0:
            cur_streak += 1
            best_streak = max(best_streak, cur_streak)
        else:
            cur_streak = 0

    # A while loop: how many days from the start until the first 2% up day?
    i = 0
    while i < len(rets) and rets.iloc[i] <= 0.02:
        i += 1
    first_big_up = rets.index[i].date() if i < len(rets) else None

    # if/elif/else: label the market regime by the latest 60-day return.
    recent = float(prices["SPY"].iloc[-1] / prices["SPY"].iloc[-61] - 1)
    if recent > 0.05:
        regime = "strong uptrend"
    elif recent > 0:
        regime = "mild uptrend"
    elif recent > -0.05:
        regime = "mild downtrend"
    else:
        regime = "strong downtrend"

    print("A03  Control flow")
    print(f"  SPY days: {up} up, {down} down, {flat} flat (of {len(rets)})")
    print(f"  Longest run of consecutive up days: {best_streak}")
    print(f"  First +2% day in the sample: {first_big_up}")
    print(f"  Latest 60-day SPY return: {recent:.1%} -> regime: {regime}")
    return {"up_days": up, "down_days": down, "flat_days": flat,
            "longest_up_streak": best_streak,
            "recent_60d_return": recent, "regime": regime}
