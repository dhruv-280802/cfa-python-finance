"""A08 — Basic visualization (CFA L1 PSM: Matplotlib, Seaborn, Plotly).

CFA idea: a chart is an argument. Before you plot, know the one number
the chart has to show — here, how lumpy SPY's monthly returns are.

This repo ships NO image files (the upload path takes text only), so this
module does two things: (1) it builds a real Matplotlib figure in memory
— the code is the lesson, and saving it is one line,
`fig.savefig("spy_monthly.png")`, left for you to uncomment; (2) it
prints the same data as a text bar chart, which is what lands in the
results file. The numbers are identical either way.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(prices, source):
    monthly = prices["SPY"].resample("M").last().pct_change().dropna()
    last12 = monthly.tail(12)

    # The Matplotlib version (figure built, not saved — see docstring).
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(last12.index, last12.values * 100)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("SPY monthly returns, last 12 months (%)")
    ax.set_ylabel("%")
    plt.close(fig)  # built and discarded; uncomment savefig to keep it

    print("A08  Visualization — SPY monthly returns, last 12 months")
    print("  (text bar chart; each '#' is ~0.5%)")
    for date, r in last12.items():
        bar = "#" * max(1, int(abs(r) * 200)) if r != 0 else ""
        sign = "+" if r >= 0 else "-"
        print(f"  {date.strftime('%Y-%m')} {sign}{abs(r):5.2%} {bar}")
    print(f"  Best month: {last12.max():+.2%} ({last12.idxmax().date()}), "
          f"worst: {last12.min():+.2%} ({last12.idxmin().date()})")
    return {"last12_best": float(last12.max()),
            "last12_worst": float(last12.min()),
            "last12_mean": float(last12.mean())}
