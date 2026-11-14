"""D13 — Matplotlib for finance (tool module).

CFA idea: a chart is an argument, and in finance the argument is
usually one of four things: (1) where is the price and where is
its trend, (2) how are returns distributed, (3) what risk-return
trade-offs exist, (4) what did the price DO inside the day. This
module builds one chart for each, saves them as PNGs in results/
(local files — the GitHub push path takes text only, so charts stay
on disk and the numbers behind them are printed below), and says
what to look for in each one.

How to read each chart:
- Price + moving averages: the 50-day average is the medium-term
  trend, the 200-day is the long-term trend. Price above both =
  uptrend; a "golden cross" (50 crossing above 200) is the classic
  bullish signal B10 backtests; a "death cross" is the reverse.
- Returns histogram vs the normal curve: real returns have FAT
  TAILS — more very bad and very good days than the bell curve
  allows — and a slight positive skew. The gap between the bars
  and the curve at the extremes is the visual proof that
  normal-based risk models understate crashes.
- Efficient frontier scatter: each dot is a random portfolio; the
  upper-left edge is the efficient frontier (most return per unit
  of risk). The red star is the maximum-Sharpe portfolio, the
  green dot the minimum-volatility one. Any portfolio inside the
  cloud is dominated — the same return exists somewhere with less
  risk.
- Candlestick chart: each candle is one day — body = open to close,
  wicks = high/low. Green = close above open, red = close below.
  Long wicks = intraday indecision; a run of long red bodies =
  selling pressure; a "hammer" (long lower wick after a fall) is
  the classic bottoming shape.
"""
import os

import numpy as np
import pandas as pd


def run(prices, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from common import RESULTS_DIR

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ret = prices.pct_change().dropna()
    print("D13  Matplotlib for finance — four charts, saved to results/")
    saved = []

    # ---- 1. Price + 50/200-day moving averages -----------------------
    spy = prices["SPY"]
    ma50 = spy.rolling(50).mean()
    ma200 = spy.rolling(200).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(spy.index, spy, label="SPY", linewidth=1)
    ax.plot(ma50.index, ma50, label="50-day MA", linewidth=1)
    ax.plot(ma200.index, ma200, label="200-day MA", linewidth=1)
    ax.set_title("SPY price with 50/200-day moving averages")
    ax.legend()
    path = os.path.join(RESULTS_DIR, "d13_price_ma.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    above = bool(spy.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1])
    print(f"  Price chart: SPY ${spy.iloc[-1]:.2f}, 50-day MA "
          f"${ma50.iloc[-1]:.2f}, 200-day MA ${ma200.iloc[-1]:.2f} — "
          f"{'price sits above both averages: uptrend by the textbook definition' if above else 'price is NOT above both averages — the trend is mixed or down'}")

    # ---- 2. Returns histogram vs the normal curve --------------------
    aapl = ret["AAPL"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(aapl, bins=60, density=True, alpha=0.7,
            label="AAPL daily returns")
    xs = np.linspace(aapl.min(), aapl.max(), 200)
    ax.plot(xs, (1 / (aapl.std() * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((xs - aapl.mean()) / aapl.std()) ** 2),
            label="Fitted normal", linewidth=1.5)
    ax.set_title("AAPL daily returns vs the normal curve (fat tails)")
    ax.legend()
    path = os.path.join(RESULTS_DIR, "d13_returns_hist.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    tail_share = float((aapl.abs() > 3 * aapl.std()).mean())
    print(f"  Histogram: {tail_share:.1%} of days move more than 3 "
          f"standard deviations — the normal curve predicts 0.3%. "
          f"That gap IS the fat-tail problem: crashes happen ~{tail_share / 0.0027:.0f}x "
          f"more often than the bell curve says")

    # ---- 3. Efficient frontier scatter -------------------------------
    rng = np.random.default_rng(42)
    mu = ret.mean().to_numpy() * 252
    cov = ret.cov().to_numpy() * 252
    n = len(ret.columns)
    sims, vols, rets = [], [], []
    for _ in range(5000):
        w = rng.random(n)
        w /= w.sum()
        sims.append(w)
        vols.append(float(np.sqrt(w @ cov @ w)))
        rets.append(float(w @ mu))
    vols, rets = np.array(vols), np.array(rets)
    sharpe = (rets - 0.04) / vols
    i_max, i_min = int(np.argmax(sharpe)), int(np.argmin(vols))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(vols, rets, s=2, alpha=0.4, label="random portfolios")
    ax.scatter([vols[i_max]], [rets[i_max]], marker="*", s=200,
               label=f"max Sharpe {sharpe[i_max]:.2f}")
    ax.scatter([vols[i_min]], [rets[i_min]], marker="o", s=60,
               label=f"min vol {vols[i_min]:.1%}")
    ax.set_xlabel("Annual volatility")
    ax.set_ylabel("Annual return")
    ax.set_title("Efficient frontier (5,000 random portfolios)")
    ax.legend()
    path = os.path.join(RESULTS_DIR, "d13_frontier.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Frontier: max-Sharpe portfolio returns {rets[i_max]:.1%} "
          f"at {vols[i_max]:.1%} vol (Sharpe {sharpe[i_max]:.2f}); "
          f"min-vol portfolio {rets[i_min]:.1%} at {vols[i_min]:.1%} — "
          f"any dot below-left of the frontier edge is dominated, "
          f"i.e. the same return exists with less risk")

    # ---- 4. Candlestick-style chart (last 60 days, AAPL) -------------
    import yfinance as yf

    hist = yf.Ticker("AAPL").history(period="3mo", auto_adjust=False)
    hist = hist.tail(60)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (_, row) in enumerate(hist.iterrows()):
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=1)
        ax.add_patch(Rectangle(
            (i - 0.35, min(row["Open"], row["Close"])), 0.7,
            max(abs(row["Close"] - row["Open"]), 0.01),
            facecolor=color, edgecolor=color))
    ax.set_title("AAPL candlesticks, last 60 trading days")
    ax.set_xticks([0, 20, 40, 59])
    ax.set_xticklabels([hist.index[j].date() for j in (0, 20, 40, 59)])
    path = os.path.join(RESULTS_DIR, "d13_candlestick.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    up_days = int((hist["Close"] >= hist["Open"]).sum())
    print(f"  Candlesticks: {up_days} of {len(hist)} days closed up; "
          f"last close ${hist['Close'].iloc[-1]:.2f} vs 60-day high "
          f"${hist['High'].max():.2f} — long upper wicks near the high "
          f"would mean sellers stepping in; long lower wicks would "
          f"mean buyers defending")
    print(f"  Saved: {', '.join(os.path.basename(p) for p in saved)}")
    return {"charts_saved": int(len(saved)),
            "spy_above_both_ma": bool(above),
            "tail_share_3sigma": tail_share,
            "max_sharpe": float(sharpe[i_max]),
            "min_vol": float(vols[i_min])}
