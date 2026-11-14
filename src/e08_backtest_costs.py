"""E08 — Backtesting with transaction costs (Hilpisch "Python for
Finance": backtesting trend strategies; the vectorised backtest in
Chapter 3 of "Python for Finance" is the ancestor of this module).

CFA idea: a backtest without costs is a fantasy. Every trade pays
the spread, commissions and market impact — here modelled as a flat
5 basis points per trade (0.05%), charged on every position change.
The honest comparison is GROSS vs NET: the same strategy, with and
without the cost line, so the drag is visible in dollars and in
Sharpe points.

How to read the result:
- Number of trades: a 50/200-day crossover on 5 years of data
  should trade a handful of times (trend systems are slow). 50+
  trades means the signal is whipsawing — each flip pays the cost
  and the strategy bleeds.
- Turnover: the share of the portfolio turned over per year. High
  turnover + costs = death by a thousand cuts.
- Cost drag: gross return minus net return, in %/yr. If the drag is
  bigger than the strategy's edge over buy-and-hold, the strategy
  only works on paper.
- Sharpe gross vs net: if the net Sharpe falls below buy-and-hold's,
  the strategy does not survive contact with reality — no matter how
  good the gross numbers looked.
- Win rate / profit factor on the NET returns: a strategy can be
  right 60% of the time and still lose money if the losers are
  bigger than the winners. Costs make the losers bigger.
"""
import numpy as np
import pandas as pd


def run(prices, source):
    spy = prices["SPY"]
    ret = spy.pct_change().dropna()
    print("E08  Backtest with transaction costs — SPY 50/200 crossover")
    ma50 = spy.rolling(50).mean()
    ma200 = spy.rolling(200).mean()
    signal = (ma50 > ma200).astype(float)  # 1 = invested, 0 = cash
    position = signal.shift(1).fillna(0)  # trade at tomorrow's open
    trades = position.diff().abs()
    n_trades = int(trades.sum())
    cost_per_trade = 0.0005  # 5 bps per position change
    gross_ret = position * ret
    cost = trades * cost_per_trade
    net_ret = gross_ret - cost
    buy_hold = ret

    def stats(series, label):
        total = float((1 + series).prod() - 1)
        ann = float((1 + series).prod() ** (252 / len(series)) - 1)
        vol = float(series.std() * np.sqrt(252))
        sharpe = (ann - 0.04) / vol if vol > 0 else np.nan
        dd = float(((1 + series).cumprod()
                    / (1 + series).cumprod().cummax() - 1).min())
        return total, ann, vol, sharpe, dd

    g_total, g_ann, g_vol, g_sharpe, g_dd = stats(gross_ret, "gross")
    n_total, n_ann, n_vol, n_sharpe, n_dd = stats(net_ret, "net")
    bh_total, bh_ann, bh_vol, bh_sharpe, bh_dd = stats(buy_hold, "buy-hold")
    turnover = float(trades.sum() / (len(trades) / 252))
    drag = (g_ann - n_ann) * 100
    print(f"  Trades: {n_trades} position changes in {len(ret)} days "
          f"(turnover ~{turnover:.1f}x per year) — "
          f"{'a slow trend system, as it should be' if n_trades < 20 else 'frequent flipping: each flip pays the cost'}")
    print(f"  Gross: total {g_total:+.1%}, {g_ann:+.1%}/yr, vol "
          f"{g_vol:.1%}, Sharpe {g_sharpe:.2f}, max DD {g_dd:.1%}")
    print(f"  Net (5 bps/trade): total {n_total:+.1%}, {n_ann:+.1%}/yr, "
          f"vol {n_vol:.1%}, Sharpe {n_sharpe:.2f}, max DD {n_dd:.1%}")
    print(f"  Buy-and-hold: total {bh_total:+.1%}, {bh_ann:+.1%}/yr, "
          f"Sharpe {bh_sharpe:.2f}, max DD {bh_dd:.1%}")
    print(f"  Cost drag: {drag:.2f} percentage points per year — "
          f"{'small: the strategy’s edge survives costs' if drag < 0.5 else 'material: costs eat a visible chunk of the edge'}; "
          f"net Sharpe {n_sharpe:.2f} vs buy-and-hold {bh_sharpe:.2f} -> "
          f"{'the strategy still beats buy-and-hold after costs' if n_sharpe > bh_sharpe else 'the strategy does NOT beat buy-and-hold after costs — the gross edge was an artefact of ignoring friction'}")
    print("  Lesson: always report NET numbers. A backtest that "
          "ignores costs is not optimistic — it is wrong")
    return {"bt_trades": n_trades, "bt_cost_drag_pp_yr": float(drag),
            "bt_sharpe_gross": float(g_sharpe),
            "bt_sharpe_net": float(n_sharpe),
            "bt_sharpe_buy_hold": float(bh_sharpe),
            "bt_total_net": float(n_total)}
