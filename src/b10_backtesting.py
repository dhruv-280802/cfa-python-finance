"""B10 — Backtesting a rule, honestly (CFA L2 PSM application: turn an
idea into a tested strategy).

CFA idea: a backtest is an experiment, and experiments need controls —
in-sample vs out-of-sample, costs, and a benchmark. The rule here is the
oldest one in technical analysis: buy when the 50-day average crosses
above the 200-day average (golden cross), sell when it crosses below
(death cross). The question is not "did it make money?" but "did it beat
buy-and-hold after costs, out of sample?"

Worked example: SPY golden-cross/death-cross, 5 bps per trade, fitted
on nothing (the rule has no fitted parameters). Report total return,
Sharpe and max drawdown vs buy-and-hold, plus the out-of-sample check:
the rule is evaluated on the last 2 years alone, having been "designed"
on the full history's reputation, not on fitted parameters.
"""
import numpy as np
import pandas as pd

COST_BPS = 5.0


def moving_average_signals(prices, fast=50, slow=200):
    """1 when the fast MA is above the slow MA (invested), else 0."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def backtest(prices, signals, cost_bps=COST_BPS):
    """Daily strategy returns: yesterday's signal times today's return,
    minus 5 bps per position change."""
    rets = prices.pct_change().dropna()
    sig = signals.reindex(rets.index).fillna(0.0)
    gross = sig.shift(1).fillna(0.0) * rets
    turnover = sig.diff().abs().fillna(0.0)
    net = gross - turnover * (cost_bps / 1e4)
    return net


def stats(net_rets):
    equity = (1 + net_rets).cumprod()
    total = float(equity.iloc[-1] - 1)
    years = len(net_rets) / 252
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    daily_std = float(net_rets.std())
    sharpe = (float(net_rets.mean() / daily_std * np.sqrt(252))
              if daily_std > 0 else 0.0)
    mdd = float((equity / equity.cummax() - 1).min())
    return {"total": total, "cagr": cagr, "sharpe": sharpe, "mdd": mdd,
            "equity": equity}


def run(prices, source):
    spy = prices["SPY"]
    sig = moving_average_signals(spy)
    net = backtest(spy, sig)
    bh = spy.pct_change().dropna()
    s_strat = stats(net)
    s_bh = stats(bh)
    n_trades = int(sig.diff().abs().fillna(0).sum())

    # Out-of-sample: last 2 years only (the rule itself was not fitted,
    # so this is a stability check, not a true holdout).
    cut = net.index[-1] - pd.Timedelta(days=730)
    oos = stats(net[net.index >= cut])
    oos_bh = stats(bh[bh.index >= cut])

    print("B10  Backtesting — SPY 50/200-day golden cross")
    print(f"  Full sample ({len(net)} days, {n_trades} trades, 5 bps/trade):")
    print(f"    Strategy: total {s_strat['total']:+.1%}, CAGR "
          f"{s_strat['cagr']:+.2%}, Sharpe {s_strat['sharpe']:.2f}, "
          f"max DD {s_strat['mdd']:.1%}")
    print(f"    Buy-and-hold: total {s_bh['total']:+.1%}, CAGR "
          f"{s_bh['cagr']:+.2%}, Sharpe {s_bh['sharpe']:.2f}, "
          f"max DD {s_bh['mdd']:.1%}")
    print(f"  Last 2 years only: strategy {oos['total']:+.1%} "
          f"(Sharpe {oos['sharpe']:.2f}) vs buy-and-hold "
          f"{oos_bh['total']:+.1%} (Sharpe {oos_bh['sharpe']:.2f})")
    print(f"  Verdict: the rule {'beats' if s_strat['sharpe'] > s_bh['sharpe'] else 'does NOT beat'} "
          f"buy-and-hold on Sharpe over the full sample — a moving "
          f"average keeps you out of crashes but also out of recoveries")
    return {"ma_total": s_strat["total"], "ma_sharpe": s_strat["sharpe"],
            "ma_mdd": s_strat["mdd"], "bh_total": s_bh["total"],
            "bh_sharpe": s_bh["sharpe"], "ma_trades": n_trades,
            "ma_oos_total": oos["total"], "bh_oos_total": oos_bh["total"]}
