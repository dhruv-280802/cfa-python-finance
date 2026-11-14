"""E12 — Drawdown analysis (Hilpisch "Python for Finance": risk
management; the risk number investors actually FEEL).

CFA idea: volatility is an abstraction — nobody lies awake over a
standard deviation. What investors feel is the DRAWDOWN: how far
the portfolio has fallen from its peak, and how long it stays
underwater. This module measures it properly.
- Maximum drawdown (MDD): the worst peak-to-trough fall. -20%
  means an investor who bought the peak watched a fifth of their
  money disappear before any recovery.
- Drawdown duration: how long the portfolio stayed below its peak.
  A -20% drawdown that recovers in 3 months is a scare; one that
  lasts 3 years is a career event.
- Time to recovery: peak to new peak. Long recoveries are where
  investors capitulate and sell at the bottom — the behavioural
  cost of drawdowns is larger than the arithmetic one.
- Calmar ratio: annualised return / |max drawdown|. Above 1.0 is
  strong (you earn more per year than your worst loss); below 0.5
  is weak (the worst loss eats two years of returns). It is the
  ratio that answers "is the return worth the pain?"
- Underwater periods: how much of the time the portfolio spends
  below its peak at all. A strategy that is underwater 40% of the
  time tests an investor's patience even if it ends higher.

Worked example: equal-weight portfolio (SPY/AAPL/MSFT/TLT) vs SPY
alone — same 5 years, same data, so the comparison is clean.
"""
import numpy as np
import pandas as pd


def _drawdown_stats(equity, label):
    """Peak, trough, duration and recovery for one equity curve."""
    equity = pd.Series(equity)
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    trough_date = dd.idxmin()
    peak_before = equity.loc[:trough_date].idxmax()
    # Recovery: first date after the trough back at/above the peak.
    peak_value = equity.loc[peak_before]
    after = equity.loc[trough_date:]
    recovered = after[after >= peak_value]
    recovery_date = recovered.index[0] if len(recovered) else None
    duration = (trough_date - peak_before).days
    recovery_days = (recovery_date - peak_before).days \
        if recovery_date is not None else None
    underwater = float((dd < 0).mean())
    return {"max_dd": max_dd, "trough": trough_date,
            "peak_before": peak_before, "duration_days": duration,
            "recovery_days": recovery_days, "underwater_share": underwater}


def run(prices, source):
    ret = prices.pct_change().dropna()
    print("E12  Drawdown analysis — what investors actually feel")
    eq_ew = (1 + ret.mean(axis=1)).cumprod()
    eq_spy = (1 + ret["SPY"]).cumprod()
    n_years = len(ret) / 252
    for equity, label in ((eq_ew, "Equal-weight portfolio"),
                          (eq_spy, "SPY buy-and-hold")):
        s = _drawdown_stats(equity, label)
        cagr = float(equity.iloc[-1] ** (1 / n_years) - 1)
        calmar = cagr / abs(s["max_dd"]) if s["max_dd"] != 0 else np.nan
        print(f"  {label}:")
        print(f"    Max drawdown {s['max_dd']:.1%} (peak "
              f"{s['peak_before'].date()} -> trough {s['trough'].date()}, "
              f"{s['duration_days']} days peak-to-trough) — "
              f"{'a severe fall: a fifth or more of the money gone at the worst point' if s['max_dd'] < -0.2 else 'a moderate worst fall' if s['max_dd'] < -0.1 else 'a mild worst fall'}")
        print(f"    Recovery: "
              f"{s['recovery_days']} days peak-to-new-peak"
              if s["recovery_days"] is not None else
              "not yet recovered by the end of the sample — the portfolio finished underwater")
        print(f"    Underwater {s['underwater_share']:.0%} of the time — "
              f"{'almost the whole journey is spent below a previous peak — new highs are brief, and most days feel like a loss even in a rising market' if s['underwater_share'] > 0.7 else 'nearly half the journey is spent below a previous peak; that is normal, and it is why investors sell at the bottom' if s['underwater_share'] > 0.3 else 'mostly at or near highs'}")
        print(f"    CAGR {cagr:.1%}/yr, Calmar ratio {calmar:.2f} — "
              f"{'strong: each year’s return exceeds the worst loss' if calmar > 1 else 'weak: the worst loss eats more than a year of returns' if calmar < 0.5 else 'middling: the worst loss costs roughly a year of returns'}")
    ew = _drawdown_stats(eq_ew, "ew")
    spy = _drawdown_stats(eq_spy, "spy")
    print(f"  Reading: the equal-weight portfolio's max drawdown "
          f"({ew['max_dd']:.1%}) vs SPY's ({spy['max_dd']:.1%}) — "
          f"{'diversification softened the worst fall' if ew['max_dd'] > spy['max_dd'] else 'diversification did NOT soften the worst fall — in this window the assets fell together'}")
    return {"dd_ew_max": float(ew["max_dd"]),
            "dd_spy_max": float(spy["max_dd"]),
            "dd_ew_underwater": float(ew["underwater_share"]),
            "dd_ew_recovery_days": float(ew["recovery_days"])
            if ew["recovery_days"] is not None else np.nan}
