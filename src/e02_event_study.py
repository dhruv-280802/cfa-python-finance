"""E02 — Event study: do earnings surprises move the stock?
(Hilpisch "Python for Finance": event studies and market efficiency.)

CFA idea: an event study asks the market-efficiency question in its
purest form — when news arrives, does the price move, by how much,
and does it move FAST (efficient) or drift (inefficient)? The recipe:
1. Pick the event dates (here: AAPL earnings announcements from
   Yahoo Finance, with the reported EPS surprise).
2. Estimate the NORMAL return with a market model (alpha + beta x
   market) over an estimation window BEFORE the event, so the event
   itself cannot contaminate the benchmark.
3. ABNORMAL return (AR) = actual return - normal return, each day in
   a short window around the event. CAR = the sum over the window.
4. Average the CARs across events and test whether the average is
   significantly different from zero (t-test). A significant
   positive average CAR after positive surprises = the market rewards
   beats; a CAR that keeps growing AFTER the event day = slow
   adjustment (post-earnings-announcement drift — a famous anomaly).

How to read the result:
- CAR per event: the stock's extra return around that earnings day,
  stripped of the market's move. +2% means the stock beat the market
  by 2% around the announcement.
- Average CAR and its t-stat: t = mean / (std / sqrt(n)). |t| > 2
  means the average is unlikely to be zero by chance. A positive
  average CAR with positive surprises supports "the market rewards
  beats"; an average CAR near zero with a small t means earnings
  days are, on average, a coin flip for this stock.
- Surprise vs CAR correlation: if bigger beats earn bigger CARs, the
  market prices the SIZE of the surprise, not just its sign.
- Earnings are reported AFTER the close, so the event day is the
  NEXT trading day — getting this wrong by one day destroys the study.
"""
import os

import numpy as np
import pandas as pd


def run(prices, source):
    import statsmodels.api as sm

    print("E02  Event study — AAPL earnings announcements")
    import yfinance as yf

    from common import DATA_DIR

    # Earnings dates are fetched once and cached (the Yahoo scrape
    # endpoint is flaky — a timeout must skip the study, not kill
    # the whole run).
    cache = os.path.join(DATA_DIR, "aapl_earnings_dates.csv")
    ed = None
    if os.path.exists(cache):
        try:
            ed = pd.read_csv(cache, index_col=0, parse_dates=True)
            if len(ed) < 4:
                ed = None
        except Exception:  # noqa: BLE001 - corrupt cache -> refetch
            ed = None
    if ed is None:
        try:
            ed = yf.Ticker("AAPL").earnings_dates
        except Exception as exc:  # noqa: BLE001 - network/timeout
            print(f"  Earnings dates unavailable ({exc}) — study skipped")
            return {"event_n": 0}
        if ed is not None and len(ed) > 0:
            try:
                ed.to_csv(cache)
            except Exception:  # noqa: BLE001 - cache is a bonus
                pass
    if ed is None or len(ed) == 0:
        print("  Earnings dates unavailable from Yahoo today — study skipped")
        return {"event_n": 0}
    # Only past events with a reported EPS (drop the upcoming one).
    ed = ed[pd.notna(ed["Reported EPS"])].copy()
    ed = ed.sort_index(ascending=False).head(8).sort_index()
    print(f"  {len(ed)} earnings events in the study "
          f"({ed.index[0].date()} to {ed.index[-1].date()})")

    aapl = prices["AAPL"].pct_change()
    spy = prices["SPY"].pct_change()
    frame = pd.DataFrame({"aapl": aapl, "spy": spy}).dropna()
    cars, rows = [], []
    for event_date, row in ed.iterrows():
        event_day = pd.Timestamp(event_date.date())
        # Event day = next trading day (earnings land after the close).
        pos = frame.index.searchsorted(event_day)
        if pos == 0 or pos + 6 >= len(frame) or pos - 121 < 0:
            continue
        event_pos = pos if frame.index[pos] == event_day else pos
        # Estimation window: 120 trading days ending 11 days before.
        est = frame.iloc[event_pos - 121:event_pos - 10]
        X = sm.add_constant(est["spy"])
        model = sm.OLS(est["aapl"], X).fit()
        alpha, beta = float(model.params["const"]), float(model.params["spy"])
        window = frame.iloc[event_pos - 5:event_pos + 6].copy()
        window["ar"] = window["aapl"] - (alpha + beta * window["spy"])
        car = float(window["ar"].sum())
        cars.append(car)
        surprise = float(row["Surprise(%)"])
        rows.append({"date": event_day.date(), "surprise_pct": surprise,
                     "car_pct": car * 100, "beta": beta})
        print(f"  {event_day.date()}: EPS surprise {surprise:+.1f}% -> "
              f"CAR [-5,+5] {car:+.2%} (beta {beta:.2f}) — "
              f"{'the stock beat the market around the announcement' if car > 0.01 else 'the stock lagged the market around the announcement' if car < -0.01 else 'roughly in line with the market'}")
    cars = np.array(cars)
    table = pd.DataFrame(rows)
    mean_car = float(cars.mean())
    t_stat = float(mean_car / (cars.std(ddof=1) / np.sqrt(len(cars)))) \
        if len(cars) > 1 else np.nan
    corr = float(np.corrcoef(table["surprise_pct"], table["car_pct"])[0, 1]) \
        if len(table) > 2 else np.nan
    print(f"  Average CAR {mean_car:+.2%} over {len(cars)} events, "
          f"t-stat {t_stat:+.2f} -> "
          f"{'SIGNIFICANT: earnings days systematically move the stock beyond the market — the market prices the news' if abs(t_stat) > 2 else 'NOT significant: on average, earnings days are a coin flip once the market move is stripped out'}")
    print(f"  Surprise vs CAR correlation {corr:+.2f} -> "
          f"{'bigger beats earn bigger abnormal returns — the market prices the SIZE of the surprise' if corr > 0.3 else 'weak link between the size of the beat and the abnormal return — the market reacts to the news, not its size, or the surprise was already expected'}")
    print("  Caveat: 8 events is a small sample — one wild quarter "
          "moves the average. The method is the lesson; a real study "
          "uses 40+ events")
    return {"event_n": int(len(cars)), "event_mean_car": mean_car,
            "event_t_stat": t_stat, "event_surprise_corr": corr}
