"""C05 — Fixed income: the yield curve, bond pricing, duration and
convexity (CFA L2 Fixed Income).

CFA idea: a bond is a bundle of future cash flows, and the yield curve
is the market's price of money at every horizon. Duration measures how
much the price moves when yields move; convexity measures how that
sensitivity itself changes. The two together give a second-order price
approximation that is accurate enough for most risk work.

Worked example: REAL Treasury yields from Yahoo Finance (^IRX 13-week,
^FVX 5-year, ^TNX 10-year, ^TYX 30-year). Linearly interpolate the
zero curve between them, price a 10-year 5% coupon bond off that curve,
and report its Macaulay duration, modified duration, convexity, and the
price change for a 1% yield shock — exact vs duration-only vs
duration+convexity, so you can see what convexity adds.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq


def fetch_treasury_yields():
    """Latest yields (%) for 13-week, 5y, 10y, 30y Treasuries from
    Yahoo Finance (^IRX, ^FVX, ^TNX, ^TYX). Returns a dict of
    {years: yield_decimal}. Falls back to stated levels if the fetch
    fails — and says so."""
    symbols = {"^IRX": 0.25, "^FVX": 5.0, "^TNX": 10.0, "^TYX": 30.0}
    out = {}
    try:
        import yfinance as yf

        raw = yf.download(list(symbols), period="5d", interval="1d",
                          auto_adjust=True, progress=False)
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) \
            else raw[["Close"]]
        for sym, yrs in symbols.items():
            out[yrs] = float(closes[sym].iloc[-1]) / 100.0
        return out, "REAL (Yahoo Finance Treasury yield indices)"
    except Exception as exc:  # noqa: BLE001 - any failure -> fallback
        print(f"  Treasury yield fetch failed ({exc}); using stated levels.")
        return {0.25: 0.042, 5.0: 0.041, 10.0: 0.045, 30.0: 0.048}, \
            "ASSUMED (fetch failed)"


def zero_rate(t, curve):
    """Zero rate at horizon t (years) by linear interpolation on the
    observed points, flat outside the observed range."""
    xs = np.array(sorted(curve))
    ys = np.array([curve[x] for x in xs])
    return float(np.interp(t, xs, ys))


def bond_metrics(face, coupon_rate, years, curve, payments_per_year=2):
    """Price a bond off the zero curve; return price, Macaulay duration
    (years), modified duration, convexity, and the exact price after a
    +1% parallel yield shock."""
    n = int(years * payments_per_year)
    coupon = face * coupon_rate / payments_per_year
    times = np.array([(i + 1) / payments_per_year for i in range(n)])
    cfs = np.full(n, coupon)
    cfs[-1] += face
    dfs = np.array([np.exp(-zero_rate(t, curve) * t) for t in times])
    pv = cfs * dfs
    price = float(pv.sum())
    mac_dur = float((times * pv).sum() / price)
    ytm = brentq(lambda y: sum(
        cf / (1 + y / payments_per_year) ** (t * payments_per_year)
        for t, cf in zip(times, cfs)) - price, 0.001, 0.5)
    mod_dur = mac_dur / (1 + ytm / payments_per_year)
    convex = float((times * (times + 1 / payments_per_year) * pv).sum()
                   / price / (1 + ytm / payments_per_year) ** 2)
    shocked = {t: r + 0.01 for t, r in curve.items()}
    dfs_up = np.array([np.exp(-zero_rate(t, shocked) * t) for t in times])
    price_up = float((cfs * dfs_up).sum())
    return {"price": price, "ytm": float(ytm), "mac_duration": mac_dur,
            "mod_duration": float(mod_dur), "convexity": convex,
            "price_up_1pct": price_up}


def run(prices, source):
    curve, curve_source = fetch_treasury_yields()
    m = bond_metrics(1_000, 0.05, 10, curve)
    # Approximation check: duration-only vs duration+convexity.
    dy = 0.01
    approx_dur = m["price"] * (1 - m["mod_duration"] * dy)
    approx_both = m["price"] * (1 - m["mod_duration"] * dy
                                + 0.5 * m["convexity"] * dy ** 2)
    exact_change = m["price_up_1pct"] - m["price"]

    print("C05  Fixed income — Treasury curve, duration, convexity")
    print(f"  Treasury yields [{curve_source}]: " + ", ".join(
        f"{y:g}y {r:.2%}" for y, r in sorted(curve.items())))
    print(f"  10y 5% coupon $1,000 bond: price ${m['price']:.2f}, "
          f"YTM {m['ytm']:.2%}")
    print(f"  Macaulay duration {m['mac_duration']:.2f}y, modified "
          f"duration {m['mod_duration']:.2f}, convexity {m['convexity']:.1f}")
    print(f"  +1% yield shock: exact new price ${m['price_up_1pct']:.2f} "
          f"(change ${exact_change:+.2f})")
    print(f"    duration-only estimate: ${approx_dur:.2f} "
          f"(error ${approx_dur - m['price_up_1pct']:+.2f}); "
          f"duration+convexity: ${approx_both:.2f} "
          f"(error ${approx_both - m['price_up_1pct']:+.2f})")
    return {"curve_source": curve_source,
            "y10": float(curve[10.0]), "bond_price": m["price"],
            "bond_ytm": m["ytm"], "mac_duration": m["mac_duration"],
            "mod_duration": m["mod_duration"],
            "convexity": m["convexity"],
            "price_change_1pct": float(exact_change)}
