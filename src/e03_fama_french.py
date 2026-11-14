"""E03 — Fama-French three-factor model (Hilpisch "Python for
Finance": asset pricing — CAPM, Fama-French 3-factor, Carhart
4-factor).

CFA idea: CAPM says one factor (the market) explains returns. Fama
and French (1993) said: not enough — small stocks and cheap
(value) stocks earn different returns, so add two more factors:
- SMB (Small Minus Big): return of small-cap stocks minus large-cap.
  A positive SMB loading means the stock behaves like a small stock.
- HML (High Minus Low): return of high book-to-market (value)
  stocks minus low book-to-market (growth) stocks. A positive HML
  loading means the stock behaves like a value stock.
The regression is: R - Rf = alpha + beta_mkt*(Mkt-Rf) + beta_smb*SMB
+ beta_hml*HML + error. The factors come from Kenneth French's data
library (Dartmouth) — the same source every academic paper uses —
downloaded once and cached in data/ff_factors_daily.csv.

How to read the result:
- Alpha: the return left over after all three factors. A
  significant positive alpha = the stock beat what its factor
  exposures predict (skill or luck — with 5 years of daily data,
  check the t-stat, not the sign). A zero alpha = the factors
  explain the stock (the model "works").
- Factor loadings: beta_mkt ~1 = moves with the market; SMB
  negative = behaves like a LARGE stock (expected for Apple);
  HML negative = behaves like a GROWTH stock (expected — Apple is
  a growth stock, not a value stock). The SIGNS are the test: if
  Apple's HML loading came out positive, something would be wrong.
- R² vs CAPM: the three-factor R² should be a little higher than
  the one-factor R². If it barely moves, the extra factors add
  nothing for this stock — which is itself informative.
- t-stats: |t| > 2 = the loading is real. A loading with |t| < 2 is
  indistinguishable from zero — do not build a story on it.
"""
import os

import numpy as np
import pandas as pd

from common import DATA_DIR

FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
          "ftp/F-F_Research_Data_Factors_daily_CSV.zip")
CACHE = os.path.join(DATA_DIR, "ff_factors_daily.csv")


def load_ff_factors():
    """Kenneth French daily factors, cached locally after first download.
    Returns a DataFrame with Mkt-RF, SMB, HML, RF in decimal (per day)."""
    if os.path.exists(CACHE):
        ff = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        if len(ff) > 1000:
            return ff
    import urllib.request
    import zipfile
    from io import BytesIO

    print("  Downloading Fama-French daily factors (Dartmouth) ...")
    with urllib.request.urlopen(FF_URL, timeout=60) as resp:
        data = resp.read()
    with zipfile.ZipFile(BytesIO(data)) as z:
        name = z.namelist()[0]
        raw = z.read(name).decode("latin1")
    lines = raw.splitlines()
    header_idx = next(i for i, l in enumerate(lines)
                      if l.startswith(",Mkt-RF"))
    rows = []
    for line in lines[header_idx + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue  # footer text / blank lines
        rows.append(parts)
    ff = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    ff["date"] = pd.to_datetime(ff["date"], format="%Y%m%d")
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100.0
    ff = ff.set_index("date").dropna()
    ff.to_csv(CACHE)
    return ff


def run(prices, source):
    import statsmodels.api as sm

    print("E03  Fama-French three-factor model (AAPL)")
    ff = load_ff_factors()
    ret = prices.pct_change().dropna()
    frame = pd.DataFrame({
        "aapl_excess": ret["AAPL"] - ff["RF"],
        "mkt_rf": ff["Mkt-RF"], "smb": ff["SMB"], "hml": ff["HML"],
    }).dropna()
    # Align to the price sample window.
    frame = frame.loc[ret.index[0]:ret.index[-1]]
    print(f"  {len(frame)} overlapping days "
          f"({frame.index[0].date()} to {frame.index[-1].date()}); "
          f"factors from Kenneth French's data library (Dartmouth)")

    y = frame["aapl_excess"]
    capm = sm.OLS(y, sm.add_constant(frame[["mkt_rf"]])).fit()
    ff3 = sm.OLS(y, sm.add_constant(
        frame[["mkt_rf", "smb", "hml"]])).fit()
    print(f"  CAPM: alpha {float(capm.params['const']):+.4%}/day "
          f"(t {float(capm.tvalues['const']):+.1f}, "
          f"p {float(capm.pvalues['const']):.3g}), market beta "
          f"{float(capm.params['mkt_rf']):.2f}, R² {float(capm.rsquared):.3f}")
    print("  Fama-French 3-factor:")
    for name in ["const", "mkt_rf", "smb", "hml"]:
        p = float(ff3.pvalues[name])
        print(f"    {name:>7}: {float(ff3.params[name]):+.4f} "
              f"(t {float(ff3.tvalues[name]):+.1f}, p {p:.3g}) -> "
              f"{'significant' if p < 0.05 else 'NOT significant'}")
    print(f"  Reading: SMB loading {float(ff3.params['smb']):+.2f} — "
          f"{'negative, as expected: Apple behaves like a LARGE stock' if float(ff3.params['smb']) < 0 else 'positive (unexpected for a mega-cap — check the data)'}; "
          f"HML loading {float(ff3.params['hml']):+.2f} — "
          f"{'negative, as expected: Apple behaves like a GROWTH stock, not a value stock' if float(ff3.params['hml']) < 0 else 'positive (unexpected for Apple — growth stocks load negative on HML)'}")
    print(f"  Alpha {float(ff3.params['const']):+.4%}/day "
          f"(p {float(ff3.pvalues['const']):.3g}) -> "
          f"{'significant alpha — the stock beat its factor exposures (skill, luck, or a missing factor)' if float(ff3.pvalues['const']) < 0.05 else 'alpha indistinguishable from zero — the three factors explain the stock, the model works'}")
    print(f"  R² CAPM {float(capm.rsquared):.3f} -> FF3 "
          f"{float(ff3.rsquared):.3f}: the extra factors add "
          f"{float(ff3.rsquared - capm.rsquared):.3f} of explained variance — "
          f"{'a real improvement' if float(ff3.rsquared - capm.rsquared) > 0.02 else 'almost nothing new — for a mega-cap growth stock the market factor does nearly all the work'}")
    return {"ff3_alpha": float(ff3.params["const"]),
            "ff3_alpha_p": float(ff3.pvalues["const"]),
            "ff3_beta_mkt": float(ff3.params["mkt_rf"]),
            "ff3_beta_smb": float(ff3.params["smb"]),
            "ff3_beta_hml": float(ff3.params["hml"]),
            "ff3_r2": float(ff3.rsquared),
            "capm_r2": float(capm.rsquared)}
