"""E11 — Performance attribution (Brinson) (CFA L3 portfolio
management: return attribution).

CFA idea: a portfolio beat its benchmark — WHY? Brinson attribution
splits the active return into three pieces, using two asset classes
here (equities: SPY/AAPL/MSFT, bonds: TLT):
- ALLOCATION effect: the gain/loss from weighting the classes
  differently from the benchmark. Overweighting the class that did
  better earns a positive allocation effect.
- SELECTION effect: the gain/loss from picking better securities
  WITHIN a class than the benchmark holds. (Here the "selection"
  inside equities is the equal-weighted SPY/AAPL/MSFT trio vs the
  benchmark's SPY-only equities.)
- INTERACTION effect: the cross-term (overweighting a class AND
  picking better securities in it). Usually small; a large
  interaction means allocation and selection are tangled together.

How to read the result:
- Active return = portfolio return - benchmark return. The three
  effects must SUM to it (that is the arithmetic check — if they
  do not sum, the attribution is wrong).
- A positive allocation effect with a negative selection effect
  says: "right asset classes, wrong stocks". The reverse says:
  "wrong classes, right stocks". Both matter — a manager who only
  ever gets allocation right is a timer, not a picker.
- The benchmark here is 60% equities (SPY) / 40% bonds (TLT); the
  portfolio is equal weight (25% each asset = 75% equities /
  25% bonds). The attribution is computed year by year and summed.
"""
import numpy as np
import pandas as pd


def run(prices, source):
    ret = prices.pct_change().dropna()
    print("E11  Performance attribution (Brinson) — equal-weight vs 60/40")
    # Portfolio: 25% in each asset. Benchmark: 60% SPY (equities) / 40% TLT.
    eq_names = ["SPY", "AAPL", "MSFT"]
    # Full calendar years only: the sample starts 2021-11-15 and ends
    # 2026-11-13, so annualising the 2021 stub (6 weeks) or the 2026
    # stub would fabricate a full year out of a few weeks.
    years = [y for y in sorted(ret.index.year.unique())
             if len(ret[ret.index.year == y]) >= 200]
    skipped = sorted(set(ret.index.year.unique()) - set(years))
    print(f"  Full years in the study: {years} "
          f"(partial years {skipped} excluded — annualising a stub "
          f"year would invent returns that never happened)")
    rows = []
    for yr in years:
        r = ret[ret.index.year == yr]
        # Proper annual class returns: compound the daily class returns.
        eq_daily = r[eq_names].mean(axis=1)
        bond_daily = r["TLT"]
        eq_ann = float((1 + eq_daily).prod() ** (252 / len(r)) - 1)
        bond_ann = float((1 + bond_daily).prod() ** (252 / len(r)) - 1)
        bench_eq_ann = float((1 + r["SPY"]).prod() ** (252 / len(r)) - 1)
        bench_bond_ann = bond_ann  # benchmark holds the same bond
        w_p_eq, w_p_bond = 0.75, 0.25
        w_b_eq, w_b_bond = 0.60, 0.40
        bench_ret = w_b_eq * bench_eq_ann + w_b_bond * bench_bond_ann
        port_ret = w_p_eq * eq_ann + w_p_bond * bond_ann
        alloc = ((w_p_eq - w_b_eq) * (bench_eq_ann - bench_ret)
                 + (w_p_bond - w_b_bond) * (bench_bond_ann - bench_ret))
        select = (w_b_eq * (eq_ann - bench_eq_ann)
                  + w_b_bond * (bond_ann - bench_bond_ann))
        interact = ((w_p_eq - w_b_eq) * (eq_ann - bench_eq_ann)
                    + (w_p_bond - w_b_bond) * (bond_ann - bench_bond_ann))
        rows.append({"year": yr, "port": port_ret, "bench": bench_ret,
                     "active": port_ret - bench_ret, "alloc": alloc,
                     "select": select, "interact": interact})
    table = pd.DataFrame(rows)
    totals = table[["active", "alloc", "select", "interact"]].sum()
    check = float(totals["alloc"] + totals["select"]
                  + totals["interact"] - totals["active"])
    print("  Year-by-year active return (portfolio - benchmark):")
    for _, row in table.iterrows():
        print(f"    {int(row['year'])}: active {row['active']:+.1%} "
              f"(alloc {row['alloc']:+.1%}, select {row['select']:+.1%}, "
              f"interact {row['interact']:+.1%})")
    print(f"  TOTAL active return {totals['active']:+.1%} = allocation "
          f"{totals['alloc']:+.1%} + selection {totals['select']:+.1%} "
          f"+ interaction {totals['interact']:+.1%} "
          f"(sum check: {check:+.4f} — should be ~0)")
    print(f"  Reading: the allocation effect is "
          f"{'positive — overweighting equities (75% vs 60%) paid, because equities beat bonds over this window' if totals['alloc'] > 0 else 'negative — the 75% equity overweight hurt, because bonds beat equities over this window'}; "
          f"the selection effect is "
          f"{'positive — the equal-weighted equity trio beat SPY alone' if totals['select'] > 0 else 'negative — holding AAPL/MSFT alongside SPY did not beat SPY alone'}")
    print("  Lesson: attribution does not say whether the portfolio "
          "was GOOD — it says WHERE the return came from. A manager "
          "who wins only on allocation is a timer; one who wins only "
          "on selection is a picker")
    return {"attr_active_total": float(totals["active"]),
            "attr_alloc_total": float(totals["alloc"]),
            "attr_select_total": float(totals["select"]),
            "attr_interact_total": float(totals["interact"]),
            "attr_sum_check": check}
