"""A01 — Variables and data types (CFA L1 PSM: Python Programming Fundamentals).

CFA idea: a variable is a labelled box that holds a value. Python figures
out the type for you — int, float, str, bool — and the type decides what
you can do with it (you can add two floats, but not a float and a str).

Worked example: $10,000 invested in SPY five years ago. How much is it
worth today, and what type is each number along the way?
"""
import numpy as np


def run(prices, source):
    spy = prices["SPY"]
    start_price = float(spy.iloc[0])   # float: a price with decimals
    end_price = float(spy.iloc[-1])
    n_years = (spy.index[-1] - spy.index[0]).days / 365.25  # float
    shares = 10_000 / start_price      # int 10_000 divided by a float
    value_now = shares * end_price
    total_return = value_now / 10_000 - 1
    cagr = (value_now / 10_000) ** (1 / n_years) - 1

    print("A01  Variables & data types")
    print(f"  SPY first close: ${start_price:.2f} ({type(start_price).__name__})")
    print(f"  SPY last close:  ${end_price:.2f} ({type(end_price).__name__})")
    print(f"  $10,000 buys {shares:.2f} shares ({type(shares).__name__})")
    print(f"  Value today: ${value_now:,.2f}; total return {total_return:.1%}; "
          f"CAGR {cagr:.2%} over {n_years:.1f} years")
    print(f"  A whole number is an int ({type(10_000).__name__}); "
          f"a yes/no flag is a bool ({type(total_return > 0).__name__})")
    return {"spy_start": start_price, "spy_end": end_price,
            "value_of_10k": value_now, "spy_total_return": total_return,
            "spy_cagr": cagr}
