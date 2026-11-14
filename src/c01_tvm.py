"""C01 — Time value of money (CFA L1 Quant Methods: the foundation
everything else is built on).

CFA idea: a dollar today is worth more than a dollar tomorrow, because
today's dollar can earn interest. Every valuation in finance — bonds,
stocks, projects, loans — is just this idea applied repeatedly: move
money through time at the right rate, then compare.

Worked examples: (1) future value of $10,000 at 5% for 10 years;
(2) present value of $10,000 due in 10 years; (3) present value of a
10-year $1,000/year annuity; (4) the monthly payment on a $300,000
30-year mortgage at 6%; (5) NPV and IRR of a 5-year teaching project
(-$100,000 now, +$30,000 a year for 5 years) at a 10% discount rate;
(6) a bond's price and yield-to-maturity cross-check.
"""
import numpy as np
from scipy.optimize import brentq


def future_value(pv, rate, n):
    """FV = PV * (1 + r)^n."""
    return pv * (1 + rate) ** n


def present_value(fv, rate, n):
    """PV = FV / (1 + r)^n."""
    return fv / (1 + rate) ** n


def annuity_pv(payment, rate, n):
    """PV of an ordinary annuity: PMT * (1 - (1+r)^-n) / r."""
    if rate == 0:
        return payment * n
    return payment * (1 - (1 + rate) ** -n) / rate


def annuity_fv(payment, rate, n):
    """FV of an ordinary annuity: PMT * ((1+r)^n - 1) / r."""
    if rate == 0:
        return payment * n
    return payment * ((1 + rate) ** n - 1) / rate


def loan_payment(principal, annual_rate, years, payments_per_year=12):
    """Fixed payment on a fully amortizing loan."""
    r = annual_rate / payments_per_year
    n = years * payments_per_year
    return principal * r / (1 - (1 + r) ** -n)


def npv(rate, cash_flows):
    """Net present value: cash_flows[0] is at t=0 (usually negative)."""
    return float(sum(cf / (1 + rate) ** t
                     for t, cf in enumerate(cash_flows)))


def irr(cash_flows):
    """Internal rate of return: the discount rate that makes NPV zero,
    found by bisection (brentq)."""
    f = lambda r: sum(cf / (1 + r) ** t
                      for t, cf in enumerate(cash_flows))
    return float(brentq(f, -0.99, 10.0))


def bond_price(face, coupon_rate, ytm, years, payments_per_year=2):
    """Price of a fixed-coupon bond by discounting its cash flows."""
    coupon = face * coupon_rate / payments_per_year
    n = years * payments_per_year
    r = ytm / payments_per_year
    pv_coupons = sum(coupon / (1 + r) ** t for t in range(1, n + 1))
    pv_face = face / (1 + r) ** n
    return pv_coupons + pv_face


def run(prices, source):
    fv = future_value(10_000, 0.05, 10)
    pv = present_value(10_000, 0.05, 10)
    apv = annuity_pv(1_000, 0.05, 10)
    afv = annuity_fv(1_000, 0.05, 10)
    pmt = loan_payment(300_000, 0.06, 30)
    project = [-100_000, 30_000, 30_000, 30_000, 30_000, 30_000]
    project_npv = npv(0.10, project)
    project_irr = irr(project)
    # Bond cross-check: a 10y, 5% coupon, $1,000 face bond priced at a
    # 6% yield, then verify the yield that reproduces that price is 6%.
    price = bond_price(1_000, 0.05, 0.06, 10)
    implied_ytm = brentq(lambda y: bond_price(1_000, 0.05, y, 10) - price,
                         0.001, 0.5)

    print("C01  Time value of money")
    print(f"  $10,000 at 5% for 10y: FV = ${fv:,.2f}")
    print(f"  $10,000 due in 10y at 5%: PV = ${pv:,.2f}")
    print(f"  $1,000/year for 10y at 5%: PV = ${apv:,.2f}, "
          f"FV = ${afv:,.2f}")
    print(f"  $300,000 mortgage, 6%, 30y: monthly payment = ${pmt:,.2f}")
    print(f"  Project (-100k, +30k x5): NPV at 10% = "
          f"${project_npv:,.2f}; IRR = {project_irr:.2%} "
          f"-> {'ACCEPT' if project_npv > 0 else 'REJECT'} (NPV > 0)")
    print(f"  10y 5% bond at 6% yield: price ${price:.2f} (below par, "
          f"as expected); implied YTM from that price: {implied_ytm:.2%}")
    return {"fv_10k": float(fv), "pv_10k": float(pv),
            "annuity_pv": float(apv), "annuity_fv": float(afv),
            "mortgage_pmt": float(pmt), "project_npv": float(project_npv),
            "project_irr": float(project_irr),
            "bond_price": float(price), "bond_ytm": float(implied_ytm)}
