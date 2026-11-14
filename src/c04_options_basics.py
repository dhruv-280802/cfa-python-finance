"""C04 — Options basics: Black-Scholes, Greeks, put-call parity
(CFA L2 Derivatives).

CFA idea: an option's value splits into intrinsic value (what you would
get by exercising now) and time value (what the remaining uncertainty
is worth). Black-Scholes prices the time value from five inputs — spot,
strike, time, rate, volatility. The Greeks then say how the price moves
when each input moves: delta (spot), gamma (delta's speed), theta (time
decay), vega (volatility).

Worked example: SPY at its latest close, a 1-month at-the-money call
and put, volatility from the real trailing 60-day realised vol.
Cross-checks: the binomial tree price should match Black-Scholes to
the cent, and put-call parity (call - put = spot - PV(strike)) must
hold exactly — if it doesn't, there is a bug, not an arbitrage.
"""
import numpy as np
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option="call"):
    """Black-Scholes price for a European call or put on a
    non-dividend-paying underlying."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2)
                 - S * norm.cdf(-d1))


def greeks(S, K, T, r, sigma):
    """Delta, gamma, theta (per day), vega (per 1 vol point) for a call."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = float(norm.cdf(d1))
    gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    theta = float(-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
    vega = float(S * norm.pdf(d1) * np.sqrt(T)) / 100
    return {"delta": delta, "gamma": gamma, "theta_per_day": theta,
            "vega_per_1pt": vega}


def binomial_call(S, K, T, r, sigma, n=500):
    """Cox-Ross-Rubinstein binomial tree price for a European call.
    Converges to Black-Scholes as n grows."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    # Terminal payoffs, then fold back one step at a time.
    j = np.arange(n + 1)
    stock = S * (u ** j) * (d ** (n - j))
    values = np.maximum(stock - K, 0.0)
    disc = np.exp(-r * dt)
    for _ in range(n):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
    return float(values[0])


def run(prices, source):
    S = float(prices["SPY"].iloc[-1])
    K = round(S)  # nearest dollar: effectively at-the-money
    T = 1 / 12
    r = 0.04
    rets = prices["SPY"].pct_change().dropna()
    sigma = float(rets.tail(60).std() * np.sqrt(252))  # trailing 60d vol

    call = black_scholes(S, K, T, r, sigma, "call")
    put = black_scholes(S, K, T, r, sigma, "put")
    g = greeks(S, K, T, r, sigma)
    binom = binomial_call(S, K, T, r, sigma)
    intrinsic = max(S - K, 0)
    time_value = call - intrinsic
    parity_lhs = call - put
    parity_rhs = S - K * np.exp(-r * T)
    parity_gap = parity_lhs - parity_rhs

    print("C04  Options — Black-Scholes on SPY")
    print(f"  Inputs: SPY ${S:.2f}, strike ${K}, 1 month, Rf 4%, "
          f"realised vol (60d) {sigma:.1%}")
    print(f"  ATM call: ${call:.2f} = intrinsic ${intrinsic:.2f} + "
          f"time value ${time_value:.2f}; ATM put: ${put:.2f}")
    print(f"  Greeks (call): delta {g['delta']:.2f}, gamma {g['gamma']:.4f}, "
          f"theta {g['theta_per_day']:+.3f}/day, vega {g['vega_per_1pt']:.2f} per vol point")
    print(f"  Binomial tree (500 steps): ${binom:.2f} vs Black-Scholes "
          f"${call:.2f} (diff ${binom - call:+.4f})")
    print(f"  Put-call parity: call - put = ${parity_lhs:.2f}; "
          f"S - PV(K) = ${parity_rhs:.2f}; gap ${parity_gap:+.2e} "
          f"(~0 = the math is consistent)")
    return {"spy_price": S, "strike": float(K), "realised_vol": sigma,
            "call_price": call, "put_price": put,
            "call_delta": g["delta"], "call_gamma": g["gamma"],
            "call_theta": g["theta_per_day"], "call_vega": g["vega_per_1pt"],
            "binomial_price": binom, "parity_gap": float(parity_gap)}
