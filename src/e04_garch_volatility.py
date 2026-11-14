"""E04 — GARCH volatility (Hilpisch "Python for Finance": ARCH/GARCH
processes and volatility modelling).

CFA idea: volatility CLUSTERS — big days follow big days, calm
follows calm. A rolling standard deviation sees this but reacts
slowly and forgets abruptly (a crash drops out of the window 20
days later as if it never happened). GARCH(1,1) models it
properly: today's variance is a weighted average of three things —
a long-run level (omega), yesterday's squared surprise (alpha:
how much a shock matters), and yesterday's variance (beta: how
much volatility persists). The model is estimated by maximum
likelihood — the parameters that make the observed returns most
probable — implemented here from scratch with scipy, so every
step is visible (no black-box library).

The variance equation: sigma²_t = omega + alpha * eps²_{t-1}
+ beta * sigma²_{t-1}, where eps is yesterday's demeaned return.

How to read the result:
- alpha: the shock coefficient. 0.05-0.15 is typical — a 2%
  surprise today raises tomorrow's variance by alpha x 0.02².
  alpha near 0 = shocks do not matter; alpha above 0.2 = the
  market overreacts to every surprise.
- beta: the persistence. 0.80-0.90 is typical — most of today's
  variance carries into tomorrow. beta near 1 = volatility barely
  decays (a crisis lasts for months).
- alpha + beta (persistence): below 1 = stationary (volatility
  mean-reverts); at/above 1 = non-stationary (shocks never die).
  0.95-0.99 is normal for equities — shocks die, but slowly.
- Half-life of a vol shock: ln(0.5)/ln(alpha+beta) days. 10-20 days
  is typical; 50+ days means a crisis echoes for a quarter.
- Unconditional (long-run) volatility: sqrt(omega / (1 - alpha -
  beta)), annualised. Compare it with the plain historical vol —
  they should be close; a big gap means the sample is dominated
  by one regime.
- 10-day-ahead forecast: GARCH mean-reverts toward the long-run
  level, so the forecast sits between today's conditional vol and
  the long-run vol. If the forecast is far above the trailing
  realised vol, the model is saying "the recent calm will not last".
"""
import numpy as np


def _garch_loglik(params, eps):
    """Negative log-likelihood of a GARCH(1,1) with normal errors."""
    omega, alpha, beta = params
    n = len(eps)
    var = np.empty(n)
    var[0] = np.var(eps)
    for t in range(1, n):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
    var = np.maximum(var, 1e-12)
    return 0.5 * float(np.sum(np.log(2 * np.pi * var)
                              + eps ** 2 / var))


def fit_garch(eps):
    """Maximum-likelihood GARCH(1,1). Returns (omega, alpha, beta)."""
    from scipy.optimize import minimize

    var0 = float(np.var(eps))
    x0 = np.array([var0 * 0.05, 0.08, 0.88])
    bounds = [(1e-10, None), (0.0, 0.5), (0.0, 0.999)]
    cons = {"type": "ineq",
            "fun": lambda p: 0.999 - (p[1] + p[2])}  # alpha+beta < 1
    res = minimize(_garch_loglik, x0, args=(eps,), method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 500})
    if not res.success:
        raise RuntimeError(f"GARCH fit failed: {res.message}")
    return res.x


def conditional_vol(eps, params):
    omega, alpha, beta = params
    var = np.empty(len(eps))
    var[0] = np.var(eps)
    for t in range(1, len(eps)):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
    return np.sqrt(np.maximum(var, 1e-12))


def run(prices, source):
    ret = prices["SPY"].pct_change().dropna()
    eps = (ret - ret.mean()).to_numpy()
    print("E04  GARCH(1,1) volatility — SPY daily returns, from scratch")
    omega, alpha, beta = fit_garch(eps)
    persistence = alpha + beta
    half_life = float(np.log(0.5) / np.log(persistence)) \
        if persistence < 1 else float("inf")
    long_run_daily = float(np.sqrt(omega / (1 - persistence)))
    long_run_ann = long_run_daily * np.sqrt(252)
    hist_ann = float(ret.std() * np.sqrt(252))
    cond = conditional_vol(eps, (omega, alpha, beta))
    last_cond_ann = float(cond[-1] * np.sqrt(252))
    # 10-day-ahead variance forecast (mean-reverting recursion).
    var_f = cond[-1] ** 2
    for _ in range(10):
        var_f = omega + (alpha + beta) * var_f
    fcst_ann = float(np.sqrt(var_f) * np.sqrt(252))
    realised_20d = float(ret.tail(20).std() * np.sqrt(252))
    print(f"  omega {omega:.3g}, alpha {alpha:.3f}, beta {beta:.3f}")
    print(f"  alpha {alpha:.3f} -> {'a typical shock response: yesterday’s surprise moves today’s variance, but modestly' if 0.03 < alpha < 0.2 else 'outside the usual 0.05-0.15 range — check the data'}")
    print(f"  beta {beta:.3f} -> {'typical persistence: most of today’s variance carries into tomorrow' if 0.75 < beta < 0.95 else 'outside the usual 0.80-0.90 range'}")
    print(f"  Persistence (alpha+beta) {persistence:.3f} -> "
          f"{'stationary: shocks die out (good — the model is stable)' if persistence < 1 else 'NON-STATIONARY: shocks never die — the model is misspecified'}; "
          f"half-life of a volatility shock {half_life:.1f} days — "
          f"{'a crisis echoes for about two weeks' if half_life < 25 else 'shocks linger for over a month'}")
    print(f"  Long-run volatility {long_run_ann:.1%}/yr vs historical "
          f"{hist_ann:.1%}/yr — "
          f"{'they agree, so the sample is not dominated by one regime' if abs(long_run_ann - hist_ann) < 0.03 else 'they disagree — the sample mixes regimes (e.g. a crisis and a calm stretch)'}")
    print(f"  Conditional vol today {last_cond_ann:.1%}/yr; 10-day "
          f"forecast {fcst_ann:.1%}/yr; trailing 20-day realised vol "
          f"{realised_20d:.1%}/yr — the forecast sits between today's "
          f"conditional vol and the long-run level because GARCH "
          f"mean-reverts; if it is far above realised vol, the model "
          f"says the recent calm will not last")
    return {"garch_alpha": float(alpha), "garch_beta": float(beta),
            "garch_persistence": float(persistence),
            "garch_half_life_days": half_life,
            "garch_long_run_vol_ann": long_run_ann,
            "garch_forecast_10d_ann": fcst_ann,
            "garch_cond_vol_today_ann": last_cond_ann}
