"""
european_pricer.py

Two ways of pricing a plain vanilla European option, used only to sanity
check gbm_paths.py before the more complex LSM machinery is added:

1. Monte Carlo: simulate GBM paths to expiry, compute the payoff at T on
   each path, discount, and average. Because a European option can only be
   exercised at expiry, there is no early-exercise decision to make, so
   plain forward simulation is all that's needed.

2. Black-Scholes closed form: the analytic price, used as ground truth for
   the Monte Carlo estimate. Agreement between the two (within the Monte
   Carlo confidence interval) confirms the path simulator is correct.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from gbm_paths import simulate_gbm_paths


@dataclass
class MCResult:
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int

    def __repr__(self):
        return (
            f"MCResult(price={self.price:.4f}, std_error={self.std_error:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], n_paths={self.n_paths})"
        )


def mc_confidence_interval(discounted_payoffs: np.ndarray) -> MCResult:
    """
    Given an array of discounted payoffs (one per simulated path), compute
    the Monte Carlo price estimate, its standard error, and a 95% CI.

    The estimator is just the sample mean; by the CLT its standard error is
    the sample standard deviation divided by sqrt(n_paths), and a 95% CI
    uses the usual +/- 1.96 * std_error (z, not t, since n_paths is large).
    """
    n = len(discounted_payoffs)
    price = float(np.mean(discounted_payoffs))
    se = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n))
    return MCResult(
        price=price,
        std_error=se,
        ci_lower=price - 1.96 * se,
        ci_upper=price + 1.96 * se,
        n_paths=n,
    )


def european_mc_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    n_paths: int = 100_000,
    n_steps: int = 50,
    q: float = 0.0,
    antithetic: bool = False,
    seed: int | None = None,
) -> MCResult:
    """Price a European call or put by Monte Carlo simulation."""
    paths = simulate_gbm_paths(
        s0, r, sigma, T, n_steps, n_paths, q=q, antithetic=antithetic, seed=seed
    )
    s_T = paths[:, -1]
    if option_type == "put":
        payoff = np.maximum(k - s_T, 0.0)
    elif option_type == "call":
        payoff = np.maximum(s_T - k, 0.0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    discounted = np.exp(-r * T) * payoff
    return mc_confidence_interval(discounted)


def black_scholes_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    q: float = 0.0,
) -> float:
    """Closed-form Black-Scholes price for a European call or put."""
    if T <= 0:
        intrinsic = max(s0 - k, 0.0) if option_type == "call" else max(k - s0, 0.0)
        return intrinsic

    d1 = (np.log(s0 / k) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = s0 * np.exp(-q * T) * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = k * np.exp(-r * T) * norm.cdf(-d2) - s0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'put' or 'call'")
    return float(price)


if __name__ == "__main__":
    params = dict(s0=100, k=100, r=0.06, sigma=0.2, T=1.0)

    for opt_type in ("put", "call"):
        bs = black_scholes_price(option_type=opt_type, **params)
        mc = european_mc_price(
            option_type=opt_type, n_paths=200_000, n_steps=50, antithetic=True, seed=42, **params
        )
        in_ci = mc.ci_lower <= bs <= mc.ci_upper
        print(f"\nEuropean {opt_type}:")
        print(f"  Black-Scholes price : {bs:.4f}")
        print(f"  Monte Carlo         : {mc}")
        print(f"  BS price inside MC 95% CI: {in_ci}")
