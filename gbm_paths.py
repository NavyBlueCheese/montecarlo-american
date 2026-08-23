"""
gbm_paths.py

Simulates paths of Geometric Brownian Motion (GBM):

    dS_t = r * S_t * dt + sigma * S_t * dW_t

using the exact lognormal solution (not Euler discretization, since GBM has
a closed-form transition density):

    S_{t+dt} = S_t * exp( (r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )

where Z ~ N(0,1). q is a continuous dividend yield (0 for the base case).

Antithetic variates are supported as a simple variance-reduction option:
for every random draw Z we also use -Z, which cuts the variance of the
Monte Carlo estimator when the payoff is monotonic in the path (as option
payoffs typically are) without introducing any bias.
"""

import numpy as np


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    antithetic: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM stock price paths.

    Parameters
    ----------
    s0 : initial stock price
    r : risk-free rate (continuously compounded)
    sigma : volatility
    T : time horizon in years
    n_steps : number of time steps between 0 and T
    n_paths : number of simulated paths. If antithetic=True, this must be
              even; half the paths are driven by Z, half by -Z.
    q : continuous dividend yield (default 0, non-dividend paying stock)
    antithetic : if True, use antithetic variates for variance reduction
    seed : optional random seed for reproducibility

    Returns
    -------
    paths : ndarray of shape (n_paths, n_steps + 1)
            paths[:, 0] is always s0. paths[:, n_steps] is the price at T.
    """
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must both be positive")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])  # (n_paths, n_steps)
    else:
        z = rng.standard_normal((n_paths, n_steps))

    log_increments = drift + vol * z
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])

    paths = s0 * np.exp(log_paths)
    return paths


if __name__ == "__main__":
    # quick smoke test
    paths = simulate_gbm_paths(
        s0=100, r=0.06, sigma=0.2, T=1.0, n_steps=50, n_paths=10, seed=1
    )
    print("paths shape:", paths.shape)
    print("all start at s0:", np.allclose(paths[:, 0], 100.0))
    print("sample terminal prices:", np.round(paths[:5, -1], 2))
