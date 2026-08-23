"""
lsm_pricer.py

Longstaff-Schwartz (2001) Least Squares Monte Carlo (LSM) pricer for
American-style options.

Algorithm sketch
-----------------
1. Simulate n_paths GBM paths from 0 to T on a grid of n_steps time steps.
2. At maturity, the option's value on each path is just its payoff there.
   This is also, for now, each path's "cash flow" -- the amount it will
   actually pay out, and when.
3. Step backward one time slice at a time, from t = T - dt down to t = dt
   (never at t=0, since a decision "now" isn't a choice on a single date --
   note the price itself is read off at t=0 as a simple discounted average,
   no regression needed there). At each time step t_i:
     a. Find the paths that are in the money at t_i (only these are
        candidates for early exercise -- out-of-the-money paths would never
        exercise, and including them would just add noise to the
        regression, which is the refinement Longstaff-Schwartz proposed
        over a naive version of this idea).
     b. For those in-the-money paths, regress the realized discounted
        future cash flow (whatever that path actually pays out later,
        discounted back to t_i, given the exercise decisions already made
        at later times) on a polynomial basis in the current stock price.
        The fitted value is the estimated continuation value: what you'd
        expect to get, in present-value terms at t_i, from holding rather
        than exercising.
     c. Compare the immediate exercise payoff to the estimated continuation
        value, path by path. Where exercise payoff is strictly greater,
        mark that path as exercising at t_i: overwrite its future cash
        flow with the exercise payoff at t_i, and discard any later cash
        flow that path had recorded (once you exercise, nothing after
        matters).
4. After working back to t_1, discount every path's final recorded cash
   flow (and the time it occurs) back to t=0 and average. That average is
   the LSM price. Its standard error and 95% CI come from the same
   discounted cash flows exactly as in the European case.

Basis functions: this implementation uses ordinary polynomials in the
stock price, 1, S, S^2, ..., S^degree (degree 2 or 3 as recommended in the
project brief). Longstaff-Schwartz's original paper uses (shifted)
Laguerre polynomials, but for a single underlying and a low degree, plain
polynomials work essentially the same in practice and are far easier to
read.
"""

from dataclasses import dataclass, field

import numpy as np

from gbm_paths import simulate_gbm_paths
from european_pricer import MCResult, mc_confidence_interval


@dataclass
class LSMResult:
    mc_result: MCResult
    exercise_time: np.ndarray  # per-path index of exercise step, or -1 if never
    stopping_stock_price: np.ndarray  # stock price at the exercise step, or nan
    paths: np.ndarray  # the simulated paths themselves, for plotting
    dt: float
    exercised_early_mask: np.ndarray  # True if exercised before the final step

    @property
    def price(self) -> float:
        return self.mc_result.price


def _payoff(s: np.ndarray, k: float, option_type: str) -> np.ndarray:
    if option_type == "put":
        return np.maximum(k - s, 0.0)
    elif option_type == "call":
        return np.maximum(s - k, 0.0)
    raise ValueError("option_type must be 'put' or 'call'")


def lsm_american_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    n_paths: int = 100_000,
    n_steps: int = 50,
    degree: int = 3,
    q: float = 0.0,
    antithetic: bool = False,
    seed: int | None = None,
) -> LSMResult:
    """
    Price an American option with Longstaff-Schwartz Least Squares Monte
    Carlo. See module docstring for the algorithm.
    """
    paths = simulate_gbm_paths(
        s0, r, sigma, T, n_steps, n_paths, q=q, antithetic=antithetic, seed=seed
    )
    dt = T / n_steps
    disc_1step = np.exp(-r * dt)

    # cash_flow[p] = the (undiscounted, as of its own exercise time) payoff
    # path p will ultimately receive; exercise_step[p] = the step index at
    # which that payoff occurs. Initialize at maturity: if you never
    # exercise early, you take the European payoff at step n_steps.
    cash_flow = _payoff(paths[:, -1], k, option_type)
    exercise_step = np.full(n_paths, n_steps, dtype=int)

    # backward induction from n_steps-1 down to 1 (t=0 has no decision)
    for i in range(n_steps - 1, 0, -1):
        s_i = paths[:, i]
        exercise_value = _payoff(s_i, k, option_type)
        itm = exercise_value > 0.0

        if np.any(itm):
            # discount each path's recorded future cash flow back to t_i
            steps_ahead = exercise_step[itm] - i
            discounted_future_cf = cash_flow[itm] * disc_1step**steps_ahead

            x = s_i[itm]
            basis = np.vander(x, N=degree + 1, increasing=True)  # [1, x, x^2, ...]
            coeffs, *_ = np.linalg.lstsq(basis, discounted_future_cf, rcond=None)
            continuation_value = basis @ coeffs

            ex_val_itm = exercise_value[itm]
            exercise_now = ex_val_itm > continuation_value

            itm_idx = np.where(itm)[0]
            do_exercise_idx = itm_idx[exercise_now]

            cash_flow[do_exercise_idx] = ex_val_itm[exercise_now]
            exercise_step[do_exercise_idx] = i

    discounted_cf = cash_flow * np.exp(-r * dt * exercise_step)
    mc_result = mc_confidence_interval(discounted_cf)

    exercised_early = exercise_step < n_steps
    stopping_price = np.full(n_paths, np.nan)
    idx = np.arange(n_paths)
    stopping_price = paths[idx, exercise_step]

    return LSMResult(
        mc_result=mc_result,
        exercise_time=exercise_step,
        stopping_stock_price=stopping_price,
        paths=paths,
        dt=dt,
        exercised_early_mask=exercised_early,
    )


def exercise_boundary(result: LSMResult, k: float, option_type: str = "put") -> tuple:
    """
    Estimate the exercise boundary: for each time step, the stock price at
    which the algorithm judged exercise to be (just) optimal, based on the
    paths that exercised at exactly that step.

    Returns (time_steps, boundary_prices) with NaN where no path exercised
    at that step (so there is no data to estimate the boundary there).
    For a put, the boundary is the *highest* stopping price observed at
    each step (the boundary is approached from above -- exercise happens
    for prices at or below it). For a call (dividend case), it's the
    lowest.
    """
    n_steps = result.paths.shape[1] - 1
    boundary = np.full(n_steps + 1, np.nan)
    for i in range(1, n_steps):
        mask = result.exercise_time == i
        if np.any(mask):
            prices_here = result.stopping_stock_price[mask]
            boundary[i] = prices_here.max() if option_type == "put" else prices_here.min()
    times = np.arange(n_steps + 1) * result.dt
    return times, boundary


if __name__ == "__main__":
    from binomial_tree import crr_american_price

    params = dict(s0=100, k=100, r=0.06, sigma=0.2, T=1.0)

    print("American put:")
    lsm = lsm_american_price(
        option_type="put", n_paths=100_000, n_steps=50, degree=3,
        antithetic=True, seed=7, **params,
    )
    crr = crr_american_price(option_type="put", n_steps=2000, **params)
    print(f"  LSM  : {lsm.mc_result}")
    print(f"  CRR  : {crr:.4f}")
    print(f"  CRR inside LSM 95% CI: {lsm.mc_result.ci_lower <= crr <= lsm.mc_result.ci_upper}")
    print(f"  fraction exercised early: {lsm.exercised_early_mask.mean():.3f}")
