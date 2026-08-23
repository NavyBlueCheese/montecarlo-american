"""
binomial_tree.py

Cox-Ross-Rubinstein (CRR) binomial tree pricer for American options.

This is the ground-truth benchmark the LSM Monte Carlo price is checked
against. A binomial tree handles early exercise "for free": at every node
you can compare the value of exercising immediately to the value of
continuing (the discounted expected value of the two children nodes), and
simply keep whichever is larger. With enough steps the tree price converges
to the true continuous-time American option value, so it's a reliable
reference even though it isn't the object of the LSM lesson itself.

Up/down factors and the risk-neutral probability, per Cox-Ross-Rubinstein
(1979):

    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp((r - q) * dt) - d) / (u - d)
"""

import numpy as np


def crr_american_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    n_steps: int = 2000,
    q: float = 0.0,
) -> float:
    """
    Price an American call or put with a CRR binomial tree.

    Vectorized over each layer of the tree (no Python-level loop over
    nodes), so n_steps=2000+ still runs quickly.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} outside (0,1); "
            "reduce dt (increase n_steps) or check r, sigma, q."
        )

    # stock prices at maturity: s0 * u^j * d^(n-j) for j = 0..n_steps
    j = np.arange(n_steps + 1)
    s_T = s0 * u**j * d ** (n_steps - j)

    if option_type == "put":
        values = np.maximum(k - s_T, 0.0)
    elif option_type == "call":
        values = np.maximum(s_T - k, 0.0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    # step backward through the tree
    for i in range(n_steps - 1, -1, -1):
        j = np.arange(i + 1)
        s_i = s0 * u**j * d ** (i - j)
        continuation = disc * (p * values[1 : i + 2] + (1 - p) * values[0 : i + 1])
        if option_type == "put":
            exercise = np.maximum(k - s_i, 0.0)
        else:
            exercise = np.maximum(s_i - k, 0.0)
        values = np.maximum(continuation, exercise)

    return float(values[0])


if __name__ == "__main__":
    params = dict(s0=100, k=100, r=0.06, sigma=0.2, T=1.0)

    for opt_type in ("put", "call"):
        price = crr_american_price(option_type=opt_type, n_steps=2000, **params)
        print(f"American {opt_type} (CRR, 2000 steps): {price:.4f}")

    # sanity check: an American call on a non-dividend stock should equal
    # the European call, since early exercise is never optimal (no dividend
    # to capture, and time value of money makes waiting weakly better).
    from european_pricer import black_scholes_price

    euro_call = black_scholes_price(option_type="call", **params)
    amer_call = crr_american_price(option_type="call", n_steps=2000, **params)
    print(f"\nEuropean call (BS): {euro_call:.4f}")
    print(f"American call (CRR): {amer_call:.4f}")
    print(f"Difference: {abs(euro_call - amer_call):.6f} (should be ~0)")
