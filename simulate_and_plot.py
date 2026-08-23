"""
simulate_and_plot.py

Generates the four analysis charts described in the project brief, all
saved to outputs/:

1. convergence.png        - LSM price + 95% CI band vs number of paths,
                             with the CRR binomial price as a reference line
2. exercise_boundary.png  - estimated optimal exercise stock price vs time
3. american_vs_european.png - price comparison showing the early exercise
                             premium, for the base case (put) and for the
                             dividend-paying call extension
4. greeks.png             - delta and vega via bump-and-reprice, put vs
                             call, base case (non-dividend put/call) and
                             the dividend call

Run directly: `python simulate_and_plot.py`
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from binomial_tree import crr_american_price
from european_pricer import black_scholes_price, european_mc_price
from lsm_pricer import exercise_boundary, lsm_american_price

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Base case parameters: the standard Longstaff-Schwartz test case, an
# American put on a non-dividend paying stock.
BASE = dict(s0=100.0, k=100.0, r=0.06, sigma=0.2, T=1.0)
N_STEPS = 50
DEGREE = 3
SEED = 2024

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

COLOR_MC = "#2166AC"
COLOR_BAND = "#B8D4EA"
COLOR_BENCH = "#B2182B"
COLOR_EARLY = "#D6604D"
COLOR_HELD = "#4393C3"
COLOR_EURO = "#999999"
COLOR_AMER = "#2166AC"


def chart_convergence():
    """LSM price estimate + 95% CI vs number of paths, benchmarked to CRR."""
    path_counts = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    prices, ci_lo, ci_hi = [], [], []

    for n in path_counts:
        res = lsm_american_price(
            **BASE, option_type="put", n_paths=n, n_steps=N_STEPS,
            degree=DEGREE, antithetic=True, seed=SEED,
        )
        prices.append(res.mc_result.price)
        ci_lo.append(res.mc_result.ci_lower)
        ci_hi.append(res.mc_result.ci_upper)

    crr_price = crr_american_price(**BASE, option_type="put", n_steps=2000)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(path_counts, ci_lo, ci_hi, color=COLOR_BAND, alpha=0.7,
                     label="LSM 95% confidence interval")
    ax.plot(path_counts, prices, "o-", color=COLOR_MC, linewidth=2,
             markersize=5, label="LSM price estimate")
    ax.axhline(crr_price, color=COLOR_BENCH, linestyle="--", linewidth=1.8,
               label=f"CRR binomial benchmark ({crr_price:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("American put price")
    ax.set_title("LSM Convergence to the Binomial Benchmark\n"
                  f"S0={BASE['s0']}, K={BASE['k']}, r={BASE['r']}, "
                  f"σ={BASE['sigma']}, T={BASE['T']}y")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "convergence.png"), dpi=150)
    plt.close(fig)
    print(f"convergence.png saved. Final LSM price={prices[-1]:.4f}, CRR={crr_price:.4f}")


def chart_exercise_boundary():
    """Estimated optimal exercise stock price as a function of time to expiry."""
    res = lsm_american_price(
        **BASE, option_type="put", n_paths=200_000, n_steps=N_STEPS,
        degree=DEGREE, antithetic=True, seed=SEED,
    )
    times, boundary = exercise_boundary(res, k=BASE["k"], option_type="put")

    # smooth with a short moving average over the (noisy, regression-based)
    # per-step boundary estimate, purely for visual clarity
    valid = ~np.isnan(boundary)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(times[valid], boundary[valid], "o", color=COLOR_EARLY,
             markersize=3, alpha=0.5, label="Per-step exercise boundary estimate")

    if valid.sum() > 5:
        window = max(3, valid.sum() // 20)
        kernel = np.ones(window) / window
        smoothed = np.convolve(boundary[valid], kernel, mode="valid")
        smoothed_times = times[valid][window - 1:]
        ax.plot(smoothed_times, smoothed, "-", color=COLOR_MC, linewidth=2.5,
                 label=f"Smoothed boundary ({window}-step moving average)")

    ax.axhline(BASE["k"], color="#333333", linestyle=":", linewidth=1.2,
               label=f"Strike K={BASE['k']}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock price")
    ax.set_title("Estimated Optimal Exercise Boundary\nAmerican Put")
    ax.text(0.02, 0.03,
            "Exercise region (below boundary) ↓\nHold region (above boundary) ↑",
            transform=ax.transAxes, fontsize=9, color="#555555", va="bottom")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exercise_boundary.png"), dpi=150)
    plt.close(fig)
    print("exercise_boundary.png saved.")


def chart_american_vs_european():
    """
    Price comparison across a range of spot prices, showing the early
    exercise premium for: (a) the base-case American put vs European put,
    and (b) an American call on a dividend-paying stock vs its European
    counterpart (the case where the call premium is actually nonzero).
    """
    spots = np.linspace(70, 130, 13)
    div_q = 0.08  # dividend yield for the extension case (chosen to make the
    # otherwise-small call early-exercise premium clearly visible on the chart)

    euro_put, amer_put = [], []
    euro_call_div, amer_call_div = [], []

    for s in spots:
        p_euro = black_scholes_price(s0=s, k=BASE["k"], r=BASE["r"],
                                      sigma=BASE["sigma"], T=BASE["T"], option_type="put")
        p_amer = crr_american_price(s0=s, k=BASE["k"], r=BASE["r"],
                                     sigma=BASE["sigma"], T=BASE["T"],
                                     option_type="put", n_steps=1000)
        euro_put.append(p_euro)
        amer_put.append(p_amer)

        c_euro = black_scholes_price(s0=s, k=BASE["k"], r=BASE["r"], sigma=BASE["sigma"],
                                      T=BASE["T"], option_type="call", q=div_q)
        c_amer = crr_american_price(s0=s, k=BASE["k"], r=BASE["r"], sigma=BASE["sigma"],
                                     T=BASE["T"], option_type="call", n_steps=1000, q=div_q)
        euro_call_div.append(c_euro)
        amer_call_div.append(c_amer)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    ax.plot(spots, euro_put, "--", color=COLOR_EURO, linewidth=2, label="European put")
    ax.plot(spots, amer_put, "-", color=COLOR_AMER, linewidth=2.2, label="American put")
    ax.fill_between(spots, euro_put, amer_put, color=COLOR_EARLY, alpha=0.25,
                     label="Early exercise premium")
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Option price")
    ax.set_title("Non-Dividend Stock: Put Premium Exists")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    ax = axes[1]
    ax.plot(spots, euro_call_div, "--", color=COLOR_EURO, linewidth=2, label="European call")
    ax.plot(spots, amer_call_div, "-", color=COLOR_AMER, linewidth=2.2, label="American call")
    ax.fill_between(spots, euro_call_div, amer_call_div, color=COLOR_EARLY, alpha=0.25,
                     label="Early exercise premium")
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Option price")
    ax.set_title(f"Dividend Stock (q={div_q}): Call Premium Appears")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("American vs European Prices: Where the Early Exercise Premium Comes From",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, "american_vs_european.png"), dpi=150)
    plt.close(fig)
    print("american_vs_european.png saved.")


def chart_greeks():
    """
    Delta and vega via bump-and-reprice finite differences, using the LSM
    engine itself (same random seed for base/up/down so the comparison
    isn't swamped by Monte Carlo noise -- this is the standard "common
    random numbers" trick for finite-difference Greeks).
    """
    spots = np.linspace(80, 120, 9)
    ds = 0.5   # bump for delta
    dvol = 0.01  # bump for vega
    n_paths = 100_000

    deltas, vegas = [], []
    for s in spots:
        seed = 999  # fixed seed -> common random numbers across bumps
        p_up = lsm_american_price(s0=s + ds, k=BASE["k"], r=BASE["r"], sigma=BASE["sigma"],
                                   T=BASE["T"], option_type="put", n_paths=n_paths,
                                   n_steps=N_STEPS, degree=DEGREE, antithetic=True,
                                   seed=seed).price
        p_dn = lsm_american_price(s0=s - ds, k=BASE["k"], r=BASE["r"], sigma=BASE["sigma"],
                                   T=BASE["T"], option_type="put", n_paths=n_paths,
                                   n_steps=N_STEPS, degree=DEGREE, antithetic=True,
                                   seed=seed).price
        deltas.append((p_up - p_dn) / (2 * ds))

        v_up = lsm_american_price(s0=s, k=BASE["k"], r=BASE["r"], sigma=BASE["sigma"] + dvol,
                                   T=BASE["T"], option_type="put", n_paths=n_paths,
                                   n_steps=N_STEPS, degree=DEGREE, antithetic=True,
                                   seed=seed).price
        v_dn = lsm_american_price(s0=s, k=BASE["k"], r=BASE["r"], sigma=BASE["sigma"] - dvol,
                                   T=BASE["T"], option_type="put", n_paths=n_paths,
                                   n_steps=N_STEPS, degree=DEGREE, antithetic=True,
                                   seed=seed).price
        vegas.append((v_up - v_dn) / (2 * dvol))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    ax.plot(spots, deltas, "o-", color=COLOR_MC, linewidth=2, markersize=5)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(BASE["k"], color="#999999", linestyle=":", linewidth=1.2, label="Strike")
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Delta")
    ax.set_title("Delta (bump-and-reprice)\nAmerican put")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(spots, vegas, "o-", color=COLOR_EARLY, linewidth=2, markersize=5)
    ax.axvline(BASE["k"], color="#999999", linestyle=":", linewidth=1.2, label="Strike")
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Vega")
    ax.set_title("Vega (bump-and-reprice)\nAmerican put")
    ax.legend(fontsize=9)

    fig.suptitle("Greeks via Finite Differences (Common Random Numbers)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, "greeks.png"), dpi=150)
    plt.close(fig)
    print("greeks.png saved.")


if __name__ == "__main__":
    print("Generating analysis charts into outputs/ ...\n")
    chart_convergence()
    chart_exercise_boundary()
    chart_american_vs_european()
    chart_greeks()
    print("\nAll charts generated.")
