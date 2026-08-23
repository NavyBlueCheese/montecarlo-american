# Monte Carlo Pricing of an American Option (Longstaff–Schwartz)

A from-scratch implementation of the Longstaff–Schwartz Least Squares Monte
Carlo (LSM) method for pricing American-style options, plus a live
in-browser visualization and a set of analysis charts that check the
method against a known-correct benchmark.

```
monte-carlo-american-option/
  gbm_paths.py          simulate GBM stock price paths
  european_pricer.py    European MC pricer + Black-Scholes closed form
  binomial_tree.py       CRR binomial tree (American price benchmark)
  lsm_pricer.py          Longstaff-Schwartz American option pricer
  simulate_and_plot.py   generates all four analysis charts
  visualization.html     live in-browser animation (open directly, no server)
  outputs/                generated charts land here
  requirements.txt
```

## 1. Why American options need more than plain Monte Carlo

Plain Monte Carlo prices a **European** option by simulating many random
paths of the stock price out to expiry, computing the payoff at that one
fixed date on each path, discounting it back to today, and averaging.
That works because a European option can only be exercised at expiry —
there's no decision to make along the way.

An **American** option can be exercised at any time before expiry. Pricing
it means deciding, at every point on every simulated path, whether to
exercise now or keep holding. Holding is only the right choice if the
*continuation value* — the expected value of holding on — is higher than
what you'd get by exercising immediately. The catch: continuation value
depends on the future, and forward simulation alone doesn't give you that
directly. You'd need to know, from this point on every path, what happens
next — which is exactly what you don't have yet when simulating forward.

## 2. Geometric Brownian Motion (GBM)

The stock is modeled as GBM under the risk-neutral measure:

```
dS_t = (r - q) S_t dt + sigma S_t dW_t
```

`r` is the risk-free rate, `q` a continuous dividend yield (zero for the
base case), `sigma` the volatility, and `W_t` a standard Brownian motion.
GBM has a closed-form (lognormal) transition density, so paths can be
simulated exactly at any set of time steps without discretization error:

```
S_{t+dt} = S_t * exp( (r - q - 0.5 sigma^2) dt + sigma sqrt(dt) Z ),   Z ~ N(0,1)
```

`gbm_paths.py` implements exactly this, with an optional **antithetic
variates** flag: for every draw `Z` it also uses `-Z`, halving the number
of independent random draws needed and reducing the variance of the final
price estimate for payoffs that are monotonic in the path (which option
payoffs are), without introducing any bias.

## 3. Longstaff–Schwartz: regression to estimate continuation value

Longstaff and Schwartz (2001) solve the early-exercise problem with a mix
of forward simulation and *backward induction*:

1. Simulate all paths forward, as usual.
2. At maturity, each path's cash flow is just its payoff there.
3. Step backward one time slice at a time. At each step:
   - Look only at paths that are **in the money** — out-of-the-money paths
     would never exercise, so including them would just add noise.
   - Regress each in-the-money path's *realized, discounted future cash
     flow* (whatever it actually goes on to pay, given exercise decisions
     already made at later times) on a small polynomial basis in the
     current stock price: `1, S, S^2, S^3`.
   - The fitted value at each path's current price is the estimated
     continuation value.
   - Compare it to the immediate exercise payoff. Where exercising now is
     better, mark that path as exercising here: overwrite its recorded
     cash flow with today's payoff, and discard whatever it had recorded
     for later (once you exercise, nothing after matters).
4. After reaching the first time step, discount every path's final
   recorded cash flow back to `t=0` and average. That average is the LSM
   price. Its standard error and 95% confidence interval come from the
   sample the same way they would for a European Monte Carlo price.

This project uses a degree-3 polynomial basis, as suggested in the brief;
`lsm_pricer.py` is a direct, heavily-commented implementation of the
algorithm above.

## 4. The exercise boundary

A side effect of running LSM is that you learn, at each time step,
approximately which stock prices trigger exercise — the **exercise
boundary**. For an American put, it's a curve that starts below the
strike near expiry (only deep-in-the-money paths are worth exercising
early right before expiry, since time value evaporates anyway) and drops
further below the strike as time-to-expiry increases (more time value
means holding is worth more, so it takes an even lower stock price to
make exercising worthwhile). `exercise_boundary()` in `lsm_pricer.py`
estimates this by taking, at each time step, the most favorable stopping
price observed among paths that exercised exactly there.

## 5. Why the benchmark matters

LSM is an approximation — the regression only estimates continuation
value, so there's no guarantee it converges to the exact price without
checking. `binomial_tree.py` implements a Cox-Ross-Rubinstein (CRR)
binomial tree, a much older and well-understood method that handles early
exercise exactly (at every node, compare exercising now to the discounted
expected value of continuing, keep the larger one). With enough steps it
converges to the true price, so it's used here purely as ground truth to
confirm the LSM engine is working correctly — not because it's the
technique being demonstrated.

As a related sanity check: an American **call on a non-dividend stock**
should never be exercised early (there's no dividend to capture by
exercising, and money is worth more later than now, so waiting is always
at least as good). `binomial_tree.py`'s `__main__` block confirms this
numerically — the American and European call prices come out equal.

## 6. What's in `outputs/` after running the analysis

Running `python simulate_and_plot.py` regenerates four charts:

- **`convergence.png`** — the LSM price estimate and its 95% confidence
  band as the number of simulated paths grows, with the CRR binomial price
  drawn as a horizontal reference line. The band should visibly narrow and
  close in on the reference line as paths increase.
- **`exercise_boundary.png`** — the estimated optimal exercise stock price
  as a function of time to expiry, the classic Longstaff-Schwartz result
  chart described in section 4.
- **`american_vs_european.png`** — American vs European prices across a
  range of spot prices, for the base-case put (where the early exercise
  premium is real even with no dividend) and for a call on a
  dividend-paying stock (where the premium only appears once there's a
  dividend to capture — see section 5).
- **`greeks.png`** — delta and vega for the American put, computed by
  bump-and-reprice finite differences: reprice with the input nudged up
  and down and divide by twice the bump size. Uses common random numbers
  (the same seed for the up and down repricing) so the finite difference
  isn't swamped by Monte Carlo noise between the two runs.

## 7. Running the pricing engine

```bash
pip install -r requirements.txt

# quick smoke tests, each module can be run directly
python gbm_paths.py
python european_pricer.py     # checks Monte Carlo against Black-Scholes
python binomial_tree.py       # checks American call == European call (no dividend)
python lsm_pricer.py          # checks LSM against the binomial benchmark

# regenerate all four analysis charts into outputs/
python simulate_and_plot.py
```

To price something with your own parameters:

```python
from lsm_pricer import lsm_american_price
from binomial_tree import crr_american_price

result = lsm_american_price(
    s0=100, k=100, r=0.06, sigma=0.2, T=1.0,
    option_type="put", n_paths=100_000, n_steps=50,
    degree=3, antithetic=True, seed=42,
)
print(result.mc_result)   # price, standard error, 95% CI

benchmark = crr_american_price(s0=100, k=100, r=0.06, sigma=0.2, T=1.0,
                                option_type="put", n_steps=2000)
print(benchmark)
```

## 8. Using the live visualization

Open `visualization.html` directly in a browser — it's a single
self-contained file with no build step and no server required. It
reimplements the same GBM simulation and LSM backward induction in
JavaScript (verified against the Python engine's numbers) so the whole
thing runs client-side.

- Adjust spot price, strike, volatility, and the number of simulated
  paths with the sliders, and switch between an American put and an
  American call on a dividend-paying stock.
- Click **Run simulation**. Paths are simulated and priced immediately,
  then revealed on the chart in batches, left to right, as if watching
  the run happen live. The price readout in the header updates after each
  batch completes, the same way a running Monte Carlo average builds up.
- Once the exercise decision is known, paths are colored by outcome:
  blue for held to expiry, amber (with a marker dot) for exercised early.
- The dashed horizontal line is the strike. The risk-free rate (6%),
  expiry (1 year), time steps (50), and regression basis (degree 3) are
  fixed to match the Python engine, so the numbers you see here are
  directly comparable to `lsm_pricer.py`'s output for the same inputs.

## 9. Non-goals / possible extensions

This pass deliberately stays with:

- Single-asset GBM only — no local/stochastic volatility, no
  jump-diffusion, no multi-asset baskets.
- A simple polynomial basis and antithetic variates as the only variance
  reduction technique.

Control variates (e.g. using the European price, which has a closed form,
as a control) and importance sampling are natural next steps if more
precision is needed for the same simulation budget, but are left out here
to keep the implementation easy to follow.
