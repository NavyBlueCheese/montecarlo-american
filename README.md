# Monte Carlo Pricing of an American Option (Longstaff–Schwartz)

## American options actually need more than plain Monte Carlo

Plain Monte Carlo prices a European option by simulating many random
paths of the stock price out to expiry, computing the payoff at that one
fixed date on each path, discounting it back to today (European option can only be exercised at expiry)

An American option can be exercised at any time before expiry. Pricing
it means that deciding, at every point on every simulated path, whether to
exercise now or keep holding. Holding is only the right choice if the
continuation value the expected value of holding on is higher than
what you'd get by exercising immediately. 

>> continuation value depends on the future and forward simulation alone doesn't give you that
directly


## Using the live visualization (in process)

[live site](https://example.com](https://navybluecheese.github.io/montecarlo-american/) 


- Adjust spot price, strike, volatility, and the number of simulated
  paths with the sliders, and switch between an American put and an
  American call on a dividend-paying stock.
- Click Run simulation. Paths are simulated and priced
  then showed on the chart in batches from left to right. The price readout in the header updates after each
  batch completes
- Once the exercise decision is known, paths are colored by outcome:
  blue for held to expiry, and amber (with a marker dot) for exercised early
- The dashed horizontal line is the strike.
