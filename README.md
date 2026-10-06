# Monte Carlo Pricing of an American Option (Longstaff–Schwartz)

Plain Monte Carlo prices a European option by simulating many random
paths of the stock price out to expiry, computing the payoff at that one
fixed date on each path, discounting it back to today (European option can only be exercised at expiry)

An American option can be exercised at any time before expiry. Pricing
it means that deciding, at every point on every simulated path, whether to
exercise now or keep holding. Holding is only the right choice if the
continuation value the expected value of holding on is higher than
what you'd get by exercising immediately. 

> continuation value depends on the future and forward simulation alone doesn't give you that
directly


## live visualization (in process)

[live site](https://navybluecheese.github.io/montecarlo-american/) 
