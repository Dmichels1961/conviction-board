# Live test, pre-registered 29 October 2026

Two boards are frozen today and tested live for 12 months, to 29 October 2027. The rules below are written before any live result
exists, so the result cannot be fitted after the fact. Changing a rule restarts the clock.

## What is frozen

| | Main board | Risk-controlled board |
|---|---|---|
| Size | 25 names | 40 names |
| Enter / leave | top 25 / below 35 | top 40 / below 55 |
| Weights | earnings 60, overlap 20, flow 5, momentum 5, fundamentals and market signals 10 | same signals and weights, but each signal ranked within its sector, residual momentum, active-weight best ideas, low volatility 5% |
| Earnings surprise | last 4 quarters, equal weight | same |
| Weighting | equal | equal |

Code version: the commit that adds this file. Signal definitions are in `weekly.py` (`SIGNALS`, `score_signals`, `risk_controlled`).

## What may change without restarting the clock

- Data plumbing that does not change a definition: adding funds whose daily holdings files are found, fixing downloads, retries.
- Flow's weight, only by the rule already written: 5 to 10 points if its own top 25 beats the pool on at least two thirds of weekly
  boards over 3+ months; to 0 if it trails over 6 months.

Anything else (weights, new signals, construction) is a new model and starts its own 12-month test alongside.

## Primary measure (decided now)

**Beta-adjusted excess return of the risk-controlled board versus the pool**: each month, the board's return minus each stock's beta
times the S&P 500's return, against the same for every scored stock. Success = a t-statistic of 2 or more over the 12 months, or an
information ratio above 0.5 with the backtest's sign.

Secondary: the same for the main board; raw return versus the pool, the S&P 500 and QQQ; rank correlation of the score with next
month's beta-adjusted return.

## What the backtest said (for comparison, not proof)

November 2021 to September 2026, 59 months, point in time:

- Main board: +0.90%/month beta-adjusted vs the pool, t 1.45, information ratio 0.65, average beta 1.42.
- Risk-controlled board: +0.86%/month beta-adjusted, t 2.46, information ratio 1.11, average beta 1.11; t 1.3 in the first 60% of
  months and 2.4 in the last 40%.

About 25 model variants were tried on this data before freezing, so these numbers are optimistic. The live year is the real test.

## Known limits

Survivorship (prices exist only for stocks still listed; funds were chosen in 2026), today's sector labels applied to the past,
no trading costs, estimate revisions and guidance have no history and are untested.
