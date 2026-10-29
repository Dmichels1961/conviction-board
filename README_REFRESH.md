# Running the conviction board automatically (free)

1. Create a GitHub repository and upload this folder (do NOT upload `.env`).
2. Repo Settings, Secrets and variables, Actions: add secrets
   - `ALPHAVANTAGE_API_KEY`  your free Alpha Vantage key (earnings calendar + surprise history, ~22 names/day)
   - `SEC_USER_AGENT`        e.g. `Your Name you@example.com` (SEC rejects requests without a contact)
3. Actions tab, run "refresh-conviction-board" once by hand and read the log. Every source that fails is logged and skipped.
4. After that it runs itself: weekdays 23:00 UTC (daily re-rank, `data/daily/<date>.csv`) and Mondays 14:30 UTC
   (freezes `data/weeks/<Monday>/`, which is what the manager-flow signal compares against).

Never run against live sites from the build sandbox, so the first run is the real test. Known limits:
- N-PORT (mutual funds, many ETFs): free and complete, but public 1-4 months after the report date. Flow from these is slow.
- Daily-disclosing ETFs should be added to `fund_registry.csv` with `format=csv` or `html` and the issuer's holdings URL.
- Estimate revisions have no free daily source: save a daily consensus snapshot and measure changes yourself (not built yet).
- Fund lists with track records: `data_web/funds/candidates_all.csv` (86 funds, many numbers scraped and unverified).
Add a fund = add a row to `fund_registry.csv` (etf, manager, benchmark, style, format, url).

## Added Oct 22
- **Backtest** (`backtest.py`, run by `python refresh.py --history` or the workflow's "history" checkbox): pulls up to 16 past N-PORT filings per fund plus Stooq prices, then tests whether top-25 by overlap / flow / both beat the pool of all held stocks over 1, 3, 6 months. Uses each filing's public date, so no look-ahead. Self-test proves it finds an edge when one exists and none when it doesn't. Read the printed bias list: funds were chosen for recent performance, so early periods are flattered. Results land in `data/backtest_summary.csv`.
- **Estimate revisions** (`revisions.py`): saves consensus EPS daily to `data/consensus/snapshots.csv`. Until 25 days of our own history exist it uses Yahoo's "30 days ago" figure; after that it uses our own. Revision breadth (up vs down) is stored but not scored.
- **Insider buying** (`form4.py`): distinct insiders with open-market purchases in 90 days; weight 3; reweighted away until data exists.
- **Funds**: registry now has 39 funds from 17 managers (N-PORT works for any US fund ticker). The engine's own track-record test decides which count as skilled, so weak funds are down-weighted automatically. Track records in `data_web/funds/candidates_all.csv` are scraped and unverified.
- Nothing network-facing has run yet. First workflow run is the real test; start with the history checkbox.
