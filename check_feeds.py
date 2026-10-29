"""Feed diagnostics: run this once on your machine before trusting the weekly board.

    python check_feeds.py            # every ETF in etf_universe.csv + its benchmark
    python check_feeds.py CGGR SPY   # specific funds

For each fund it shows which source supplied the holdings (csv / url / fmp / yahoo_top10), how many
positions came back, the weight total, and the top names, and flags anything that looks partial or
wrong. It also tests the live-quote feed. Nothing is written to the weekly snapshots.
"""
import os
import sys

import pandas as pd

import screener as S
from data_sources import LiveProvider, holdings_sources, live_quotes

FULL = {"csv", "url", "fmp"}


def main():
    funds = sys.argv[1:]
    if not funds:
        u = S.load_universe()
        funds = sorted(set(u["etf"]) | set(u["benchmark"]))
    print("Keys configured:", {k: bool(os.environ.get(k)) for k in ("FMP_API_KEY", "POLYGON_API_KEY", "FINNHUB_API_KEY")})
    print("Issuer links in holdings_sources.csv:", sorted(holdings_sources()) or "none")
    prov = LiveProvider()
    rows = []
    for f in funds:
        h = prov.holdings(f)
        src = h.attrs.get("source", "unknown")
        top = ", ".join(f"{t} {w:.1f}" for t, w in zip(h["ticker"].head(4), h["weight"].head(4)))
        flag = ""
        if src not in FULL:
            flag = "PARTIAL: add an issuer link or FMP key"
        elif len(h) < 15:
            flag = "very short list: check the file"
        elif not 60 <= h["weight"].sum() <= 105:
            flag = f"weights sum to {h['weight'].sum():.0f}%: check the file"
        rows.append(dict(fund=f, source=src, positions=len(h), weight_sum=round(h["weight"].sum(), 1), top=top, flag=flag))
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print(pd.DataFrame(rows).to_string(index=False))
    q = live_quotes(["SPY", "AAPL"])
    print("\nLive quotes:", q or "none returned (check keys / network)")


if __name__ == "__main__":
    main()
