"""Offline tests for the holdings and quote feed adapters (HTTP is mocked).

    python tests/test_feeds.py

These verify the parsing and the fallback chain against payload shapes I assumed from vendor docs and
common issuer formats. They cannot prove the real services return those shapes: run check_feeds.py
with your keys/links to confirm against live endpoints.
"""
import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import data_sources as D  # noqa: E402

D.CACHE_DIR = Path(tempfile.mkdtemp())

# ---- parser: iShares-style preamble, "% of Net Assets" header, cash/non-equity rows ----
txt = ('Fund,Example\nAs of,"Oct 14, 2026"\n\n'
       'Ticker,Name,Asset Class,% of Net Assets,Shares\n'
       'NVDA,Nvidia,Equity,7.5,100\nBRK.B,Berkshire,Equity,2.5,10\nCASH,Cash,Cash,0.4,1\n')
h = D.parse_holdings_text(txt, "x", "url")
assert set(h["ticker"]) == {"NVDA", "BRK-B"} and h.attrs["source"] == "url"

# ---- FMP parser: alias handling, fractions vs percent, errors ----
h = D.parse_fmp_holdings([{"asset": "AAPL", "weightPercentage": 6.2}, {"symbol": "MSFT", "weight": 5.1}, {"name": "x"}])
assert dict(zip(h["ticker"], h["weight"])) == {"AAPL": 6.2, "MSFT": 5.1} and h.attrs["source"] == "fmp"
h = D.parse_fmp_holdings([{"symbol": "AAPL", "weight": 0.6}, {"symbol": "MSFT", "weight": 0.4}])
assert abs(h["weight"].sum() - 100) < 1e-9
for bad in ({"Error Message": "Limit Reach"}, [{"foo": 1}]):
    try:
        D.parse_fmp_holdings(bad)
        raise AssertionError("should fail")
    except ValueError as e:
        assert str(e)

# ---- URL adapter: csv, xlsx with preamble, html page ----
csv_bytes = txt.encode()
buf = io.BytesIO()
pd.DataFrame([["Fund", "Example"], ["As of", "Oct 14"], [None, None], ["Ticker", "Weight"], ["NVDA", 7.5], ["AMD", 2.5]]) \
    .to_excel(buf, header=False, index=False)
xlsx_bytes = buf.getvalue()
responses = {"https://x/h.csv": (csv_bytes, "text/csv"), "https://x/h.xlsx": (xlsx_bytes, "application/octet-stream"),
             "https://x/page": (b"<!DOCTYPE html><html>", "text/html")}
D._http_get = lambda url, params=None, headers=None, timeout=30: responses[url]
assert set(D.holdings_from_url("https://x/h.csv", "F")["ticker"]) == {"NVDA", "BRK-B"}
hx = D.holdings_from_url("https://x/h.xlsx", "F")
assert set(hx["ticker"]) == {"NVDA", "AMD"} and hx.attrs["source"] == "url"
try:
    D.holdings_from_url("https://x/page", "F")
    raise AssertionError("html must be rejected")
except ValueError as e:
    assert "HTML" in str(e)

# ---- quotes: Polygon -> Finnhub -> Yahoo fallback order ----
calls = []


def fake_get(url, params=None, headers=None, timeout=30):
    calls.append(url)
    if "polygon" in url:
        return json.dumps({"tickers": [{"ticker": "AAPL", "lastTrade": {"p": 201.5}}]}).encode(), "application/json"
    if "finnhub" in url:
        return json.dumps({"c": 500.25 if params["symbol"] == "SPY" else 0}).encode(), "application/json"
    raise AssertionError(url)


D._http_get = fake_get
D.LiveProvider.last_prices = lambda self, tk: {t: 1.0 for t in tk}      # stand-in for Yahoo
import os  # noqa: E402
os.environ["POLYGON_API_KEY"], os.environ["FINNHUB_API_KEY"] = "p", "f"
q = D.live_quotes(["AAPL", "SPY", "ZZZ"])
assert q == {"AAPL": 201.5, "SPY": 500.25, "ZZZ": 1.0}, q
for k in ("POLYGON_API_KEY", "FINNHUB_API_KEY"):
    del os.environ[k]

# ---- holdings fallback chain: csv file > issuer link > FMP > Yahoo ----
D.HOLDINGS_DIR = Path(tempfile.mkdtemp())
prov = D.LiveProvider()
D.yahoo_top_holdings = lambda f: pd.DataFrame([["AAA", 9.0]], columns=["ticker", "weight"]).pipe(
    lambda d: (d.attrs.update(source="yahoo_top10"), d)[1])
D.holdings_sources = lambda: {"CGGR": "https://x/h.csv"}
D._http_get = lambda url, params=None, headers=None, timeout=30: responses.get(url) or (
    json.dumps([{"asset": "TSM", "weightPercentage": 5.0}]).encode(), "application/json")
assert prov.holdings("CGGR").attrs["source"] == "url"
os.environ["FMP_API_KEY"] = "k"
assert prov.holdings("TCHP").attrs["source"] == "fmp"
del os.environ["FMP_API_KEY"]
assert prov.holdings("FBCG").attrs["source"] == "yahoo_top10"
(D.HOLDINGS_DIR / "DFAC.csv").write_text(txt)
assert prov.holdings("DFAC").attrs["source"] == "csv"

# source labels feed the flow logic: url/fmp/csv count as full lists, yahoo_top10 as partial
import weekly as W  # noqa: E402
assert {"csv", "url", "fmp"} == W.FULL_SOURCES and "yahoo_top10" not in W.FULL_SOURCES
print("ALL FEED TESTS PASSED")
