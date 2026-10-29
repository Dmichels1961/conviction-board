"""Data access for the screener: ETF prices, ETF holdings, earnings data.

Everything that touches the network lives here so the screening logic in
screener.py can be tested offline with a fake provider.

Holdings come from, in order:
  1. holdings/<TICKER>.csv  (full holdings you download from the issuer)
  2. Yahoo Finance top-10 holdings (free, but only the top 10 positions)
Weights are always returned in PERCENT (e.g. 4.2 means 4.2%).
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HOLDINGS_DIR = Path(__file__).parent / "holdings"
CACHE_DIR = Path(__file__).parent / ".cache"


# ---------- tiny daily JSON cache ----------
def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return CACHE_DIR / f"{date.today().isoformat()}_{safe}.json"


def _cached(name: str, fn):
    p = _cache_path(name)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    val = fn()
    try:
        p.write_text(json.dumps(val))
    except Exception:
        pass
    return val


# ---------- holdings ----------
def normalize_ticker(t) -> str:
    return str(t).strip().upper().replace(".", "-").replace("/", "-")


_WEIGHT_WORDS = ("weight", "% of net", "% of fund", "% net", "% of total", "percent", "pct")


def _is_weight(col: str) -> bool:
    low = str(col).lower()
    return any(w in low for w in _WEIGHT_WORDS)


def read_holdings_csv(path: Path) -> pd.DataFrame:
    return parse_holdings_text(Path(path).read_text(encoding="utf-8-sig", errors="ignore"), Path(path).name, "csv")


def parse_holdings_text(text: str, label: str = "holdings", source: str = "csv") -> pd.DataFrame:
    """Parse an issuer holdings export into columns [ticker, weight(%)].

    Handles preamble lines (iShares / SSGA style) by locating the header row that
    contains a ticker/symbol column and a weight column ("Weight", "% of Net Assets", ...).
    """
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if ("ticker" in low or "symbol" in low) and _is_weight(low):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"{label}: could not find a header row with ticker/symbol and weight columns")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), thousands=",")
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next(cols[c] for c in cols if c in ("ticker", "symbol") or "ticker" in c or "symbol" in c)
    wcol = next(cols[c] for c in cols if _is_weight(c))
    if "asset class" in cols:
        df = df[df[cols["asset class"]].astype(str).str.lower().str.contains("equity")]
    out = pd.DataFrame({
        "ticker": df[tcol].map(normalize_ticker),
        "weight": pd.to_numeric(df[wcol].astype(str).str.replace("%", "", regex=False), errors="coerce"),
    }).dropna()
    out = out[out["ticker"].str.fullmatch(r"[A-Z0-9-]{1,8}") & (out["weight"] > 0)]
    if out["weight"].sum() <= 1.5:  # fractions -> percent
        out["weight"] *= 100
    out = out.groupby("ticker", as_index=False)["weight"].sum()
    out.attrs["source"] = source
    return out


def yahoo_top_holdings(etf: str) -> pd.DataFrame:
    import yfinance as yf

    def fetch():
        th = yf.Ticker(etf).funds_data.top_holdings
        if th is None or len(th) == 0:
            return []
        th = th.reset_index()
        sym = th.columns[0]
        wcol = next(c for c in th.columns if "percent" in c.lower())
        return [[normalize_ticker(s), float(w) * 100] for s, w in zip(th[sym], th[wcol])]

    rows = _cached(f"top_{etf}", fetch)
    out = pd.DataFrame(rows, columns=["ticker", "weight"])
    out.attrs["source"] = "yahoo_top10"
    return out


# ---------- env + http ----------
def load_dotenv(path: Path | None = None):
    """Minimal .env loader (KEY=value lines); real environment variables win."""
    p = Path(path or Path(__file__).parent / ".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()
UA = {"User-Agent": "Mozilla/5.0 (research script)"}


def _http_get(url: str, params=None, headers=None, timeout: int = 30):
    """Returns (content_bytes, content_type). Isolated so tests can replace it."""
    import requests

    r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "")


# ---------- full holdings: issuer download links ----------
def holdings_sources() -> dict:
    """holdings_sources.csv: etf,url  (paste the CSV/XLSX download link from each fund's holdings page)."""
    p = Path(__file__).parent / "holdings_sources.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p).dropna(subset=["etf", "url"])
    return {str(e).strip().upper(): str(u).strip() for e, u in zip(d["etf"], d["url"])}


def holdings_from_url(url: str, label: str) -> pd.DataFrame:
    content, ctype = _http_get(url)
    if content.lstrip()[:1] == b"<":
        raise ValueError(f"{label}: got an HTML page, not a data file. Use the fund page's CSV/XLSX download link "
                         f"(right-click the Download button, or find it in the browser dev tools Network tab)")
    if content[:2] == b"PK" or "spreadsheet" in ctype or url.lower().split("?")[0].endswith((".xlsx", ".xls")):
        err = None
        for sh in pd.read_excel(io.BytesIO(content), header=None, sheet_name=None).values():
            try:
                return parse_holdings_text(sh.to_csv(header=False, index=False), label, "url")
            except Exception as e:
                err = e
        raise ValueError(f"{label}: no sheet in the workbook looks like holdings ({err})")
    return parse_holdings_text(content.decode("utf-8-sig", errors="ignore"), label, "url")


# ---------- full holdings: Financial Modeling Prep API (needs FMP_API_KEY) ----------
_TICKER_KEYS = ("asset", "symbol", "ticker")
_WEIGHT_KEYS = ("weightPercentage", "weight", "weightPercent", "pctOfNav", "percentage")


def parse_fmp_holdings(payload) -> pd.DataFrame:
    """Defensive parser: field names vary between API versions, so several aliases are tried."""
    if isinstance(payload, dict):
        raise ValueError(f"API error: {payload.get('Error Message') or payload.get('message') or list(payload)[:5]}")
    rows = []
    for it in payload:
        t = next((it[k] for k in _TICKER_KEYS if it.get(k)), None)
        w = next((it[k] for k in _WEIGHT_KEYS if it.get(k) is not None), None)
        if t and w is not None:
            rows.append((normalize_ticker(t), w))
    if not rows:
        seen = sorted(payload[0].keys()) if payload else []
        raise ValueError(f"no usable holdings rows; fields seen: {seen}")
    df = pd.DataFrame(rows, columns=["ticker", "weight"])
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna()
    df = df[df["ticker"].str.fullmatch(r"[A-Z0-9-]{1,8}") & (df["weight"] > 0)]
    if df["weight"].sum() <= 1.5:
        df["weight"] *= 100
    df = df.groupby("ticker", as_index=False)["weight"].sum()
    df.attrs["source"] = "fmp"
    return df


def fmp_holdings(etf: str, key: str) -> pd.DataFrame:
    content, _ = _http_get("https://financialmodelingprep.com/stable/etf/holdings",
                           params={"symbol": etf, "apikey": key})
    return parse_fmp_holdings(json.loads(content))


# ---------- live quotes (dashboard) ----------
def _finnhub_quotes(tickers, key) -> dict:
    out = {}
    for t in tickers:
        try:
            c, _ = _http_get("https://finnhub.io/api/v1/quote",
                             params={"symbol": t.replace("-", "."), "token": key}, timeout=10)
            p = json.loads(c).get("c")
            if p:
                out[t] = float(p)
        except Exception:
            pass
    return out


def _polygon_quotes(tickers, key) -> dict:
    out = {}
    for i in range(0, len(tickers), 250):
        chunk = tickers[i:i + 250]
        try:
            c, _ = _http_get("https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers",
                             params={"tickers": ",".join(t.replace("-", ".") for t in chunk), "apiKey": key})
            for it in json.loads(c).get("tickers", []):
                p = ((it.get("lastTrade") or {}).get("p") or (it.get("min") or {}).get("c")
                     or (it.get("day") or {}).get("c") or (it.get("prevDay") or {}).get("c"))
                if p:
                    out[normalize_ticker(it["ticker"])] = float(p)
        except Exception:
            pass
    return out


def live_quotes(tickers) -> dict:
    """Latest price per ticker. Order: Polygon (POLYGON_API_KEY) -> Finnhub (FINNHUB_API_KEY) -> Yahoo."""
    tickers = sorted(set(tickers))
    out = {}
    if os.environ.get("POLYGON_API_KEY"):
        out.update(_polygon_quotes(tickers, os.environ["POLYGON_API_KEY"]))
    if os.environ.get("FINNHUB_API_KEY"):
        out.update(_finnhub_quotes([t for t in tickers if t not in out], os.environ["FINNHUB_API_KEY"]))
    missing = [t for t in tickers if t not in out]
    if missing:
        try:
            out.update(LiveProvider().last_prices(missing))
        except Exception:
            pass
    return out


# ---------- providers ----------
class LiveProvider:
    """Real data: holdings from CSV / issuer link / API / Yahoo, prices and earnings from Yahoo."""

    def prices(self, tickers, years: int = 3) -> pd.DataFrame:
        """Monthly adjusted closes, columns = tickers."""
        import yfinance as yf

        tickers = sorted(set(tickers))
        raw = yf.download(tickers, period=f"{years}y", interval="1mo",
                          auto_adjust=True, progress=False, group_by="column")
        if raw.empty:
            return pd.DataFrame()
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        return close.dropna(how="all")

    def holdings(self, fund: str) -> pd.DataFrame:
        """Full holdings if at all possible. Order: holdings/<F>.csv -> issuer link in
        holdings_sources.csv -> FMP API (FMP_API_KEY) -> Yahoo top-10 (partial)."""
        p = HOLDINGS_DIR / f"{fund}.csv"
        if p.exists():
            return read_holdings_csv(p)

        def cached_df(tag, fn):
            def fetch():
                df = fn()
                return dict(source=df.attrs.get("source", tag), rows=df[["ticker", "weight"]].values.tolist())
            c = _cached(f"hold_{tag}_{fund}", fetch)
            out = pd.DataFrame(c["rows"], columns=["ticker", "weight"])
            out.attrs["source"] = c["source"]
            return out

        url = holdings_sources().get(fund.upper())
        if url:
            try:
                return cached_df("url", lambda: holdings_from_url(url, fund))
            except Exception as e:
                print(f"  ! {fund}: issuer link failed ({e}); trying next source")
        key = os.environ.get("FMP_API_KEY")
        if key:
            try:
                return cached_df("fmp", lambda: fmp_holdings(fund, key))
            except Exception as e:
                print(f"  ! {fund}: FMP API failed ({e}); falling back to Yahoo top 10")
        try:
            return yahoo_top_holdings(fund)
        except Exception as e:  # network / schema problems -> empty, caller warns
            print(f"  ! could not get holdings for {fund}: {e}")
            out = pd.DataFrame(columns=["ticker", "weight"])
            out.attrs["source"] = "none"
            return out

    def earnings(self, ticker: str) -> dict:
        """Earnings beat history + forward growth estimates for one stock."""
        import yfinance as yf

        def fetch():
            time.sleep(0.15)  # be polite to Yahoo
            t = yf.Ticker(ticker)
            res = {"fwd_growth": None, "cur_growth": None, "surprises": [], "beta": None,
                   "rev_30d": None, "gross_margin": None, "roe": None, "fcf_margin": None,
                   "fcf_yield": None, "fwd_pe": None, "short_pct": None, "name": None, "sector": None}
            try:
                eh = t.earnings_history
                if eh is not None and len(eh):
                    for est, act in zip(eh["epsEstimate"], eh["epsActual"]):
                        if pd.notna(est) and pd.notna(act) and est != 0:
                            res["surprises"].append(float((act - est) / abs(est)))
            except Exception:
                pass
            try:
                ee = t.earnings_estimate
                if ee is not None and len(ee):
                    if "+1y" in ee.index and pd.notna(ee.loc["+1y", "growth"]):
                        res["fwd_growth"] = float(ee.loc["+1y", "growth"])
                    if "0y" in ee.index and pd.notna(ee.loc["0y", "growth"]):
                        res["cur_growth"] = float(ee.loc["0y", "growth"])
            except Exception:
                pass
            try:  # estimate revisions: change in consensus EPS over the last 30 days
                tr = t.eps_trend
                revs = []
                for per in ("0y", "+1y"):
                    if per in tr.index:
                        cur, old = tr.loc[per, "current"], tr.loc[per, "30daysAgo"]
                        if pd.notna(cur) and pd.notna(old) and old != 0:
                            revs.append(float((cur - old) / abs(old)))
                if revs:
                    res["rev_30d"] = float(np.mean(revs))
            except Exception:
                pass
            try:  # quality / valuation / crowding fields
                info = t.info

                def g(k):
                    v = info.get(k)
                    return float(v) if isinstance(v, (int, float)) and v == v else None

                res["beta"], res["fwd_pe"] = g("beta"), g("forwardPE")
                res["gross_margin"], res["roe"], res["short_pct"] = g("grossMargins"), g("returnOnEquity"), g("shortPercentOfFloat")
                fcf, rev, mc = g("freeCashflow"), g("totalRevenue"), g("marketCap")
                if fcf is not None and rev:
                    res["fcf_margin"] = fcf / rev
                if fcf is not None and mc:
                    res["fcf_yield"] = fcf / mc
                res["name"] = info.get("longName") or info.get("shortName")
                res["sector"] = info.get("sector")
            except Exception:
                pass
            return res

        return _cached(f"earn_{ticker}", fetch)

    def last_prices(self, tickers) -> dict:
        """Latest close for each ticker (batched)."""
        import yfinance as yf

        tickers = sorted(set(tickers))
        out = {}
        for i in range(0, len(tickers), 100):
            chunk = tickers[i:i + 100]
            raw = yf.download(chunk, period="7d", interval="1d", auto_adjust=True,
                              progress=False, group_by="column")
            if raw.empty:
                continue
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(chunk[0])
            for c in close.columns:
                s = close[c].dropna()
                if len(s):
                    out[c] = float(s.iloc[-1])
        return out

    def stock_history(self, tickers) -> pd.DataFrame:
        """Two years of monthly closes (for 12-1 momentum)."""
        import yfinance as yf

        tickers = sorted(set(tickers))
        raw = yf.download(tickers, period="2y", interval="1mo", auto_adjust=True,
                          progress=False, group_by="column")
        if raw.empty:
            return pd.DataFrame()
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        return close.dropna(how="all")
