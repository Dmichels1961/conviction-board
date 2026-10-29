"""Five more signals with published evidence and free data. Each writes columns into data_web/auto.csv.

  filing_sim     cosine similarity of the latest 10-K/10-Q text with the same form a year earlier. Companies that change their filings
                 a lot underperform for months afterwards; non-changers do better (Cohen, Malloy & Nguyen 2020, "Lazy Prices").
                 SEC EDGAR, checked weekly per name, recomputed only when a new filing appears.
  put_skew       implied volatility of a ~10% out-of-the-money put minus at-the-money call volatility, ~1 month out. Expensive
                 downside protection precedes weaker returns (Xing, Zhang & Zhao 2010). Yahoo option chains, 3-day rotation.
  customer_mom   last month's return of a supplier's main customers (Cohen & Frazzini 2008: suppliers react late to customer news),
                 plus the customers' capital-spending growth from SEC filings for AI build-out suppliers. The supplier -> customer
                 map is a hand-made list of well-known public relationships (SUPPLY below), not data from a vendor.
  dispersion     (highest - lowest) analyst EPS estimate for next year / the average. More disagreement, lower later returns
                 (Diether, Malloy & Scherbina 2002). Taken from the daily Yahoo consensus snapshot, no extra requests.
  ivol           idiosyncratic volatility: the yearly volatility of daily returns left after removing the S&P 500's move. The most
                 volatile stocks have had low later returns (Ang, Hodrick, Xing & Zhang 2006). Computed from prices already pulled.
Not yet included: hedge-fund crowding from 13F filings (needs a vetted list of hedge-fund filers; planned).
"""
import json, re, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import autodata as AD

ROOT = Path(__file__).resolve().parent
HYPER = ["MSFT", "GOOGL", "AMZN", "META", "ORCL"]
SUPPLY = {  # supplier: main customers (public, well-known relationships; edit freely)
    "NVDA": HYPER, "AVGO": ["GOOGL", "META", "AAPL"], "MRVL": ["AMZN", "MSFT"], "AMD": ["MSFT", "META", "ORCL"], "ANET": ["MSFT", "META", "ORCL"],
    "CLS": ["GOOGL", "AMZN"], "VRT": HYPER, "MU": ["NVDA"] + HYPER, "WDC": HYPER, "STX": HYPER, "SNDK": HYPER, "TSM": ["NVDA", "AAPL", "AMD", "AVGO"],
    "ASML": ["TSM", "MU", "INTC"], "AMAT": ["TSM", "MU", "INTC"], "LRCX": ["TSM", "MU", "INTC"], "KLAC": ["TSM", "MU", "INTC"],
    "COHR": ["NVDA", "GOOGL"], "LITE": ["NVDA", "GOOGL"], "ALAB": ["NVDA", "AMZN"], "CRDO": ["AMZN", "MSFT"], "GLW": HYPER, "APH": ["NVDA"] + HYPER,
    "SMCI": HYPER, "DELL": HYPER, "CIEN": HYPER, "PWR": HYPER, "GEV": HYPER, "ETN": HYPER, "FIX": HYPER, "EME": HYPER, "CEG": HYPER, "VST": HYPER,
    "TSLA": [], }


# ----------------------------------------------------------------- idiosyncratic volatility (from closes)
def ivol(closes, days=252):
    if "SPY" not in closes.columns: return {}
    r = closes.iloc[-days - 1:].pct_change().iloc[1:]; m = r["SPY"]; out = {}
    for t in r.columns:
        if t == "SPY": continue
        y = r[t]; ok = y.notna() & m.notna()
        if ok.sum() < 150: continue
        b = np.polyfit(m[ok], y[ok], 1); res = y[ok] - np.polyval(b, m[ok])
        out[t] = float(res.std() * np.sqrt(252))
    return out


def resmom(closes, days=252, skip=21):
    """Residual momentum (Blitz, Huij & Martens 2011): sum of daily returns left after removing the S&P 500's move, over months 12 to 2,
    divided by their volatility. Less crash-prone than plain momentum and less tied to the market's direction."""
    if "SPY" not in closes.columns or len(closes) < days + 1: return {}
    r = closes.iloc[-days - 1:].pct_change().iloc[1:]; m = r["SPY"]; out = {}
    for t in r.columns:
        if t == "SPY": continue
        y = r[t]; ok = y.notna() & m.notna()
        if ok.sum() < 200: continue
        b = np.polyfit(m[ok], y[ok], 1); res = (y - np.polyval(b, m)).where(ok).iloc[:-skip].dropna()
        if len(res) > 150: out[t] = float(res.sum() / res.std()) if res.std() > 1e-4 else 0.0   # no stock-specific movement: no signal
    return out


# ----------------------------------------------------------------- customer momentum (from closes + SEC capex)
def customer_mom(closes, capex):
    out = {}
    if closes.empty: return out
    m1 = closes.iloc[-1] / closes.iloc[-22] - 1
    for s, cs in SUPPLY.items():
        cs = [c for c in cs if c in m1.index and pd.notna(m1[c])]
        if not cs: continue
        mom = float(m1[cs].mean())
        cg = [capex[c] for c in cs if c in capex and pd.notna(capex[c])]
        out[s] = mom + (0.25 * float(np.mean(cg)) if cg else 0.0)     # capex growth of 40% adds 10 pts, on par with a strong month
    return out


# ----------------------------------------------------------------- options skew (Yahoo)
def skew_from_chain(calls, puts, spot):
    def iv_near(df, k):
        col = lambda n: df[n].fillna(0) if n in df.columns else pd.Series(0, index=df.index)
        d = df[(df.impliedVolatility > 0.01) & ((col("openInterest") > 0) | (col("volume") > 0))]
        if d.empty: return np.nan
        row = d.iloc[(d.strike - k).abs().argsort().iloc[0]]
        return float(row.impliedVolatility) if abs(row.strike / k - 1) < 0.05 else np.nan
    return iv_near(puts, 0.9 * spot) - iv_near(calls, spot)


def refresh_skew(a, pool, today, closes, log=print, cap=220, fetch=None, sleep=0.3):
    if fetch is None:
        import yfinance as yf
        def fetch(t):
            tk = yf.Ticker(t); exps = [e for e in tk.options if (pd.Timestamp(e) - pd.Timestamp(today)).days >= 20]
            if not exps: return None
            ch = tk.option_chain(exps[0]); return ch.calls, ch.puts
    st = AD.load_state(); done = st.setdefault("skew", {}); ok = 0
    for t in AD.due(pool, done, today, 3, cap):
        try:
            if t not in closes.columns or closes[t].dropna().empty: done[t] = str(today); continue
            ch = fetch(t)
            if ch is not None:
                v = skew_from_chain(ch[0], ch[1], float(closes[t].dropna().iloc[-1]))
                if pd.notna(v): AD.setv(a, t, "put_skew", round(v, 4)); ok += 1
            done[t] = str(today)
        except Exception as e:
            log(f"skew {t}: {str(e)[:80]}")
        if sleep: time.sleep(sleep)
    AD.save_state(st); log(f"options skew (Yahoo): {ok} names refreshed")


# ----------------------------------------------------------------- filing text similarity (SEC)
WORD = re.compile(r"[a-z]{3,}")


def text_of(html):
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html); html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = re.sub(r"&[a-z#0-9]+;", " ", html)
    return Counter(WORD.findall(html.lower()))


def cosine(a, b):
    common = set(a) & set(b); num = sum(a[w] * b[w] for w in common)
    den = np.sqrt(sum(v * v for v in a.values())) * np.sqrt(sum(v * v for v in b.values()))
    return float(num / den) if den else np.nan


def filing_pair(sub):
    """Newest 10-K or 10-Q and the same form about a year earlier -> ((acc, doc), (acc, doc)) or None."""
    r = pd.DataFrame(sub.get("filings", {}).get("recent", {}))
    if r.empty: return None
    r = r[r.form.isin(["10-K", "10-Q"]) & (r.primaryDocument.astype(str) != "")].copy()
    if r.empty: return None
    r["rd"] = pd.to_datetime(r.reportDate, errors="coerce"); r = r.dropna(subset=["rd"]).sort_values("rd")
    new = r.iloc[-1]; old = r[(r.form == new.form) & ((r.rd - (new.rd - pd.DateOffset(years=1))).abs() <= pd.Timedelta(days=45))]
    if old.empty: return None
    o = old.iloc[-1]
    return (new.accessionNumber, new.primaryDocument, str(new.filingDate)), (o.accessionNumber, o.primaryDocument)


def refresh_filings(a, pool, today, log=print, cap=200, budget_s=1200):
    import edgar as EG
    try: cmap = AD.cik_map()
    except Exception as e: log(f"filing text: SEC ticker map FAILED ({e})"); return
    st = AD.load_state(); done = st.setdefault("filing_sim", {}); seen = st.setdefault("filing_acc", {}); t0 = time.time(); ok = 0
    url = lambda cik, acc, doc: f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
    for t in AD.due(pool, done, today, 7, cap):
        if time.time() - t0 > budget_s: log("filing text: time budget reached, rest continue tomorrow"); break
        cik = cmap.get(t.upper()); done[t] = str(today)
        if not cik: continue
        try:
            pair = filing_pair(json.loads(EG._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")))
            if not pair: continue
            (acc, doc, filed), (acc0, doc0) = pair
            if seen.get(t) == acc: continue                          # unchanged since last check: keep the stored value
            h1, h0 = EG._get(url(cik, acc, doc)), EG._get(url(cik, acc0, doc0))
            sim = cosine(text_of(h1), text_of(h0))
            try:                                    # emerging-technology talk in the same two filings (for the Emerging winners list)
                import emerging as EM
                n1, p1 = EM.theme_hits(EM.strip_html(h1)); n0, p0 = EM.theme_hits(EM.strip_html(h0))
                rise = {k: p1[k] - p0.get(k, 0) for k in p1}; top = max(rise, key=rise.get) if max(rise.values()) > 0 else max(p1, key=p1.get)
                AD.setv(a, t, "theme_level", round(n1, 2)); AD.setv(a, t, "theme_change", round(n1 - n0, 2)); AD.setv(a, t, "top_theme", top)
            except Exception as e: log(f"themes {t}: {str(e)[:60]}")
            if pd.notna(sim):
                AD.setv(a, t, "filing_sim", round(sim, 4)); AD.setv(a, t, "filing_date", filed); seen[t] = acc; ok += 1
            time.sleep(0.3)
        except Exception as e:
            log(f"filing text {t}: {str(e)[:80]}")
    AD.save_state(st); log(f"filing text similarity (SEC): {ok} new comparisons")


# ----------------------------------------------------------------- run all
def refresh(a, pool, today, closes, log=print):
    for name, fn in [("ivol", lambda: ivol(closes)), ("resmom", lambda: resmom(closes)), ("customer_mom", lambda: customer_mom(closes, a["capex_growth"].dropna().to_dict() if "capex_growth" in a.columns else {}))]:
        try:
            v = fn(); [AD.setv(a, t, name, round(x, 4)) for t, x in v.items() if t in a.index]; log(f"{name}: {len(v)} names")
        except Exception as e: log(f"{name} FAILED ({e})")
    for fn in (lambda: refresh_skew(a, pool, today, closes, log=log), lambda: refresh_filings(a, pool, today, log=log)):
        try: fn()
        except Exception as e: log(f"signal step FAILED ({type(e).__name__}: {e})")
        AD.save_auto(a)


def selftest():
    a, b = Counter("the risk factors remain the same".split()), Counter("the risk factors remain the same".split())
    assert abs(cosine(a, b) - 1) < 1e-9 and cosine(a, Counter({"zzz": 1})) == 0
    assert text_of("<p>Revenue&nbsp;grew</p><script>var x</script>") == Counter({"revenue": 1, "grew": 1})
    sub = {"filings": {"recent": {"form": ["10-Q", "10-K", "10-Q", "8-K"], "reportDate": ["2026-06-30", "2025-12-31", "2025-06-30", "2026-07-01"],
                                  "accessionNumber": ["a3", "a2", "a1", "a4"], "primaryDocument": ["q.htm", "k.htm", "q0.htm", "x.htm"],
                                  "filingDate": ["2026-08-01", "2026-02-01", "2025-08-01", "2026-07-02"]}}}
    assert filing_pair(sub) == (("a3", "q.htm", "2026-08-01"), ("a1", "q0.htm"))
    calls = pd.DataFrame({"strike": [90, 100, 110], "impliedVolatility": [0.4, 0.3, 0.28], "openInterest": [5, 5, 5]})
    puts = pd.DataFrame({"strike": [90, 100], "impliedVolatility": [0.45, 0.33], "openInterest": [5, 5]})
    assert abs(skew_from_chain(calls, puts, 100) - 0.15) < 1e-9
    idx = pd.bdate_range("2025-01-01", periods=300); rng = np.random.default_rng(0); m = rng.normal(0, 0.01, 300)
    cl = pd.DataFrame({"SPY": 100 * np.cumprod(1 + m), "X": 100 * np.cumprod(1 + 1.5 * m + rng.normal(0, 0.02, 300)), "MSFT": 100 * np.cumprod(1 + m)}, index=idx)
    v = ivol(cl); assert 0.25 < v["X"] < 0.4, v
    rm = resmom(cl); assert "X" in rm and rm["MSFT"] == 0.0, rm                           # 2% daily idiosyncratic noise is about 32% a year
    cm = customer_mom(cl.assign(GOOGL=cl.MSFT, AMZN=cl.MSFT, META=cl.MSFT, ORCL=cl.MSFT), {"MSFT": 0.4}); assert "VRT" in cm
    print("signals2 selftest ok")


if __name__ == "__main__":
    selftest()
