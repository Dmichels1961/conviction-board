"""Daily / weekly refresh. Runs anywhere with internet (GitHub Actions, cron, your laptop). Free sources only.

  python refresh.py             daily: holdings -> prices -> earnings calendar/surprises -> re-rank -> data/daily/<date>.csv
  python refresh.py --weekly    same, then freezes the board into data/weeks/<Monday>/ (this drives the manager-flow signal)
  python refresh.py --selftest  offline checks of the parsers and diff logic (no network)

Env: ALPHAVANTAGE_API_KEY (free key, 25 calls/day: used for the earnings calendar (1 call) and surprise history (rotating, ~20 names/day)).
Every network step is wrapped: a failed source is logged and skipped, never fatal, and the previous value stays in place.
"""
import csv, io, json, os, re, subprocess, sys, time, urllib.request
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (research script; contact: owner)"}
LOG = []


def log(*a):
    m = " ".join(str(x) for x in a); LOG.append(m); print(m, flush=True)


def get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def norm_ticker(t):
    t = str(t).strip().upper().replace(" ", "-").replace(".", "-")
    return {"GOOG": "GOOGL"}.get(t, t)


# ---------------------------------------------------------------- holdings
def parse_holdings_csv(text):
    """iShares-style CSV with preamble lines: find the header row containing Ticker and a Weight column."""
    rows = list(csv.reader(io.StringIO(text)))
    for i, r in enumerate(rows):
        low = [c.strip().lower() for c in r]
        if "ticker" in low and any(c.startswith("weight") or c.startswith("% of") for c in low):
            hdr = low; ti = hdr.index("ticker"); wi = next(j for j, c in enumerate(hdr) if c.startswith("weight") or c.startswith("% of"))
            ni = hdr.index("name") if "name" in hdr else None
            out = []
            for r2 in rows[i + 1:]:
                if len(r2) <= max(ti, wi): continue
                try: w = float(r2[wi].replace("%", "").replace(",", ""))
                except ValueError: continue
                out.append((norm_ticker(r2[ti]), w, r2[ni] if ni is not None else ""))
            return clean(pd.DataFrame(out, columns=["ticker", "weight", "name"]))
    raise ValueError("no Ticker/Weight header found")


def parse_holdings_html(text):
    """Any page with a holdings table: pick the table that has a ticker-like and a weight-like column."""
    for t in pd.read_html(io.StringIO(text)):
        cols = {c: str(c).lower() for c in t.columns}
        tc = next((c for c, l in cols.items() if "ticker" in l or "symbol" in l), None)
        wc = next((c for c, l in cols.items() if "weight" in l or "% of" in l or "%" == l.strip()), None)
        if tc is not None and wc is not None:
            d = pd.DataFrame({"ticker": t[tc].map(norm_ticker),
                              "weight": pd.to_numeric(t[wc].astype(str).str.replace("%", "").str.replace(",", ""), errors="coerce"),
                              "name": ""})
            return clean(d.dropna(subset=["weight"]))
    raise ValueError("no holdings table found")


def clean(d):
    d = d[(d.weight > 0) & d.ticker.str.match(r"^[A-Z][A-Z0-9\-]{0,5}$") & (d.ticker != "NAN")]
    d = d[~d.ticker.isin({"USD", "XTSLA", "JPY", "KRW", "TWD", "AUD", "EUR", "GBP", "CAD"})]
    return d.groupby("ticker", as_index=False).agg(weight=("weight", "sum"), name=("name", "first"))


def refresh_holdings(today):
    reg = pd.read_csv(ROOT / "fund_registry.csv")
    h = pd.read_csv(ROOT / "data_web/holdings.csv")
    ok = []
    for _, f in reg.iterrows():
        try:
            asof, src = str(today), f.url
            if f.format == "nport":      # SEC EDGAR N-PORT (free, full list, 1-4 months old)
                import edgar as EG
                d, asof, _, src = EG.fetch_fund(f.url if isinstance(f.url, str) and f.url else f.etf)
                asof = asof or str(today)
                if (today - pd.Timestamp(asof).date()).days > 150: raise ValueError(f"newest public N-PORT is {asof}, too old")
            else:
                txt = get(f.url)
                d = parse_holdings_csv(txt) if f.format == "csv" else parse_holdings_html(txt)
            if len(d) < 5 or d.weight.sum() < 30: raise ValueError(f"only {len(d)} rows / {d.weight.sum():.0f}% of weight")
            hist = ROOT / "data/holdings_history" / f.etf; hist.mkdir(parents=True, exist_ok=True)
            d.to_csv(hist / f"{asof}.csv", index=False)
            h = h[h.etf != f.etf]
            h = pd.concat([h, d.assign(etf=f.etf, source=src, asof=str(asof), fresh=True)[["etf", "ticker", "weight", "source", "asof", "fresh"]]])
            ok.append(f.etf); log(f"holdings {f.etf}: {len(d)} rows, {d.weight.sum():.1f}% of weight, as of {asof}")
        except Exception as e:
            log(f"holdings {f.etf}: FAILED ({e}); keeping previous list")
    h.to_csv(ROOT / "data_web/holdings.csv", index=False)
    # universe = existing funds + any registry fund
    u = pd.read_csv(ROOT / "etf_universe.csv")
    add = reg[~reg.etf.isin(u.etf)][["etf", "manager", "benchmark", "style"]]
    if len(add): pd.concat([u, add]).to_csv(ROOT / "etf_universe.csv", index=False)
    return ok


# ---------------------------------------------------------------- prices (stooq, free CSV, no key)
def price_stats(csv_text):
    d = pd.read_csv(io.StringIO(csv_text), parse_dates=["Date"]).sort_values("Date")
    c = d.set_index("Date")["Close"]
    if len(c) < 260: return None
    last = c.iloc[-1]; m1 = c[c.index <= c.index[-1] - pd.Timedelta(days=30)].iloc[-1]; y1 = c[c.index <= c.index[-1] - pd.Timedelta(days=365)].iloc[-1]
    hi = c.iloc[-252:].max()
    return dict(r52w=last / y1 - 1, r1m=last / m1 - 1, mom_12_1=(last / y1) / (last / m1) - 1, off_high=last / hi - 1)


def refresh_prices(tickers):
    """12-1 month momentum and distance from the 52-week high. Yahoo via yfinance (Stooq now requires a browser check)."""
    base = pd.read_csv(ROOT / "data_web/earnings_base.csv").set_index("ticker"); n = 0
    try:
        import yfinance as yf
        for k in range(0, len(tickers), 100):
            b = list(tickers[k:k + 100])
            d = yf.download(b, period="14mo", interval="1d", auto_adjust=True, progress=False, threads=True)["Close"]
            if hasattr(d, "to_frame") and getattr(d, "ndim", 2) == 1: d = d.to_frame(b[0])
            for t in b:
                if t not in d.columns or t not in base.index: continue
                c = d[t].dropna()
                if len(c) < 240: continue
                st = price_stats("Date,Close\n" + "\n".join(f"{i.date()},{v}" for i, v in c.items()))
                if st:
                    base.loc[t, "mom_12_1"] = round(st["mom_12_1"], 4)
                    if "off_high" in base.columns: base.loc[t, "off_high"] = round(st["off_high"], 4)
                    n += 1
    except Exception as e:
        log(f"prices FAILED ({e})")
    base.reset_index().to_csv(ROOT / "data_web/earnings_base.csv", index=False); log(f"prices: updated {n} names")


# ---------------------------------------------------------------- estimate revisions (own daily consensus snapshots)
def refresh_revisions(pool, today):
    try:
        import revisions as RV
        snaps = RV.snapshot(pool, today, log=log)
        r = RV.rev_30d(snaps, today)
        base = pd.read_csv(ROOT / "data_web/earnings_base.csv").set_index("ticker")
        for c in ("rev_breadth", "rev_source"):
            base[c] = base[c].astype(object) if c in base.columns else np.nan
        for t, row in r.iterrows():
            if t in base.index:
                base.loc[t, "rev_30d"] = round(row.rev_30d, 4); base.loc[t, "rev_breadth"] = row.rev_breadth; base.loc[t, "rev_source"] = row.rev_source
        base.reset_index().to_csv(ROOT / "data_web/earnings_base.csv", index=False)
        log(f"revisions: {len(r)} names ({(r.rev_source == 'own').sum()} from our own history)")
    except Exception as e:
        log(f"revisions FAILED ({e}); keeping previous values")


# ---------------------------------------------------------------- earnings (Alpha Vantage, free key)
AV = "https://www.alphavantage.co/query"


def av_json(params, key):
    j = json.loads(get(AV + "?" + "&".join(f"{k}={v}" for k, v in params.items()) + f"&apikey={key}"))
    if any(k in j for k in ("Note", "Information")) and "quarterlyEarnings" not in j: raise RuntimeError(list(j.values())[0][:120])
    return j


def surprises_from_av(j, n=4):
    q = [x for x in j.get("quarterlyEarnings", []) if x.get("surprisePercent") not in (None, "None", "")][:n]
    return [round(float(x["surprisePercent"]) / 100, 4) for x in reversed(q)]  # oldest -> newest


def refresh_earnings(pool, key, today, per_day=22):
    if not key: log("earnings: no ALPHAVANTAGE_API_KEY, skipped"); return
    base = pd.read_csv(ROOT / "data_web/earnings_base.csv").set_index("ticker")
    try:  # one call: every expected report date for the next 3 months
        cal = pd.read_csv(io.StringIO(get(f"{AV}?function=EARNINGS_CALENDAR&horizon=3month&apikey={key}")))
        cal = cal.sort_values("reportDate").drop_duplicates("symbol")
        for t, d in zip(cal.symbol, cal.reportDate):
            t = norm_ticker(t)
            if t in base.index: base.loc[t, "next_earnings"] = d
        log(f"earnings calendar: {len(cal)} rows")
    except Exception as e:
        log(f"earnings calendar FAILED ({e})")
    state = ROOT / "data/av_state.json"; seen = json.loads(state.read_text()) if state.exists() else {}
    todo = sorted([t for t in pool if t in base.index], key=lambda t: seen.get(t, ""))[:per_day]
    for t in todo:
        try:
            s = surprises_from_av(av_json({"function": "EARNINGS", "symbol": t}, key))
            if len(s) >= 2:
                s = ([np.nan] * 4 + s)[-4:]
                for i, v in enumerate(s): base.loc[t, f"s{i+1}"] = v
            seen[t] = str(today)
        except Exception as e:
            log(f"earnings {t}: {e}")
            if "rate" in str(e).lower() or "25" in str(e): break
        time.sleep(2.0)
    state.write_text(json.dumps(seen)); base.reset_index().to_csv(ROOT / "data_web/earnings_base.csv", index=False)
    log(f"earnings surprises refreshed for {len(todo)} names")


# ---------------------------------------------------------------- rebuild + snapshot
def rebuild(today, weekly):
    import tempfile
    sys.path.insert(0, str(ROOT))
    subprocess.run([sys.executable, str(ROOT / "data_web/build_live.py")], cwd=ROOT, check=False, capture_output=True)
    import weekly as W, diagnostics as D
    from web_snapshot import WebSnapshotProvider
    prov, cfg = WebSnapshotProvider(), W.Config()
    ddir = ROOT / "data" if weekly else Path(tempfile.mkdtemp())
    res = W.generate(prov, pd.read_csv(ROOT / "etf_universe.csv"), cfg, W.load_themes(), today, force=weekly, ddir=ddir, log=lambda *a: None)
    rk = res["all"].copy()
    rk["data_coverage"] = D.coverage(rk, cfg, res["flow_available"]).round(0)
    rk["days_to_earnings"] = D.event_risk(rk, prov.next_earnings(), today)
    rk["bucket"] = rk["ticker"].map(D.BUCKET).fillna("Other")
    st = D.rank_stability(rk)
    rk = rk.merge(st[["median_rank", "rank_p10", "rank_p90", "p_top10"]], left_on="ticker", right_index=True)
    out = ROOT / "data/daily"; out.mkdir(parents=True, exist_ok=True)
    rk.to_csv(out / f"{today}.csv", index=False); rk.to_csv(ROOT / "data_web" / "ranking_latest.csv", index=False)
    log(f"ranked {len(rk)} names; top 10: {', '.join(rk.sort_values('rank').ticker.head(10))}")
    return rk


def probe():
    """Record what each free source answers from the runner (status, size, first bytes) in data/probe.txt, so failures are visible."""
    import edgar as EG, urllib.error
    out = []
    tests = [("sec browse by ticker", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=CGGR&type=NPORT-P&dateb=&owner=include&count=5&output=atom", True),
             ("sec browse by ticker (MF)", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=FCNTX&type=NPORT-P&dateb=&owner=include&count=5&output=atom", True),
             ("sec mf ticker map", "https://www.sec.gov/files/company_tickers_mf.json", True),
             ("sec submissions", "https://data.sec.gov/submissions/CIK0000024238.json", True),
             ("stooq", "https://stooq.com/q/d/l/?s=aapl.us&i=d", False), ("openfigi", "https://api.openfigi.com/v3/mapping", False),
             ("ishares csv", "https://www.ishares.com/us/products/339081/ishares-a-i-innovation-and-tech-active-etf/latest-holdings.csv", False)]
    for name, url, sec in tests:
        try:
            r = EG._get(url) if sec else get(url)
            out.append(f"{name}: OK {len(r)} bytes :: {r[:160]!r}")
        except urllib.error.HTTPError as e: out.append(f"{name}: HTTP {e.code} {e.reason}")
        except Exception as e: out.append(f"{name}: {type(e).__name__} {e}")
    try:
        ids = EG.mf_ids("CGGR"); out.append(f"CGGR ids: {ids}")
        ent, used = EG.nport_entries("CGGR", 3); out.append(f"CGGR nport via {used!r}: {len(ent)} entries")
    except Exception as e: out.append(f"CGGR resolve: {type(e).__name__} {e}")
    out.append(f"SEC_UA set: {bool(os.environ.get('SEC_USER_AGENT'))}")
    (ROOT / "data/probe.txt").write_text("\n".join(out)); log("\n".join(out))


def fetch_earnings_events(tickers, log=log, limit=16, cap=600):
    """Past report dates with estimate, actual and surprise % from Yahoo, saved to data/earnings_events.csv (kept, never overwritten)."""
    import yfinance as yf
    p = ROOT / "data/earnings_events.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["ticker", "date", "eps_est", "eps_act", "surprise_pct"])
    done = set(old.ticker); rows = []
    for t in [x for x in tickers if x not in done][:cap]:
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=limit)
            if e is None or not len(e): continue
            e = e.dropna(subset=["Reported EPS"]).reset_index()
            dcol = e.columns[0]
            rows += [dict(ticker=t, date=str(pd.Timestamp(d).tz_localize(None)) if pd.Timestamp(d).tzinfo is None else str(pd.Timestamp(d).tz_convert(None)),
                          eps_est=a, eps_act=b, surprise_pct=s) for d, a, b, s in zip(e[dcol], e["EPS Estimate"], e["Reported EPS"], e["Surprise(%)"])]
        except Exception as ex:
            log(f"events {t}: {str(ex)[:80]}")
        time.sleep(0.5)
    if rows: old = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    old.to_csv(p, index=False); log(f"earnings events: {len(old)} rows for {old.ticker.nunique()} names")


def history_mode(today):
    probe()
    """One-off (re-runnable): pull every fund's past N-PORT filings and daily prices for the backtest, then run it."""
    import edgar as EG, subprocess
    reg = pd.read_csv(ROOT / "fund_registry.csv")
    for _, f in reg.iterrows():
        try: EG.history(f.url if f.format == "nport" and isinstance(f.url, str) and f.url else f.etf, n=16, out_dir=ROOT / "data/holdings_history" / f.etf, log=log)
        except Exception as e: log(f"history {f.etf}: {e}")
    names = set()
    for p in (ROOT / "data/holdings_history").glob("*/*.csv"):
        if p.name != "filed.csv": names |= set(pd.read_csv(p).ticker.astype(str))
    pdir = ROOT / "data/prices"; pdir.mkdir(parents=True, exist_ok=True)
    names.add("SPY")
    todo = [t for t in sorted(names) if not ((pdir / f"{t}.csv").exists() and time.time() - (pdir / f"{t}.csv").stat().st_mtime < 7 * 86400)]
    try:    # Yahoo via yfinance (Stooq times out from GitHub's servers)
        import yfinance as yf
        for k in range(0, len(todo), 100):
            b = todo[k:k + 100]
            d = yf.download(b, period="6y", interval="1d", auto_adjust=True, progress=False, threads=True)["Close"]
            if len(b) == 1: d = d.to_frame(b[0]) if hasattr(d, "to_frame") else d
            for t in b:
                if t in d.columns and d[t].notna().sum() > 100:
                    d[[t]].dropna().rename(columns={t: "Close"}).rename_axis("Date").reset_index().to_csv(pdir / f"{t}.csv", index=False)
        log(f"prices: {len(list(pdir.glob('*.csv')))} files")
    except Exception as e: log(f"prices FAILED ({e})")
    subprocess.run([sys.executable, str(ROOT / "backtest.py")], cwd=ROOT)
    try:
        pool = pd.read_csv(ROOT / "data_web/holdings.csv").groupby("ticker").etf.nunique().loc[lambda s: s >= 2].index.tolist()
        fetch_earnings_events(pool)
    except Exception as e: log(f"earnings events FAILED ({e})")
    subprocess.run([sys.executable, str(ROOT / "factor_backtest.py")], cwd=ROOT)
    try:
        import build_site; build_site.main()
    except Exception as e: log(f"site data FAILED ({e})")
    (ROOT / "data/history_run.log").write_text("\n".join(LOG))


def main():
    today = date.today(); weekly = "--weekly" in sys.argv
    if "--history" in sys.argv: return history_mode(today)
    ok = refresh_holdings(today)
    pool = pd.read_csv(ROOT / "data_web/holdings.csv").groupby("ticker").etf.nunique().loc[lambda s: s >= 2].index.tolist()
    refresh_prices(pool); refresh_revisions(pool, today); refresh_earnings(pool, os.environ.get("ALPHAVANTAGE_API_KEY", ""), today)
    rebuild(today, weekly)
    try:
        import build_site; build_site.main()
    except Exception as e:
        log(f"site data FAILED ({e})")
    (ROOT / "data/daily" / f"{today}.log").write_text("\n".join(LOG))


def selftest():
    import edgar, revisions, backtest, form4; edgar.selftest(); revisions.selftest(); backtest.selftest(); form4.selftest(); import factor_backtest; factor_backtest.selftest()
    s = 'iShares\nFund Holdings as of,"Oct 21, 2026"\nTicker,Name,Sector,Asset Class,Market Value,Weight (%),Notional\nMU,MICRON,IT,Equity,1,6.05,1\nNVDA,NVIDIA,IT,Equity,1,5.66,1\nUSD,USD CASH,Cash,Cash,1,-0.03,1\nBRK.B,BERKSHIRE,F,Equity,1,2.0,1\nGOOG,ALPHABET,C,Equity,1,1.0,1\n'
    d = parse_holdings_csv(s); assert list(d.ticker) == ["BRK-B", "GOOGL", "MU", "NVDA"], d
    h = "<table><tr><th>Ticker</th><th>Weight %</th></tr><tr><td>NVDA</td><td>9.97%</td></tr><tr><td>AMD</td><td>5.83%</td></tr></table>"
    assert parse_holdings_html(h).weight.sum() > 15
    assert surprises_from_av({"quarterlyEarnings": [{"surprisePercent": "5"}, {"surprisePercent": "None"}, {"surprisePercent": "-2"}]}) == [-0.02, 0.05]
    px = "Date,Open,High,Low,Close,Volume\n" + "\n".join(f"{d.date()},1,1,1,{100+i*0.1},1" for i, d in enumerate(pd.date_range("2025-01-01", periods=400)))
    st = price_stats(px); assert st and abs(st["off_high"]) < 1e-9 and st["mom_12_1"] > 0
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
