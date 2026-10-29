"""Daily / weekly refresh. Runs anywhere with internet (GitHub Actions, cron, your laptop). Free sources only.

  python refresh.py             daily: holdings -> prices -> earnings calendar/surprises -> re-rank -> data/daily/<date>.csv
  python refresh.py --weekly    same, then freezes the board into data/weeks/<Monday>/ (this drives the manager-flow signal)
  python refresh.py --selftest  offline checks of the parsers and diff logic (no network)

Env: SEC_USER_AGENT. Prices, consensus revisions and earnings (surprises, next report date) come from Yahoo via yfinance: free, no key.
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


def get_bytes(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def csv_asof(text):
    """As-of date of an issuer CSV: a 'date' column (ARK) or an 'as of' line in the preamble (iShares)."""
    m = re.search(r"^(\d{1,2})/(\d{1,2})/(\d{4}),", text, re.M) or re.search(r"as of[^0-9A-Za-z]*\"?([A-Za-z]{3}) (\d{1,2}), (\d{4})", text, re.I)
    if not m: return None
    try:
        g = m.groups()
        return str(pd.Timestamp(f"{g[2]}-{g[0]}-{g[1]}").date()) if g[0].isdigit() else str(pd.Timestamp(f"{g[0]} {g[1]} {g[2]}").date())
    except Exception: return None


def parse_holdings_xlsx(data):
    """Issuer Excel file (Capital Group style): the sheet whose header row has Ticker and a weight/percent-of-net-assets column.
    Weights given as fractions (0.0571) are converted to percent. Returns (holdings, as-of date or None)."""
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)
    for _, df in sheets.items():
        asof = None
        for _, row in df.head(15).iterrows():
            for v in row.dropna():
                m = re.search(r"as of\s*(\d{1,2}/\d{1,2}/\d{4})", str(v), re.I)
                if m: asof = str(pd.Timestamp(m.group(1)).date())
        for i in range(min(30, len(df))):
            low = [str(c).strip().lower() for c in df.iloc[i].tolist()]
            if "ticker" in low:
                wi = next((j for j, c in enumerate(low) if c.startswith("weight") or "percent of net assets" in c or c.startswith("% of")), None)
                if wi is None: continue
                ti = low.index("ticker"); ni = next((j for j, c in enumerate(low) if "name" in c), None)
                body = df.iloc[i + 1:]
                w = pd.to_numeric(body.iloc[:, wi].astype(str).str.replace("%", "").str.replace(",", ""), errors="coerce")
                d = pd.DataFrame({"ticker": body.iloc[:, ti].astype(str).map(norm_ticker), "weight": w,
                                  "name": body.iloc[:, ni].astype(str) if ni is not None else ""}).dropna(subset=["weight"])
                if d.weight.sum() <= 1.5: d["weight"] = d.weight * 100
                return clean(d), asof
    raise ValueError("no Ticker/Weight header found in any sheet")


def fetch_daily(f, today):
    """One fund's daily issuer file -> (holdings, as-of date). Raises if it can't be read or is older than 7 days."""
    if f.format == "xlsx":
        d, asof = parse_holdings_xlsx(get_bytes(f.url))
    else:
        txt = get(f.url)
        d = parse_holdings_csv(txt) if f.format == "csv" else parse_holdings_html(txt)
        asof = csv_asof(txt) if f.format == "csv" else None
    asof = asof or str(today)
    if (today - pd.Timestamp(asof).date()).days > 7: raise ValueError(f"daily file is dated {asof}, more than a week old")
    return d, asof


def refresh_holdings(today):
    reg = pd.read_csv(ROOT / "fund_registry.csv")
    h = pd.read_csv(ROOT / "data_web/holdings.csv")
    ok = []
    for _, f in reg.iterrows():
        try:
            asof, src = str(today), f.url
            d = None
            if f.format != "nport":      # issuer's own daily file; on any failure fall back to SEC N-PORT below
                try:
                    d, asof = fetch_daily(f, today)
                except Exception as e:
                    log(f"holdings {f.etf}: daily file failed ({str(e)[:90]}); using SEC filing instead")
            if d is None:                # SEC EDGAR N-PORT (free, full list, 1-4 months old)
                import edgar as EG
                d, asof, _, src = EG.fetch_fund(f.etf if f.format != "nport" or not (isinstance(f.url, str) and f.url) else f.url)
                asof = asof or str(today)
                if (today - pd.Timestamp(asof).date()).days > 150: raise ValueError(f"newest public N-PORT is {asof}, too old")
            if len(d) < 5 or d.weight.sum() < 30: raise ValueError(f"only {len(d)} rows / {d.weight.sum():.0f}% of weight")
            hist = ROOT / "data/holdings_history" / f.etf; hist.mkdir(parents=True, exist_ok=True)
            d.to_csv(hist / f"{asof}.csv", index=False)
            h = h[h.etf != f.etf]
            h = pd.concat([h, d.assign(etf=f.etf, source=src, asof=str(asof), fresh=True)[["etf", "ticker", "weight", "source", "asof", "fresh"]]])
            ok.append(f.etf); log(f"holdings {f.etf}: {len(d)} rows, {d.weight.sum():.1f}% of weight, as of {asof}")
        except Exception as e:
            log(f"holdings {f.etf}: FAILED ({e}); keeping previous list")
    # candidates: funds whose daily file is not yet confirmed. Each run tries them; one that reads cleanly joins the registry for good.
    cpath = ROOT / "fund_candidates.csv"
    if cpath.exists():
        cand = pd.read_csv(cpath); promoted = []
        for _, f in cand[~cand.etf.isin(reg.etf)].iterrows():
            try:
                d, asof = fetch_daily(f, today)
                if len(d) < 10 or d.weight.sum() < 30: raise ValueError(f"only {len(d)} rows / {d.weight.sum():.0f}% of weight")
                hist = ROOT / "data/holdings_history" / f.etf; hist.mkdir(parents=True, exist_ok=True); d.to_csv(hist / f"{asof}.csv", index=False)
                h = pd.concat([h[h.etf != f.etf], d.assign(etf=f.etf, source=f.url, asof=str(asof), fresh=True)[["etf", "ticker", "weight", "source", "asof", "fresh"]]])
                promoted.append(f); ok.append(f.etf); log(f"holdings {f.etf}: NEW FUND confirmed, {len(d)} rows, as of {asof}")
            except Exception as e:
                log(f"candidate {f.etf}: not added yet ({str(e)[:90]})")
        if promoted:
            reg = pd.concat([reg, pd.DataFrame(promoted)[reg.columns]], ignore_index=True); reg.to_csv(ROOT / "fund_registry.csv", index=False)
    h.to_csv(ROOT / "data_web/holdings.csv", index=False)
    log(f"funds in the overlap analysis: {h.etf.nunique()} ({sum(1 for _, f in reg.iterrows() if f.format != 'nport')} on daily issuer files)")
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


def refresh_regime(today):
    """Market stress gauges (shown as a warning, not used to change weights: the backtest found switching defensive cost returns,
    because stress months are often rebounds). S&P 500 vs its 200-day average from Yahoo; high-yield credit spreads from FRED."""
    out = dict(asof=str(today))
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
        out.update(spy=round(float(spy.iloc[-1]), 2), spy_200d=round(float(spy.iloc[-200:].mean()), 2), spy_below_200d=bool(spy.iloc[-1] < spy.iloc[-200:].mean()))
    except Exception as e: log(f"regime: SPY FAILED ({e})")
    try:
        txt = None
        for wait in (0, 20, 60):                    # FRED is often slow: three tries with a longer timeout
            time.sleep(wait)
            try: txt = get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2", timeout=120); break
            except Exception as e: log(f"regime: credit spreads try failed ({e})")
        if txt is None: raise RuntimeError("FRED did not answer after 3 tries")
        o = pd.read_csv(io.StringIO(txt)); o.columns = ["date", "oas"]; o["oas"] = pd.to_numeric(o.oas, errors="coerce"); o = o.dropna()
        (ROOT / "data/macro").mkdir(parents=True, exist_ok=True); o.to_csv(ROOT / "data/macro/hy_oas.csv", index=False)
        out.update(hy_spread=round(float(o.oas.iloc[-1]), 2), hy_spread_3m_change=round(float(o.oas.iloc[-1] - o.oas.iloc[-64]), 2),
                   spreads_widening=bool(o.oas.iloc[-1] - o.oas.iloc[-64] > 0.75))
    except Exception as e: log(f"regime: credit spreads FAILED ({e})")
    out["risk_off"] = bool(out.get("spy_below_200d") or out.get("spreads_widening"))
    (ROOT / "docs/data").mkdir(parents=True, exist_ok=True); (ROOT / "docs/data/regime.json").write_text(json.dumps(out))
    log(f"market regime: {'STRESS' if out['risk_off'] else 'normal'} ({out})")


def refresh_fund_returns(today):
    """1- and 3-year total returns for every fund and benchmark from Yahoo adjusted prices, so 'skilled manager' is judged on live numbers."""
    try:
        import yfinance as yf
        u = pd.read_csv(ROOT / "etf_universe.csv"); names = sorted(set(u.etf) | set(u.benchmark))
        px = yf.download(names, period="3y", interval="1d", auto_adjust=True, progress=False, threads=True)["Close"]
        rows = []
        for n in names:
            if n not in px.columns: continue
            c = px[n].dropna()
            if len(c) < 240: continue
            last = c.iloc[-1]; at = lambda days: c[c.index <= c.index[-1] - pd.Timedelta(days=days)]
            y1 = at(365); y3 = at(3 * 365 - 10)
            r1 = (last / y1.iloc[-1] - 1) * 100 if len(y1) else np.nan
            r3 = ((last / y3.iloc[-1]) ** (1 / 3) - 1) * 100 if len(y3) and (c.index[-1] - c.index[0]).days >= 3 * 365 - 15 else np.nan
            ytd = c[c.index < pd.Timestamp(today.year, 1, 1)]
            rows.append(dict(fund=n, source="yahoo", ret_1y=round(r1, 2), ret_3y=round(r3, 2) if pd.notna(r3) else np.nan,
                             ytd=round((last / ytd.iloc[-1] - 1) * 100, 2) if len(ytd) else np.nan))
        old = pd.read_csv(ROOT / "data_web/returns.csv")
        new = pd.concat([old[old.source != "yahoo"], pd.DataFrame(rows)], ignore_index=True)
        new.to_csv(ROOT / "data_web/returns.csv", index=False); log(f"fund returns (Yahoo): {len(rows)} funds and benchmarks")
    except Exception as e:
        log(f"fund returns FAILED ({e}); keeping previous values")


def refresh_prices(tickers, a):
    """12-1 month momentum and distance from the 52-week high for the whole pool. Yahoo via yfinance. Returns the closes (for reactions)."""
    import autodata as AD
    n = 0; frames = []
    try:
        import yfinance as yf
        tickers = list(tickers) + ["SPY", "QQQ"]
        for k in range(0, len(tickers), 100):
            b = list(tickers[k:k + 100])
            d = yf.download(b, period="14mo", interval="1d", auto_adjust=True, progress=False, threads=True)["Close"]
            if hasattr(d, "to_frame") and getattr(d, "ndim", 2) == 1: d = d.to_frame(b[0])
            frames.append(d)
            for t in b:
                if t not in d.columns or t not in a.index: continue
                c = d[t].dropna()
                if len(c) < 240: continue
                st = price_stats("Date,Close\n" + "\n".join(f"{i.date()},{v}" for i, v in c.items()))
                if st:
                    AD.setv(a, t, "mom_12_1", round(st["mom_12_1"], 4)); AD.setv(a, t, "off_high", round(st["off_high"], 4)); n += 1
    except Exception as e:
        log(f"prices FAILED ({e})")
    log(f"prices: updated {n} names")
    return pd.concat(frames, axis=1).loc[:, lambda x: ~x.columns.duplicated()] if frames else pd.DataFrame()


# ---------------------------------------------------------------- estimate revisions + consensus growth (own daily snapshots)
def refresh_revisions(pool, today, a):
    import autodata as AD
    try:
        import revisions as RV
        snaps = RV.snapshot(pool, today, log=log)
        r = RV.rev_30d(snaps, today)
        for t, row in r.iterrows():
            if t in a.index:
                AD.setv(a, t, "rev_30d", round(float(row.rev_30d), 4)); AD.setv(a, t, "rev_source", str(row.rev_source))
                if pd.notna(row.rev_breadth): AD.setv(a, t, "rev_breadth", round(float(row.rev_breadth), 4))
        g = RV.growth_latest(snaps); ng = 0
        for t, row in g.iterrows():
            if t not in a.index: continue
            for c in ("fwd_growth", "cur_growth", "rev_growth", "dispersion"):
                if c in row.index and pd.notna(row[c]): AD.setv(a, t, c, round(float(row[c]), 4))
            ng += 1
        for t in set(r.index) & set(a.index): AD.setv(a, t, "consensus_asof", str(today))
        log(f"revisions: {len(r)} names ({(r.rev_source == 'own').sum()} from our own history); consensus growth: {ng} names")
    except Exception as e:
        log(f"revisions FAILED ({e}); keeping previous values")


# ---------------------------------------------------------------- earnings (Yahoo via yfinance, free, no key)
def parse_earnings_dates(e, today, n=4):
    """yfinance get_earnings_dates() frame -> (last n surprises oldest->newest as fractions, next report date or None, past event rows).
    Surprise is taken from Yahoo's Surprise(%) column, or worked out from estimate vs reported EPS when that column is missing."""
    if e is None or not len(e): return [], None, []
    e = e.reset_index(); dcol = e.columns[0]
    e["d"] = pd.to_datetime(e[dcol], utc=True).dt.tz_convert(None).dt.normalize()
    act = pd.to_numeric(e.get("Reported EPS"), errors="coerce"); est = pd.to_numeric(e.get("EPS Estimate"), errors="coerce")
    sur = pd.to_numeric(e["Surprise(%)"], errors="coerce") / 100 if "Surprise(%)" in e.columns else pd.Series(np.nan, index=e.index)
    calc = (act - est) / est.abs().where(est.abs() > 0)
    e["sur"] = sur.fillna(calc); e["act"] = act; e["est"] = est
    past = e[e.act.notna() & (e.d <= pd.Timestamp(today))].sort_values("d")
    fut = e[e.act.isna() & (e.d > pd.Timestamp(today) - pd.Timedelta(days=1))].sort_values("d")
    s = [round(float(x), 4) for x in past.sur.dropna().tail(n)]
    nxt = str(fut.d.iloc[0].date()) if len(fut) else None
    rows = [dict(date=str(d), eps_est=a, eps_act=b, surprise_pct=None if pd.isna(x) else round(float(x) * 100, 4)) for d, a, b, x in zip(past.d, past.est, past.act, past.sur)]
    return s, nxt, rows


def due_for_earnings(pool, base, seen, today, stale_days=7):
    """Names to refresh today: never done, refreshed over a week ago, or whose last known report date has now passed."""
    out = []
    for t in pool:
        if t not in base.index: continue
        last = seen.get(t); nxt = base.loc[t, "next_earnings"] if "next_earnings" in base.columns else None
        if not last or (pd.Timestamp(today) - pd.Timestamp(last)).days >= stale_days: out.append(t); continue
        if isinstance(nxt, str) and nxt and pd.Timestamp(last) <= pd.Timestamp(nxt) <= pd.Timestamp(today): out.append(t)
    return sorted(out, key=lambda t: seen.get(t, ""))


def refresh_earnings(pool, today, a, cap=500, fetch=None, sleep=0.4):
    """Surprise history (last 4 quarters), next report date and the event log for the whole pool, all from Yahoo."""
    import autodata as AD
    if fetch is None:
        import yfinance as yf
        fetch = lambda t: yf.Ticker(t).get_earnings_dates(limit=12)
    state = ROOT / "data/yahoo_earnings_state.json"; seen = json.loads(state.read_text()) if state.exists() else {}
    todo = due_for_earnings(pool, a, seen, today)[:cap]; ok = fail = 0; events = []
    for t in todo:
        try:
            s, nxt, rows = parse_earnings_dates(fetch(t), today)
            if len(s) >= 2:
                s = ([np.nan] * 4 + s)[-4:]
                for i, v in enumerate(s): AD.setv(a, t, f"s{i+1}", v)
            if nxt: AD.setv(a, t, "next_earnings", nxt)
            AD.setv(a, t, "earnings_asof", str(today))
            if rows: AD.setv(a, t, "last_report", str(pd.Timestamp(rows[-1]["date"]).date()))
            events += [dict(ticker=t, **r) for r in rows]
            seen[t] = str(today); ok += 1
        except Exception as e:
            fail += 1; log(f"earnings {t}: {str(e)[:100]}")
        if sleep: time.sleep(sleep)
    state.write_text(json.dumps(seen))
    if events:   # keep the event log current for the factor backtest and the reaction signal (one row per ticker and report date)
        p = ROOT / "data/earnings_events.csv"
        old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["ticker", "date", "eps_est", "eps_act", "surprise_pct"])
        new = pd.DataFrame(events); old["date"] = pd.to_datetime(old.date, errors="coerce").dt.normalize().astype(str); new["date"] = pd.to_datetime(new.date).dt.normalize().astype(str)
        pd.concat([old, new]).drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"]).to_csv(p, index=False)
    log(f"earnings (Yahoo): {ok} names refreshed, {fail} failed, {len(todo)} due of {len(pool)}")


def refresh_reactions(closes, today, a):
    import autodata as AD
    p = ROOT / "data/earnings_events.csv"
    if closes.empty or not p.exists(): log("reactions: no prices or events, skipped"); return
    ev = pd.read_csv(p); r = AD.reactions(closes, ev, today)
    lr = pd.to_datetime(ev.date, errors="coerce").dt.normalize()
    lr = lr[lr <= pd.Timestamp(today)].groupby(ev.ticker).max()          # last report date, for the fresh-earnings weighting
    for t, d in lr.items():
        if t in a.index and pd.notna(d): AD.setv(a, t, "last_report", str(d.date()))
    for t, v in r.items():
        if t in a.index: AD.setv(a, t, "reaction", round(v, 4))
    log(f"market reaction to reports: {len(r)} names")


# ---------------------------------------------------------------- rebuild + snapshot
def rebuild(today, weekly):
    import tempfile
    sys.path.insert(0, str(ROOT))
    subprocess.run([sys.executable, str(ROOT / "data_web/build_live.py")], cwd=ROOT, check=False, capture_output=True)
    import weekly as W, diagnostics as D
    from web_snapshot import WebSnapshotProvider
    prov, cfg = WebSnapshotProvider(), W.Config()
    prev = sorted(f for f in (ROOT / "data/daily").glob("20??-??-??.csv") if f.stem < str(today))   # sell discipline: yesterday's board
    if prev:
        pb = pd.read_csv(prev[-1]); cfg.prev_board = tuple(pb[pb["rank"] <= cfg.top_n].ticker) if "rank" in pb.columns else ()
        cfg.prev_rc_board = tuple(pb[pb["rc_rank"] <= W.RC_N].ticker) if "rc_rank" in pb.columns else ()
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


def freshness_report(a, pool, today):
    """One line per input: how many candidates have it, and how many were updated in the last 7 days."""
    cols = {"momentum": ("mom_12_1", None), "EPS growth": ("fwd_growth", "consensus_asof"), "revisions": ("rev_30d", "consensus_asof"),
            "sales growth": ("rev_growth", "consensus_asof"), "surprises": ("s4", "earnings_asof"), "reaction": ("reaction", None),
            "short interest": ("short_pct", "info_asof"), "beta": ("beta", "info_asof"), "gross profitability": ("gross_prof", "sec_asof"),
            "accruals": ("accruals", "sec_asof"), "asset growth": ("asset_growth", "sec_asof"), "net issuance": ("net_issuance", "sec_asof"),
            "insider buying": ("insider_buyers", None)}
    a = a.reindex(pool); out = [f"coverage of {len(pool)} candidates (has value / updated within 7 days):"]
    for lab, (c, asof) in cols.items():
        has = int(a[c].notna().sum()) if c in a.columns else 0
        fresh = int((pd.to_datetime(a[asof], errors="coerce") >= pd.Timestamp(today) - pd.Timedelta(days=7)).sum()) if asof and asof in a.columns else has
        out.append(f"  {lab}: {has} / {fresh}")
    return "\n".join(out)


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


def fetch_earnings_events(tickers, log=log, limit=32, cap=700):
    """Past report dates with estimate, actual and surprise % from Yahoo, saved to data/earnings_events.csv (kept, never overwritten)."""
    import yfinance as yf
    p = ROOT / "data/earnings_events.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["ticker", "date", "eps_est", "eps_act", "surprise_pct"])
    first = pd.to_datetime(old.date, errors="coerce").groupby(old.ticker).min() if len(old) else pd.Series(dtype="datetime64[ns]")
    done = set(first[first <= pd.Timestamp("2019-06-30")].index); rows = []      # names without history back to mid-2019 are fetched again
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
    if rows:
        old = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        old["date"] = pd.to_datetime(old.date, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize().astype(str)
        old = old.drop_duplicates(["ticker", "date"], keep="last")
    old.to_csv(p, index=False); log(f"earnings events: {len(old)} rows for {old.ticker.nunique()} names")


def history_mode(today):
    probe()
    """One-off (re-runnable): pull every fund's past N-PORT filings and daily prices for the backtest, then run it."""
    import edgar as EG, subprocess
    reg = pd.read_csv(ROOT / "fund_registry.csv")
    for _, f in reg.iterrows():
        try: EG.history(f.url if f.format == "nport" and isinstance(f.url, str) and f.url else f.etf, n=100, out_dir=ROOT / "data/holdings_history" / f.etf, log=log)
        except Exception as e: log(f"history {f.etf}: {e}")
    names = set()
    for p in (ROOT / "data/holdings_history").glob("*/*.csv"):
        if p.name != "filed.csv": names |= set(pd.read_csv(p).ticker.astype(str))
    pdir = ROOT / "data/prices"; pdir.mkdir(parents=True, exist_ok=True)
    names |= {"SPY", "IWF", "IWD", "MTUM", "QQQ"}          # index and factor ETFs for the alpha test
    def fresh_enough(t):            # re-download a file that is a week old or does not reach back to the start of 2019
        f = pdir / f"{t}.csv"
        if not f.exists() or time.time() - f.stat().st_mtime >= 7 * 86400: return False
        try: return pd.read_csv(f, nrows=1, parse_dates=["Date"]).Date.iloc[0] <= pd.Timestamp("2019-01-31")
        except Exception: return False
    todo = [t for t in sorted(names) if not fresh_enough(t)]
    # Yahoo via yfinance (Stooq times out from GitHub's servers). Batches of 50, slowed down; anything a batch misses is retried
    # one ticker at a time, and the coverage is logged so a partial download cannot pass unnoticed.
    import yfinance as yf
    def save(t, s):
        s = s.dropna()
        if len(s) > 100: s.rename("Close").rename_axis("Date").reset_index().to_csv(pdir / f"{t}.csv", index=False); return True
        return False
    got, missed = set(), []
    for k in range(0, len(todo), 50):
        b = todo[k:k + 50]
        try:
            d = yf.download(b, period="8y", interval="1d", auto_adjust=True, progress=False, threads=False)["Close"]
            if not hasattr(d, "columns"): d = d.to_frame(b[0])
            for t in b:
                if t in d.columns and save(t, d[t]): got.add(t)
        except Exception as e: log(f"prices batch {k}: {str(e)[:80]}")
        missed += [t for t in b if t not in got]; time.sleep(2)
    for t in missed:
        try:
            d = yf.download(t, period="8y", interval="1d", auto_adjust=True, progress=False)["Close"]
            if save(t, d.squeeze() if hasattr(d, "squeeze") else d): got.add(t)
        except Exception: pass
        time.sleep(0.5)
    early = sum(1 for f in pdir.glob("*.csv") if pd.read_csv(f, nrows=1).Date.iloc[0] <= "2019-01-31")
    log(f"prices: {len(got)} of {len(todo)} downloaded; {early} of {len(list(pdir.glob('*.csv')))} files reach back to January 2019")
    subprocess.run([sys.executable, str(ROOT / "backtest.py")], cwd=ROOT)
    try:
        pool = pd.read_csv(ROOT / "data_web/holdings.csv").groupby("ticker").etf.nunique().loc[lambda s: s >= 2].index.tolist()
        fetch_earnings_events(pool)
    except Exception as e: log(f"earnings events FAILED ({e})")
    subprocess.run([sys.executable, str(ROOT / "factor_backtest.py")], cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "pit_backtest.py")], cwd=ROOT)
    try:
        import build_site; build_site.main()
    except Exception as e: log(f"site data FAILED ({e})")
    (ROOT / "data/history_run.log").write_text("\n".join(LOG))


def main():
    today = date.today(); weekly = "--weekly" in sys.argv
    if "--history" in sys.argv: return history_mode(today)
    import autodata as AD
    ok = refresh_holdings(today)
    refresh_fund_returns(today); refresh_regime(today)
    try:
        import autodata as _AD; _AD.capex_gauge(today, log=log)
    except Exception as e: log(f"AI capex gauge FAILED ({e})")
    pool = pd.read_csv(ROOT / "data_web/holdings.csv").groupby("ticker").etf.nunique().loc[lambda s: s >= 2].index.tolist()
    a = AD.load_auto(pool)
    steps = [lambda: refresh_revisions(pool, today, a), lambda: refresh_earnings(pool, today, a),
             lambda: refresh_reactions(closes, today, a), lambda: AD.refresh_info(a, pool, today, log=log),
             lambda: AD.refresh_sec(a, pool, today, log=log), lambda: AD.refresh_insiders(a, pool, today, log=log),
             lambda: __import__("signals2").refresh(a, pool, today, closes, log=log)]
    closes = refresh_prices(pool, a); AD.save_auto(a)
    for step in steps:                        # each source is independent: one failing never stops the rest, progress is saved after each
        try: step()
        except Exception as e: log(f"step FAILED ({type(e).__name__}: {e})")
        AD.save_auto(a)
    log(freshness_report(a, pool, today))
    rk = rebuild(today, weekly)
    try:
        import pit_backtest; r = pit_backtest.main()
        oos = r.get("out_of_sample", {}).get("current", {})
        log(f"point-in-time backtest: {r.get('months_total', 0)} months; out of sample {oos.get('ex_pool_month')} per month vs pool, t = {oos.get('t_pool')}")
    except Exception as e:
        log(f"point-in-time backtest FAILED ({type(e).__name__}: {e})")
    try:
        import paper; paper.update(rk, closes, today, log=log)
    except Exception as e:
        log(f"paper portfolio FAILED ({type(e).__name__}: {e})")
    try:
        import emerging; em = emerging.build(AD.load_auto(), pool, pd.read_csv(ROOT / "fund_registry.csv"), today)
        keep = ["rank", "ticker", "name", "sector", "score", "signals", "why"] + list(emerging.WEIGHTS) + ["top_theme"]
        (ROOT / "docs/data").mkdir(parents=True, exist_ok=True)
        (ROOT / "docs/data/emerging.json").write_text(em[[c for c in keep if c in em.columns]].head(60).to_json(orient="records"))
        log(f"emerging winners: {len(em)} names with 3+ signals; top 5: {', '.join(em.ticker.head(5))}")
    except Exception as e:
        log(f"emerging winners FAILED ({type(e).__name__}: {e})")
    try:
        import portfolio; log(portfolio.write(rk, closes, AD.load_auto(), today))
    except Exception as e:
        log(f"track record / concentration FAILED ({type(e).__name__}: {e})")
    try:
        import build_site; build_site.main()
    except Exception as e:
        log(f"site data FAILED ({e})")
    (ROOT / "data/daily" / f"{today}.log").write_text("\n".join(LOG))


def earnings_selftest():
    idx = pd.to_datetime(["2026-11-20", "2026-08-21", "2026-05-29", "2026-02-27", "2025-11-21"]).tz_localize("America/New_York")
    e = pd.DataFrame({"EPS Estimate": [0.8, 0.6, 0.5, 0.4, 0.4], "Reported EPS": [np.nan, 0.66, 0.5, 0.5, 0.42], "Surprise(%)": [np.nan, 10.0, 0.0, 25.0, 5.0]}, index=idx)
    s, nxt, rows = parse_earnings_dates(e, date(2026, 10, 29))
    assert s == [0.05, 0.25, 0.0, 0.1] and nxt == "2026-11-20" and len(rows) == 4, (s, nxt)
    s2, _, _ = parse_earnings_dates(e.drop(columns="Surprise(%)"), date(2026, 10, 29))   # column missing: work it out from EPS
    assert abs(s2[-1] - 0.1) < 1e-9
    base = pd.DataFrame({"next_earnings": ["2026-10-28", "2026-12-01", None]}, index=["A", "B", "C"])
    due = due_for_earnings(["A", "B", "C", "Z"], base, {"A": "2026-10-27", "B": "2026-10-27"}, date(2026, 10, 29))
    assert set(due) == {"A", "C"}, due    # A just reported, C never done, B fresh, Z not in base
    print("earnings selftest ok")


def holdings_selftest():
    ark = ('date,fund,company,ticker,cusip,shares,market value ($),weight (%)\r\n10/29/2026,ARKW,TESLA INC,TSLA,88160R101,"361,353","$146,066,109.66",8.49%\r\n'
           '10/29/2026,ARKW,META PLATFORMS INC-CLASS A,META,30303M102,"143,896","$88,134,861.04",5.12%\r\n"Investors should carefully consider"\r\n')
    d = parse_holdings_csv(ark); assert list(d.ticker) == ["META", "TSLA"] and abs(d.weight.sum() - 13.61) < 1e-9, d
    assert csv_asof(ark) == "2026-10-29" and csv_asof('iShares\nFund Holdings as of,"Oct 27, 2026"\n') == "2026-10-27"
    try:
        import openpyxl
    except ImportError:
        print("holdings selftest: openpyxl missing, Excel parser not tested"); return
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Disclosure"; ws["B1"] = "Important Disclosures"
    w2 = wb.create_sheet("Daily Fund Holdings")
    w2.append(["Holdings for ETF", "CGGR - Capital Group Growth ETF", None, None, None, None, "As Of 10/28/2026"])
    w2.append(["Security Name", "Ticker", "Asset Type", "Shares or Principal Amount", "Market Value", "Percent of Net Assets", "CUSIP"])
    w2.append(["META PLATFORMS INC", "META", "Equity", 2433122, 1.49e9, 0.0571, "30303M102"])
    w2.append(["ALPHABET INC CL C", "GOOG", "Equity", 2534079, 8.1e8, 0.0313, "02079K107"])
    w2.append(["ALPHABET INC CL A", "GOOGL", "Equity", 2415951, 7.8e8, 0.0301, "02079K305"])
    b = io.BytesIO(); wb.save(b)
    d, asof = parse_holdings_xlsx(b.getvalue())
    assert asof == "2026-10-28" and abs(d.set_index("ticker").weight["GOOGL"] - 6.14) < 1e-9 and abs(d.weight.sum() - 11.85) < 1e-9, (d, asof)
    print("holdings selftest ok")


def selftest():
    import paper; paper.selftest(); import emerging; emerging.selftest(); import signals2; signals2.selftest(); import pit_backtest; pit_backtest.selftest(); import portfolio; portfolio.selftest(); holdings_selftest(); earnings_selftest(); import autodata; autodata.selftest()
    import edgar, revisions, backtest, form4; edgar.selftest(); revisions.selftest(); backtest.selftest(); form4.selftest(); import factor_backtest; factor_backtest.selftest()
    s = 'iShares\nFund Holdings as of,"Oct 21, 2026"\nTicker,Name,Sector,Asset Class,Market Value,Weight (%),Notional\nMU,MICRON,IT,Equity,1,6.05,1\nNVDA,NVIDIA,IT,Equity,1,5.66,1\nUSD,USD CASH,Cash,Cash,1,-0.03,1\nBRK.B,BERKSHIRE,F,Equity,1,2.0,1\nGOOG,ALPHABET,C,Equity,1,1.0,1\n'
    d = parse_holdings_csv(s); assert list(d.ticker) == ["BRK-B", "GOOGL", "MU", "NVDA"], d
    h = "<table><tr><th>Ticker</th><th>Weight %</th></tr><tr><td>NVDA</td><td>9.97%</td></tr><tr><td>AMD</td><td>5.83%</td></tr></table>"
    assert parse_holdings_html(h).weight.sum() > 15
    px = "Date,Open,High,Low,Close,Volume\n" + "\n".join(f"{d.date()},1,1,1,{100+i*0.1},1" for i, d in enumerate(pd.date_range("2025-01-01", periods=400)))
    st = price_stats(px); assert st and abs(st["off_high"]) < 1e-9 and st["mom_12_1"] > 0
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
