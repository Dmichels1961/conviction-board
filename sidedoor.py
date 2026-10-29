"""Earnings side door: strong reporters that the fund list does not own yet. Watchlist only.

Why: the main board only scores stocks held by 2+ funds, so a small/mid cap with a big quarter can go unseen until managers buy it.
Post-earnings drift (Bernard & Thomas 1989; Chordia et al. 2009) is strongest in smaller, less-covered stocks, so this list looks for them
early. It never feeds the main board, the risk-controlled board or the paper portfolio. It earns a place there only by its own live record
(PROMOTION_RULE below), the same way manager flow has to.

How it works (all free, no key):
  1. Universe: Russell 2000 + Russell Midcap constituents (iShares IWM and IWR holdings files), plus every name in the fund holdings and the
     benchmark files already in this repo. Cached; refreshed weekly. Penny stocks and anything outside those indexes are excluded.
  2. Who reported: the Nasdaq earnings calendar (one request per day covers every US report, with EPS, consensus and surprise %).
     If Nasdaq cannot be reached, a capped Yahoo scan of names held by 0-1 funds keeps the list alive.
  3. Trigger: reported in the last 30 days, held by 0 or 1 fund on the main list, EPS beat of 10% or more on a positive estimate.
  4. Enrich (Yahoo, capped, cached a week): last 4 surprises, 30-day change in consensus EPS, and the 2-day move vs SPY after the report.
  5. Score = percentile within the list: fresh surprise 30%, revisions 30%, beat consistency 20%, market reaction 20%. Missing = neutral.
  6. Track record: every name is logged with its price the day it first appears; returns since entry are compared with SPY and QQQ.

Outputs: data/sidedoor/<date>.csv, data/sidedoor/entries.csv (the live record), docs/data/sidedoor.json (the site).
"""
import io, json, re, time, urllib.request
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "data/sidedoor"
LOOKBACK_DAYS, MIN_SURPRISE, MAX_HOLDERS, N_SHOW = 30, 0.10, 1, 40
WEIGHTS = {"surprise": 0.30, "rev_30d": 0.30, "consistency": 0.20, "reaction": 0.20}
UNIVERSE_URLS = {
    "IWM": "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
    "IWR": "https://www.ishares.com/us/products/239718/ishares-russell-midcap-etf/1467271812596.ajax?fileType=csv&fileName=IWR_holdings&dataType=fund",
}
NASDAQ = "https://api.nasdaq.com/api/calendar/earnings?date={d}"
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9", "Origin": "https://www.nasdaq.com",
           "Referer": "https://www.nasdaq.com/"}
PROMOTION_RULE = ("Watchlist only. It joins the main board's scoring (as an earnings-led entry with no overlap points) only if names entering "
                  "the side door beat the S&P 500 by 2+ points on average over 90 trading days, across 40+ entries spanning 6+ months. "
                  "Until then it cannot change the boards or the paper portfolio.")


def _get(url, headers=None, timeout=40):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or BROWSER), timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def money(s):
    """'$1.23' -> 1.23, '($0.45)' -> -0.45, 'N/A' or '' -> nan."""
    if s is None: return np.nan
    s = str(s).strip()
    if not s or s.upper() in ("N/A", "NA", "--", "-"): return np.nan
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-")
    s = re.sub(r"[^\d.]", "", s)
    if not s: return np.nan
    try: v = float(s)
    except ValueError: return np.nan
    return -v if neg else v


# ----------------------------------------------------------------- 1. universe
def load_universe(today, fetch=_get, log=print, max_age_days=7):
    p = DIR / "universe.csv"
    if p.exists():
        u = pd.read_csv(p)
        if len(u) and (pd.Timestamp(today) - pd.Timestamp(str(u["asof"].iloc[0]))).days < max_age_days: return set(u.ticker)
    from refresh import parse_holdings_csv
    names = set()
    for etf, url in UNIVERSE_URLS.items():
        try:
            d = parse_holdings_csv(fetch(url, headers={"User-Agent": BROWSER["User-Agent"]}))
            if len(d) < 300: raise ValueError(f"only {len(d)} rows")
            names |= set(d.ticker); log(f"side door universe {etf}: {len(d)} names")
        except Exception as e:
            log(f"side door universe {etf}: FAILED ({str(e)[:90]})")
    for f in ("data_web/holdings.csv", "data_web/benchmarks.csv"):
        if (ROOT / f).exists(): names |= set(pd.read_csv(ROOT / f).ticker.dropna().astype(str))
    if len(names) < 500 and p.exists(): return set(pd.read_csv(p).ticker)      # index files failed: keep last week's list
    DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": sorted(names), "asof": str(today)}).to_csv(p, index=False)
    return names


# ----------------------------------------------------------------- 2. who reported
def parse_nasdaq(text, d):
    j = json.loads(text); rows = ((j.get("data") or {}).get("rows")) or []
    out = []
    for r in rows:
        t = str(r.get("symbol", "")).strip().upper().replace(".", "-").replace("/", "-")
        if not t: continue
        act, est = money(r.get("eps")), money(r.get("epsForecast"))
        sur = money(r.get("surprise"))
        sur = sur / 100 if sur == sur else ((act - est) / abs(est) if act == act and est == est and est != 0 else np.nan)
        out.append(dict(date=str(d), ticker=t, eps_act=act, eps_est=est, surprise=sur, mcap=money(r.get("marketCap"))))
    return out


def refresh_calendar(today, fetch=_get, log=print, cap=40, sleep=1.0):
    """Fetch any missing report day in the lookback window (weekends skipped). Returns all calendar rows, newest days included."""
    DIR.mkdir(parents=True, exist_ok=True)
    p, sp = DIR / "calendar.csv", DIR / "calendar_days.json"
    cal = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["date", "ticker", "eps_act", "eps_est", "surprise", "mcap"])
    done = set(json.loads(sp.read_text())) if sp.exists() else set()
    days = [today - timedelta(days=k) for k in range(LOOKBACK_DAYS + 2)]
    todo = [d for d in days if d.weekday() < 5 and (str(d) not in done or d >= today - timedelta(days=2))][:cap]   # last 2 days re-read: late rows
    ok = fail = 0; new = []
    for d in todo:
        try:
            new += parse_nasdaq(fetch(NASDAQ.format(d=d)), d); done.add(str(d)); ok += 1
        except Exception as e:
            fail += 1
            if fail <= 2: log(f"side door calendar {d}: {str(e)[:90]}")
        if sleep: time.sleep(sleep)
    if new:
        cal = pd.concat([cal, pd.DataFrame(new)], ignore_index=True).drop_duplicates(["date", "ticker"], keep="last")
    cutoff = str(today - timedelta(days=120))
    cal = cal[cal.date.astype(str) >= cutoff]
    cal.to_csv(p, index=False); sp.write_text(json.dumps(sorted(x for x in done if x >= cutoff)))
    log(f"side door calendar: {ok} days read, {fail} failed, {len(cal)} reports on file")
    return cal, ok > 0 or fail == 0


def yahoo_scan(tickers, today, fetch_dates, log=print, cap=200):
    """Fallback when Nasdaq is unreachable: rotate through the candidates with Yahoo, a slice per day."""
    sp = DIR / "yahoo_scan.json"; seen = json.loads(sp.read_text()) if sp.exists() else {}
    from refresh import parse_earnings_dates
    todo = sorted(tickers, key=lambda t: seen.get(t, ""))[:cap]; rows = []
    for t in todo:
        try:
            _, _, ev = parse_earnings_dates(fetch_dates(t), today)
            for r in ev[-1:]:
                rows.append(dict(date=str(pd.Timestamp(r["date"]).date()), ticker=t, eps_act=r["eps_act"], eps_est=r["eps_est"],
                                 surprise=None if r["surprise_pct"] is None else r["surprise_pct"] / 100, mcap=np.nan))
            seen[t] = str(today)
        except Exception: pass
        time.sleep(0.3)
    sp.write_text(json.dumps(seen)); log(f"side door Yahoo fallback: {len(todo)} names checked, {len(rows)} reports")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 3. trigger
def triggers(cal, universe, holders, today):
    c = cal.copy(); c["date"] = pd.to_datetime(c.date, errors="coerce")
    c = c[(c.date >= pd.Timestamp(today) - pd.Timedelta(days=LOOKBACK_DAYS)) & (c.date <= pd.Timestamp(today))]
    c = c.sort_values("date").drop_duplicates("ticker", keep="last")
    c = c[c.ticker.isin(universe)]
    c["holders"] = c.ticker.map(holders).fillna(0).astype(int)
    c = c[(c.holders <= MAX_HOLDERS) & (pd.to_numeric(c.eps_est, errors="coerce") > 0) & (pd.to_numeric(c.surprise, errors="coerce") >= MIN_SURPRISE)]
    return c.rename(columns={"date": "report_date"}).reset_index(drop=True)


# ----------------------------------------------------------------- 4. enrich
def _yahoo_fetchers():
    import yfinance as yf
    def dates(t): return yf.Ticker(t).get_earnings_dates(limit=12)
    def trend(t):
        tr = yf.Ticker(t).eps_trend
        v = [tr.loc[p, "current"] / tr.loc[p, "30daysAgo"] - 1 for p in ("0y", "+1y")
             if p in tr.index and pd.notna(tr.loc[p, "30daysAgo"]) and tr.loc[p, "30daysAgo"] > 0 and pd.notna(tr.loc[p, "current"])]
        return float(np.mean(v)) if v else np.nan
    def prices(ts):
        d = yf.download(list(ts) + ["SPY", "QQQ"], period="1y", interval="1d", auto_adjust=True, progress=False, threads=True)["Close"]
        return d.to_frame(ts[0]) if getattr(d, "ndim", 2) == 1 else d
    return dates, trend, prices


def enrich(c, today, fetch_dates, fetch_trend, log=print, cap=80, sleep=0.4):
    p = DIR / "enrich.csv"
    cache = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["ticker", "asof", "beats", "quarters", "avg_surprise", "rev_30d"])
    cache = cache.drop_duplicates("ticker", keep="last").set_index("ticker")
    from refresh import parse_earnings_dates
    stale = [t for t in c.ticker if t not in cache.index or (pd.Timestamp(today) - pd.Timestamp(str(cache.loc[t, "asof"]))).days >= 7
             or str(cache.loc[t, "asof"]) < str(c.set_index("ticker").report_date.get(t, pd.Timestamp(today)).date())]
    n = 0
    for t in stale[:cap]:
        row = dict(asof=str(today), beats=np.nan, quarters=np.nan, avg_surprise=np.nan, rev_30d=np.nan)
        try:
            s, _, _ = parse_earnings_dates(fetch_dates(t), today)
            if s: row.update(beats=sum(1 for x in s if x > 0), quarters=len(s), avg_surprise=float(np.mean(s)))
        except Exception: pass
        try: row["rev_30d"] = fetch_trend(t)
        except Exception: pass
        cache.loc[t, list(row)] = list(row.values()); n += 1
        if sleep: time.sleep(sleep)
    cache.reset_index().to_csv(p, index=False)
    if n: log(f"side door: enriched {n} names from Yahoo")
    return c.merge(cache.reset_index()[["ticker", "beats", "quarters", "avg_surprise", "rev_30d"]], on="ticker", how="left")


def reaction(closes, t, d):
    """2-day move vs SPY: close before the report day to the close after it."""
    if closes is None or t not in closes.columns or "SPY" not in closes.columns: return np.nan
    idx = closes.index; i = idx.searchsorted(pd.Timestamp(d))
    if i < 1 or i + 1 >= len(idx): return np.nan
    p0, p1, s0, s1 = closes[t].iloc[i - 1], closes[t].iloc[i + 1], closes["SPY"].iloc[i - 1], closes["SPY"].iloc[i + 1]
    return float((p1 / p0 - 1) - (s1 / s0 - 1)) if pd.notna(p0) and pd.notna(p1) and p0 > 0 and s0 > 0 else np.nan


# ----------------------------------------------------------------- 5. score
def score(c):
    c = c.copy()
    c["consistency"] = c.beats / c.quarters.where(c.quarters > 0)
    cols = {k: c[k] if k in c.columns else pd.Series(np.nan, index=c.index) for k in WEIGHTS}
    ranks = pd.DataFrame({k: v.rank(pct=True) * 100 for k, v in cols.items()}).fillna(50.0)
    c["score"] = sum(ranks[k] * w for k, w in WEIGHTS.items()).round(1)
    c["signals"] = pd.DataFrame(cols).notna().sum(axis=1)
    c = c.sort_values(["score", "surprise"], ascending=False).reset_index(drop=True)
    c["rank"] = np.arange(1, len(c) + 1)
    def why(r):
        parts = [f"beat by {r.surprise*100:.0f}% on {pd.Timestamp(r.report_date):%b %d}"]
        if pd.notna(r.get("beats")) and r.quarters >= 3: parts.append(f"beat {int(r.beats)} of last {int(r.quarters)}")
        if pd.notna(r.get("rev_30d")) and abs(r.rev_30d) >= 0.01: parts.append(f"estimates {'+' if r.rev_30d > 0 else ''}{r.rev_30d*100:.0f}% in 30d")
        if pd.notna(r.get("reaction")) and abs(r.reaction) >= 0.02: parts.append(f"stock {'+' if r.reaction > 0 else ''}{r.reaction*100:.0f}% vs SPY on the report")
        parts.append("held by 1 fund" if r.holders == 1 else "no fund holds it")
        return "; ".join(parts)
    c["why"] = c.apply(why, axis=1) if len(c) else []
    return c


# ----------------------------------------------------------------- 6. live record
def track(listed, closes, today):
    p = DIR / "entries.csv"
    ent = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "spy", "qqq", "rank", "score"])
    last = closes.ffill().iloc[-1] if closes is not None and len(closes) else pd.Series(dtype=float)
    live = set(ent[pd.to_datetime(ent.entry_date) >= pd.Timestamp(today) - pd.Timedelta(days=90)].ticker)
    add = []
    for _, r in listed.iterrows():
        if r.ticker in live or pd.isna(last.get(r.ticker)): continue          # one entry per name per 90 days
        add.append(dict(ticker=r.ticker, entry_date=str(today), entry_price=float(last[r.ticker]), spy=float(last.get("SPY", np.nan)),
                        qqq=float(last.get("QQQ", np.nan)), rank=int(r["rank"]), score=float(r.score)))
    if add: ent = pd.concat([ent, pd.DataFrame(add)], ignore_index=True)
    ent.to_csv(p, index=False)
    if ent.empty or last.empty: return dict(entries=0), ent
    e = ent.copy()
    px = e.ticker.map(last); e["ret"] = px / e.entry_price - 1
    e["ret_spy"] = last.get("SPY", np.nan) / e.spy - 1; e["ret_qqq"] = last.get("QQQ", np.nan) / e.qqq - 1
    e = e[e.ret.notna() & (e.entry_date < str(today))]
    if e.empty: return dict(entries=int(len(ent)), first=str(ent.entry_date.min())), ent
    ex = e.ret - e.ret_spy
    return dict(entries=int(len(ent)), measured=int(len(e)), first=str(ent.entry_date.min()), avg=round(float(e.ret.mean()), 4),
                avg_spy=round(float(e.ret_spy.mean()), 4), avg_qqq=round(float(e.ret_qqq.mean()), 4),
                beat_spy=round(float((ex > 0).mean()), 3), excess=round(float(ex.mean()), 4)), ent


# ----------------------------------------------------------------- run
def run(today, log=print, fetchers=None, get=_get):
    DIR.mkdir(parents=True, exist_ok=True)
    fetch_dates, fetch_trend, fetch_prices = fetchers or _yahoo_fetchers()
    universe = load_universe(today, fetch=get, log=log)
    h = pd.read_csv(ROOT / "data_web/holdings.csv"); holders = h.groupby("ticker").etf.nunique()
    cal, ok = refresh_calendar(today, fetch=get, log=log)
    recent = cal[pd.to_datetime(cal.date, errors="coerce") >= pd.Timestamp(today) - pd.Timedelta(days=LOOKBACK_DAYS)]
    if ok and len(recent) > 50 and pd.to_numeric(recent.eps_act, errors="coerce").notna().mean() < 0.05:
        log("side door calendar: reports listed but no actual EPS in them; using the Yahoo fallback"); ok = False
    if not ok:
        thin = [t for t in universe if holders.get(t, 0) <= MAX_HOLDERS]
        yc = yahoo_scan(thin, today, fetch_dates, log=log)
        if len(yc): cal = pd.concat([cal, yc], ignore_index=True)
    c = triggers(cal, universe, holders, today)
    log(f"side door: {len(c)} names reported a 10%+ beat in the last {LOOKBACK_DAYS} days and are held by 0-1 funds")
    if c.empty:
        out = dict(asof=str(today), rows=[], rule=PROMOTION_RULE, track=track(c, None, today)[0]); _write(out); return c
    c = enrich(c, today, fetch_dates, fetch_trend, log=log)
    closes = None
    try: closes = fetch_prices(list(c.ticker) + list(_open_entries(today)))
    except Exception as e: log(f"side door prices: FAILED ({str(e)[:80]})")
    c["reaction"] = [reaction(closes, t, d) for t, d in zip(c.ticker, c.report_date)]
    c = score(c)
    c.to_csv(DIR / f"{today}.csv", index=False)
    tr, ent = track(c.head(25), closes, today)
    keep = ["rank", "ticker", "score", "report_date", "surprise", "beats", "quarters", "rev_30d", "reaction", "holders", "why"]
    rows = c[keep].head(N_SHOW).copy(); rows["report_date"] = pd.to_datetime(rows.report_date).dt.strftime("%Y-%m-%d")
    if closes is not None and len(ent):
        last = closes.ffill().iloc[-1]; e = ent.drop_duplicates("ticker", keep="last").set_index("ticker")
        rows["since"] = [round(float(last[t] / e.loc[t, "entry_price"] - 1), 4) if t in e.index and t in last.index and pd.notna(last[t]) else None for t in rows.ticker]
        rows["listed"] = [str(e.loc[t, "entry_date"]) if t in e.index else None for t in rows.ticker]
    out = dict(asof=str(today), n=int(len(c)), rows=json.loads(rows.to_json(orient="records")), rule=PROMOTION_RULE, track=tr)
    _write(out)
    log(f"side door: top 5 {', '.join(c.ticker.head(5))}; live record {tr.get('measured', 0)} measured entries")
    return c


def _open_entries(today):
    p = DIR / "entries.csv"
    if not p.exists(): return []
    e = pd.read_csv(p); return sorted(set(e[pd.to_datetime(e.entry_date) >= pd.Timestamp(today) - pd.Timedelta(days=400)].ticker))


def _write(out):
    (ROOT / "docs/data").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/data/sidedoor.json").write_text(json.dumps(out, default=lambda x: None if x != x else x))


# ----------------------------------------------------------------- offline test
def selftest():
    import tempfile, shutil
    global ROOT, DIR
    old_root, old_dir = ROOT, DIR
    tmp = Path(tempfile.mkdtemp()); ROOT, DIR = tmp, tmp / "data/sidedoor"
    try:
        (tmp / "data_web").mkdir(parents=True)
        pd.DataFrame({"etf": ["A", "B", "A", "A"], "ticker": ["BIG", "BIG", "KLIC", "OLD"], "weight": [1, 1, 0.2, 1]}).to_csv(tmp / "data_web/holdings.csv", index=False)
        pd.DataFrame({"etf": ["IWF"], "ticker": ["RNG"], "weight": [0.1]}).to_csv(tmp / "data_web/benchmarks.csv", index=False)
        today = date(2026, 10, 29)
        ish = "iShares\nFund Holdings as of,\"Oct 28, 2026\"\nTicker,Name,Sector,Asset Class,Market Value,Weight (%),Notional\n" + \
              "\n".join(f"T{i:03d},X,Ind,Equity,1,0.1,1" for i in range(400)) + "\nHRI,HERC,Ind,Equity,1,0.2,1\nBXC,BLUELINX,Ind,Equity,1,0.1,1\nBIG,BIG,Ind,Equity,1,1,1\n"
        def cal_json(rows): return json.dumps({"data": {"rows": rows}})
        days = {"2026-10-23": [dict(symbol="HRI", eps="$3.10", epsForecast="$2.50", surprise="24", marketCap="$5,000,000,000"),
                               dict(symbol="BIG", eps="$2.00", epsForecast="$1.00", surprise="100")],
                "2026-10-28": [dict(symbol="BXC", eps="$1.50", epsForecast="$1.20", surprise="N/A"),
                               dict(symbol="KLIC", eps="$0.60", epsForecast="$0.50", surprise="20"),
                               dict(symbol="RNG", eps="$1.10", epsForecast="$1.05", surprise="4.8"),
                               dict(symbol="LOSS", eps="($0.10)", epsForecast="($0.30)", surprise="67"),
                               dict(symbol="OFFX", eps="$1.00", epsForecast="$0.50", surprise="100")]}
        def get(url, headers=None, timeout=40):
            if "ishares" in url: return ish
            d = url.rsplit("=", 1)[1]; return cal_json(days.get(d, []))
        idx = pd.to_datetime(["2026-10-28", "2026-07-29", "2026-04-29", "2026-01-28"]).tz_localize("America/New_York")
        ed = pd.DataFrame({"EPS Estimate": [1.2, 1.0, 0.9, 0.8], "Reported EPS": [1.5, 1.1, 0.85, 0.9], "Surprise(%)": [25.0, 10.0, -5.6, 12.5]}, index=idx)
        bd = pd.bdate_range("2026-01-02", "2026-10-29")
        cl = pd.DataFrame({t: np.linspace(10, 20, len(bd)) for t in ["HRI", "BXC", "KLIC", "SPY", "QQQ"]}, index=bd)
        cl.loc[pd.Timestamp("2026-10-23"):, "HRI"] *= 1.10
        fx = (lambda t: ed, lambda t: 0.05 if t == "HRI" else 0.0, lambda ts: cl)
        c = run(today, log=lambda *a: None, fetchers=fx, get=get)
        got = set(c.ticker)
        assert got == {"HRI", "BXC", "KLIC"}, got        # BIG held by 2 funds, RNG beat <10%, LOSS negative estimate, OFFX not in universe
        bxc = c.set_index("ticker").loc["BXC"]; assert abs(bxc.surprise - 0.25) < 1e-9, bxc          # N/A surprise worked out from EPS
        assert c.ticker.iloc[0] == "HRI", c[["ticker", "score"]]        # biggest revision and reaction ranks first
        assert c.set_index("ticker").loc["KLIC", "holders"] == 1 and "held by 1 fund" in c.set_index("ticker").loc["KLIC", "why"]
        assert c.set_index("ticker").loc["HRI", "reaction"] > 0.05
        out = json.loads((tmp / "docs/data/sidedoor.json").read_text()); assert len(out["rows"]) == 3 and out["track"]["entries"] == 3, out
        c2 = run(today + timedelta(days=1), log=lambda *a: None, fetchers=fx, get=get)            # next day: no duplicate entries
        assert len(pd.read_csv(DIR / "entries.csv")) == 3
        assert money("($0.45)") == -0.45 and money("$1,234.5") == 1234.5 and money("N/A") != money("N/A")
    finally:
        shutil.rmtree(tmp); ROOT, DIR = old_root, old_dir
    print("sidedoor selftest ok")


if __name__ == "__main__":
    selftest()
