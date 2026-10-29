"""Emerging winners: a forward-looking watchlist, kept apart from the two frozen boards.

Signals chosen because they lead future fundamentals rather than describe past returns (all free):
  rd_intensity    R&D / revenue, trailing 12 months (SEC). R&D-heavy firms are underpriced because accounting expenses the investment
                  at once (Chan, Lakonishok & Sougiannis 2001).
  rd_growth       year-on-year growth in R&D spending (SEC). Firms that step up R&D earn abnormal returns afterwards (Eberhart,
                  Maxwell & Siddique 2004).
  rev_accel       revenue growth now minus revenue growth a year earlier (SEC): the second derivative, often the first hard evidence
                  of product-market fit.
  gm_change       gross margin now minus a year earlier (SEC): operating leverage and pricing power.
  theme_level     how much the latest 10-K/10-Q talks about emerging technologies (mentions per 10,000 words), and
  theme_change    how much that rose from the same filing a year earlier (SEC text). Finds companies repositioning before revenue shows it.
  initiations     distinct managers who opened a NEW position in the last ~month, from funds with daily holdings files. The most
                  forward-looking thing fund data offers. Fills in as daily history builds.
  fwd_rev_growth  analysts' revenue growth estimate for next year (Yahoo, daily).

No penalty for missing data: young companies have little history by nature. A missing signal counts as neutral (50), and a name needs
at least 3 of the 8 signals to be listed. Not yet included: patents (USPTO PatentsView needs a free key) and federal contract awards
(USAspending; company-name matching still to build).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N = 25
WEIGHTS = {"rd_intensity": 1.0, "rd_growth": 1.0, "rev_accel": 1.5, "gm_change": 1.0, "theme_level": 0.5, "theme_change": 1.0,
           "initiations": 1.5, "fwd_rev_growth": 1.0}
THEMES = {   # phrases that mark emerging technologies; counted in filing text (case-insensitive)
    "AI": [r"artificial intelligence", r"generative ai", r"\bai\b", r"machine learning", r"large language model", r"\bllms?\b", r"ai agents?", r"inference"],
    "AI infrastructure": [r"data cent(er|re)s?", r"accelerated computing", r"\bgpus?\b", r"high[- ]bandwidth memory", r"liquid cooling", r"optical interconnect"],
    "Robotics & autonomy": [r"robotics?", r"humanoid", r"autonomous (vehicles?|driving|systems?)", r"\bdrones?\b"],
    "Energy transition & power": [r"small modular reactors?", r"nuclear", r"\bfusion\b", r"grid[- ]scale", r"energy storage", r"geothermal"],
    "Biotech breakthroughs": [r"glp-1", r"obesity", r"gene (therapy|editing)", r"crispr", r"\bcar-t\b", r"radiopharmaceutical"],
    "Space & defense tech": [r"satellites?", r"launch vehicles?", r"low earth orbit", r"hypersonic", r"space-based"],
    "Quantum & next compute": [r"quantum (computing|computers?)", r"photonic", r"neuromorphic"],
}
_RX = {k: re.compile("|".join(v), re.I) for k, v in THEMES.items()}


def theme_hits(text):
    """-> (mentions per 10,000 words in total, {theme: mentions per 10,000 words})."""
    words = max(1, len(re.findall(r"[A-Za-z]{2,}", text)))
    per = {k: 1e4 * len(rx.findall(text)) / words for k, rx in _RX.items()}
    return sum(per.values()), per


def strip_html(html):
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html); html = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"&[a-z#0-9]+;", " ", html)


# ----------------------------------------------------------------- SEC: R&D, revenue acceleration, margins (point in time)
def growth_features(cf):
    """From SEC companyfacts: R&D intensity and growth, revenue growth and acceleration, gross-margin change, each as of the latest filing."""
    import autodata as AD
    rev = AD._facts(cf, "revenue"); rd = AD._facts(cf, "rd"); gp = AD._facts(cf, "gross_profit"); cogs = AD._facts(cf, "cogs")
    r0, e0 = AD.ttm(rev)
    if e0 is None or not (r0 == r0): return {}
    at = lambda d, yrs: AD.ttm(d, e0 - pd.Timedelta(days=int(365 * yrs) - 35))[0]
    r1, r2 = at(rev, 1), at(rev, 2)
    d0, d1 = AD.ttm(rd)[0], at(rd, 1)
    g0 = AD.ttm(gp)[0]; g1 = at(gp, 1)
    if not (g0 == g0):
        c0, c1 = AD.ttm(cogs)[0], at(cogs, 1); g0 = r0 - c0 if c0 == c0 else np.nan; g1 = r1 - c1 if (c1 == c1 and r1 == r1) else np.nan
    ok = lambda *x: all(v == v and v is not None for v in x)
    gr0 = r0 / r1 - 1 if ok(r0, r1) and r1 > 0 else np.nan
    gr1 = r1 / r2 - 1 if ok(r1, r2) and r2 > 0 else np.nan
    return dict(rd_intensity=d0 / r0 if ok(d0, r0) and r0 > 0 else np.nan,
                rd_growth=d0 / d1 - 1 if ok(d0, d1) and d1 > 0 else np.nan,
                rev_growth_ttm=gr0, rev_accel=gr0 - gr1 if ok(gr0, gr1) else np.nan,
                gm_change=(g0 / r0 - g1 / r1) if ok(g0, g1, r0, r1) and r0 > 0 and r1 > 0 else np.nan)


# ----------------------------------------------------------------- manager initiations from daily holdings history
def initiations(registry, today, lookback_days=28, min_gap_days=7):
    """Distinct managers whose daily-file fund holds a stock now but did not at the oldest snapshot 7-28 days ago."""
    hist = ROOT / "data/holdings_history"; out = {}
    for _, f in registry[registry.format != "nport"].iterrows():
        d = hist / f.etf
        snaps = sorted(p for p in d.glob("20??-??-??.csv")) if d.exists() else []
        if len(snaps) < 2: continue
        now = snaps[-1]; t_now = pd.Timestamp(now.stem)
        old = [p for p in snaps if min_gap_days <= (t_now - pd.Timestamp(p.stem)).days <= lookback_days]
        if not old: continue
        a = set(pd.read_csv(now).ticker); b = set(pd.read_csv(old[0]).ticker)
        for t in a - b: out.setdefault(t, set()).add(f.manager)
    return {t: len(m) for t, m in out.items()}


# ----------------------------------------------------------------- the list
def build(auto, pool, registry, today):
    a = auto.reindex(pool).copy()
    ini = initiations(registry, today)
    a["initiations"] = pd.Series(ini, dtype=float).reindex(a.index)
    if ini: a["initiations"] = a["initiations"].fillna(0.0)          # funds were checked: no initiation is a real zero
    a["fwd_rev_growth"] = a["rev_growth"] if "rev_growth" in a.columns else np.nan
    cols = [c for c in WEIGHTS if c in a.columns]
    have = a[cols].notna().sum(axis=1)
    ranks = pd.DataFrame({c: a[c].rank(pct=True) * 100 for c in cols}).fillna(50.0)    # missing = neutral, never a penalty
    score = sum(ranks[c] * WEIGHTS[c] for c in cols) / sum(WEIGHTS[c] for c in cols)
    df = pd.DataFrame({"ticker": a.index, "score": score.round(1).values, "signals": have.values})
    for c in cols: df[c] = a[c].values
    df["name"] = a["name"].values if "name" in a.columns else None
    df["sector"] = a["sector"].values if "sector" in a.columns else None
    df = df[df.signals >= 3].sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    def why(r):
        g = lambda k: r[k] if k in r.index and pd.notna(r[k]) else None
        parts = []
        if (g("rev_accel") or 0) > 0.05: parts.append(f"revenue growth up {g('rev_accel')*100:.0f} pts")
        if (g("rd_intensity") or 0) > 0.12: parts.append(f"R&D {g('rd_intensity')*100:.0f}% of sales")
        if (g("rd_growth") or 0) > 0.2: parts.append(f"R&D +{g('rd_growth')*100:.0f}%")
        if (g("gm_change") or 0) > 0.02: parts.append(f"gross margin +{g('gm_change')*100:.1f} pts")
        if (g("theme_change") or 0) > 2: parts.append(f"{g('top_theme') or 'emerging-tech'} mentions rising")
        if (g("initiations") or 0) >= 1: parts.append(f"{int(g('initiations'))} new manager position(s)")
        return "; ".join(parts[:3])
    if "top_theme" in a.columns: df["top_theme"] = a["top_theme"].reindex(df.ticker).values
    df["why"] = df.apply(why, axis=1)
    out = ROOT / "data/emerging"; out.mkdir(parents=True, exist_ok=True); df.to_csv(out / f"{today}.csv", index=False)
    return df


def selftest():
    n, per = theme_hits("We expand our data center capacity for generative AI and AI agents. Humanoid robotics is early. Nuclear power too.")
    assert per["AI"] > 0 and per["AI infrastructure"] > 0 and per["Robotics & autonomy"] > 0 and per["Energy transition & power"] > 0, per
    auto = pd.DataFrame({"rd_intensity": [0.3, 0.05, np.nan, 0.2], "rd_growth": [0.4, 0.0, np.nan, 0.1], "rev_accel": [0.2, -0.1, np.nan, 0.05],
                         "gm_change": [0.03, -0.01, np.nan, 0.0], "rev_growth": [0.5, 0.02, 0.3, 0.1], "name": list("ABCD")}, index=list("ABCD"))
    reg = pd.DataFrame(columns=["etf", "manager", "format"])
    import tempfile, shutil
    global ROOT
    old = ROOT; ROOT = Path(tempfile.mkdtemp())
    try:
        df = build(auto, list("ABCD"), reg, "2026-10-29")
    finally:
        shutil.rmtree(ROOT); ROOT = old
    assert df.ticker.iloc[0] == "A" and "C" not in set(df.ticker), df          # C has only 1 signal: not listed
    print("emerging selftest ok")


if __name__ == "__main__":
    selftest()
