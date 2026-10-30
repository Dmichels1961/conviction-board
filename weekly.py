"""Weekly conviction ranking: rank 1 (highest conviction) to 25.

Starting point: the stock must be held by several active-fund managers, and the more overlap the
more it scores (25 pts, scaled by manager count; funds that have not beaten their benchmark count
at half). On top of that base:
  * MANAGER FLOW (20): managers adding / initiating vs trimming / exiting since last week. Weight
    changes are drift-adjusted, so a position that only grew because the stock rallied is not an add.
  * EARNINGS (25): forward EPS growth, beat size, beat consistency, estimate revisions.
Everything else is scored and shown, but weighted lightly (see SIGNALS).

Each run freezes a snapshot under data/weeks/<Monday>/ so the list "resets" weekly, next
week's flow is measured against it, and the dashboard can track performance and signal decay.

    python weekly.py            # this week's ranking (refuses to overwrite an existing week)
    python weekly.py --force    # recompute the current week
    python weekly.py --asof 2026-10-12

Not investment advice. A screen is a starting point for your own diligence.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import screener as S

HERE = Path(__file__).parent

# key, label, group, default weight (points out of 100), description
SIGNALS = [
    ("flow", "Manager flow", "Manager flow", 5,
     "Net adds/initiations minus trims/exits by skilled active ETFs since last week (drift-adjusted, skill-weighted)"),
    ("earn_growth", "Fwd EPS growth", "Earnings", 11, "Consensus EPS growth, next fiscal year (Yahoo, daily)"),
    ("earn_dispersion", "Analyst agreement", "Earnings", 1, "Narrow range of analysts' next-year EPS estimates scores higher (Diether, Malloy & Scherbina 2002)"),
    ("earn_surprise", "Beat size", "Earnings", 10,
     "Average EPS surprise over the last 4 quarters (capped at 50%), blended with a standardized surprise (average beat / its variability)"),
    ("earn_consistency", "Beat consistency", "Earnings", 6, "Share of last 4 quarters that beat consensus"),
    ("earn_revisions", "Estimate revisions", "Earnings", 14, "30-day change in consensus EPS (current and next fiscal year, Yahoo)"),
    ("earn_accel", "Growth acceleration", "Earnings", 5, "Next-year EPS growth minus this year's: is growth speeding up?"),
    ("earn_sales", "Sales strength", "Earnings", 5, "Consensus revenue growth next year, blended with the average revenue beat over 4 quarters"),
    ("earn_reaction", "Market reaction", "Earnings", 4, "Average 2-day move vs SPY after the last 4 reports: did the market believe the beat?"),
    ("earn_guidance", "Guidance", "Earnings", 4, "Raised / maintained / lowered guidance at the last report, adjusted for guide vs consensus"),
    ("breadth", "Best-ideas overlap", "Overlap", 16,
     "Distinct managers holding it as one of a fund's 15 largest positions (most = 100); managers that have not beaten their benchmark count at half"),
    ("active_weight", "Active weight", "Overlap", 4, "Average weight above the benchmark (an active bet, not index beta)"),
    ("momentum", "Momentum 12-1", "Momentum", 5, "12-month price return excluding the latest month"),
    ("profitability", "Gross profitability", "Fundamentals", 2, "Trailing-12-month gross profit / total assets (Novy-Marx 2013; SEC filings)"),
    ("accruals", "Earnings quality", "Fundamentals", 1.5, "Low accruals: net income backed by operating cash flow, scaled by assets (Sloan 1996; SEC filings)"),
    ("issuance", "Net share issuance", "Fundamentals", 1, "Share count shrinking (buybacks) scores higher, dilution lower (Pontiff & Woodgate 2008; SEC filings)"),
    ("crowding", "Short interest", "Fundamentals", 1, "Lower short interest and falling short interest score higher"),
    ("insider", "Insider buying", "Fundamentals", 0.5, "Distinct insiders making open-market purchases in the last 90 days (SEC Form 4, code P); 3 or more = full score"),
    ("asset_growth", "Asset growth", "Fundamentals", 0, "Slower balance-sheet growth scores higher (Cooper, Gulen & Schill 2008; off: weak lately)"),
    ("filing_sim", "Filing changes", "Market signals", 1.5, "Latest 10-K/10-Q worded much like a year earlier scores higher; big rewrites score lower (Cohen, Malloy & Nguyen 2020)"),
    ("put_skew", "Options skew", "Market signals", 1, "Cheap downside protection (low put skew) scores higher (Xing, Zhang & Zhao 2010)"),
    ("customer_mom", "Customer momentum", "Market signals", 0.5, "Main customers' last-month return and capital-spending growth, for suppliers (Cohen & Frazzini 2008)"),
    ("ivol", "Low volatility", "Market signals", 1, "Lower idiosyncratic volatility scores higher (Ang, Hodrick, Xing & Zhang 2006)"),
    ("valuation", "Valuation", "Other", 0, "Forward P/E relative to growth (PEG) and free-cash-flow yield (off)"),
    ("theme", "Structural theme", "Other", 0, "Tailwind score (0-3) from themes.csv (off: shown on the board, not scored)"),
    ("quality", "Quality", "Other", 0, "Retired: replaced by gross profitability, accruals and issuance"),
    ("beta", "Beta", "Other", 0, "Beta 1.0 or below scores 100, falling to 0 at beta 2.0 (off)"),
    ("analyst_drift", "Analyst drift", "Other", 0, "Change in mean price target over 3 months (off: targets follow prices)"),
]
SIG_KEYS = [s[0] for s in SIGNALS]
EARN_COLS = ["fwd_growth", "cur_growth", "rev_growth", "rev_surprise", "reaction", "guidance", "pt_drift", "short_chg", "share_chg", "implied_move", "rev_30d", "gross_margin", "roe", "fcf_margin", "fcf_yield", "fwd_pe", "short_pct", "beta", "insider_buyers", "gross_prof", "accruals", "asset_growth", "net_issuance", "dispersion", "fresh_surprise", "resmom", "filing_sim", "put_skew", "customer_mom", "ivol"]


@dataclass
class Config(S.Config):
    skill_gate: bool = False         # False: every active fund counts toward overlap (skill is a weight)
    unskilled_mult: float = 0.5      # weight of a fund that has not beaten its benchmark
    allow_niche: bool = True         # small/mid caps with little or no benchmark weight qualify if funds overweight them
    coverage_penalty: float = 0.35   # score x (1 - penalty x share of signal weight that lacks real data)
    max_per_bucket: int = 8          # theme cap on the board (names beyond it drop below the board; "Other" is exempt)
    exit_rank: int = 35              # sell discipline: a name already on the board stays while it ranks this high or better
    prev_board: tuple = ()           # the previous board's tickers (set by the caller); empty = no buffer
    prev_rc_board: tuple = ()        # the previous risk-controlled board's tickers
    gaap_surprise_cap: float = 0.05  # GAAP-only reporters: beat size is capped at 5% so they cannot out-beat adjusted reporters
    min_beats: int = 2               # sanity gate only; beat quality is scored
    min_fwd_growth: float = 0.0
    best_n: int = 15                 # a fund's top-N positions count as its best ideas
    flow_min_abs: float = 0.15       # pts of weight: ignore smaller changes
    flow_min_rel: float = 0.08       # ...and changes under 8% of the position
    flow_full: float = 2.0           # a 2-pt weight change = full-strength signal
    flow_partial_certainty: float = 0.5   # entering/leaving a top-10 list is ambiguous
    flow_max_age: int = 10           # flow only from funds whose holdings are at most this many days old (daily files);
                                     # quarterly SEC filings would report months-old trades as this week's buying
    stale_full_days: int = 45        # overlap: holdings up to this age count fully...
    stale_half_days: int = 120       # ...fading linearly to half weight at this age (older still: half)
    sector_mult: float = 0.5         # sector and thematic funds hold their sector by mandate: their votes count half in overlap and flow
    flow_min_kept: float = 0.5       # a fund whose list changed by more than half in a week is treated as a data-source switch, not trades
    weights: dict = field(default_factory=lambda: {k: w for k, _, _, w, _ in SIGNALS})

    def __post_init__(self):
        assert abs(sum(self.weights.values()) - 100) < 1e-6, "signal weights must sum to 100"


# =====================  storage  =====================
def data_dir() -> Path:
    return Path(os.environ.get("DASH_DATA_DIR", HERE / "data"))


def load_gaap_only() -> set:
    f = HERE / "data_web" / "gaap_only.csv"
    return set(pd.read_csv(f).ticker) if f.exists() else set()


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def list_weeks(ddir: Path | None = None) -> list[str]:
    wd = (ddir or data_dir()) / "weeks"
    return sorted(p.name for p in wd.iterdir() if (p / "ranking.csv").exists()) if wd.exists() else []


def previous_week(week: date, ddir: Path | None = None) -> str | None:
    older = [w for w in list_weeks(ddir) if w < week.isoformat()]
    return older[-1] if older else None


# =====================  manager flow  =====================
HOLD_COLS = ["etf", "ticker", "weight", "source", "price", "etf_price"]
FULL_SOURCES = {"csv", "url", "fmp"}   # complete holdings lists; anything else (yahoo_top10, none) is partial


def build_holdings_long(skilled, holdings, stock_px, etf_px) -> pd.DataFrame:
    rows = []
    for etf in skilled["etf"]:
        h = holdings.get(etf)
        if h is None or h.empty:
            continue
        src = h.attrs.get("source", "unknown")
        for t, w in zip(h["ticker"], h["weight"]):
            rows.append(dict(etf=etf, ticker=t, weight=float(w), source=src,
                             price=stock_px.get(t, np.nan), etf_price=etf_px.get(etf, np.nan)))
    return pd.DataFrame(rows, columns=HOLD_COLS)


def age_factor(age_days, cfg: Config) -> float:
    """Overlap weight for holdings of a given age: 1 up to stale_full_days, falling linearly to 0.5 at stale_half_days."""
    if age_days is None or age_days != age_days or age_days <= cfg.stale_full_days: return 1.0
    return max(0.5, 1.0 - 0.5 * (age_days - cfg.stale_full_days) / max(1, cfg.stale_half_days - cfg.stale_full_days))


def compute_flow(skilled: pd.DataFrame, cur: pd.DataFrame, prev: pd.DataFrame, cfg: Config):
    """Compare this week's holdings with last week's. Returns (events, per-stock aggregate)."""
    ir = skilled.set_index("etf")["ir"].clip(lower=0.1)
    ir_w = ir / ir.mean()
    mgr = skilled.set_index("etf")["manager"]
    events = []
    for etf in skilled["etf"]:
        c, p = cur[cur["etf"] == etf], prev[prev["etf"] == etf]
        if c.empty or p.empty:
            continue                                   # no baseline for this ETF
        if len(set(c["ticker"]) & set(p["ticker"])) < cfg.flow_min_kept * min(len(set(c["ticker"])), len(set(p["ticker"]))):
            continue                                   # most of the list changed at once: a source switch, not trading
        partial = (c["source"].iloc[0] not in FULL_SOURCES) or (p["source"].iloc[0] not in FULL_SOURCES)
        pe, ce = p["etf_price"].iloc[0], c["etf_price"].iloc[0]
        cm, pm = c.set_index("ticker"), p.set_index("ticker")
        if pd.notna(pe) and pd.notna(ce) and pe:
            r_f = ce / pe - 1
        else:                                          # no fund price: the fund's return is last week's weights times each stock's move
            both = pm.index.intersection(cm.index)
            rr = (cm["price"].reindex(both) / pm["price"].reindex(both) - 1).replace([np.inf, -np.inf], np.nan)
            ww = pm["weight"].reindex(both)[rr.notna()]
            r_f = float((ww * rr.dropna()).sum() / ww.sum()) if ww.sum() > 0 else 0.0
        for t in sorted(set(cm.index) | set(pm.index)):
            w1 = float(cm["weight"].get(t, 0.0))
            w0 = float(pm["weight"].get(t, 0.0))
            w0_adj = w0
            if t in cm.index and t in pm.index:       # strip out price drift
                ps, cs = pm["price"].get(t, np.nan), cm["price"].get(t, np.nan)
                if pd.notna(ps) and pd.notna(cs) and ps:
                    w0_adj = w0 * (cs / ps) / (1 + r_f)
            delta = w1 - w0_adj
            if w0 == 0 and w1 > 0:
                ev, cert = "initiation", (cfg.flow_partial_certainty if partial else 1.0)
            elif w1 == 0 and w0 > 0:
                ev, cert = "exit", (cfg.flow_partial_certainty if partial else 1.0)
            elif abs(delta) >= max(cfg.flow_min_abs, cfg.flow_min_rel * w0_adj):
                ev, cert = ("add" if delta > 0 else "trim"), 1.0
            else:
                continue
            strength = np.sign(delta) * min(abs(delta) / cfg.flow_full, 1.0)
            events.append(dict(etf=etf, manager=mgr[etf], ticker=t, event=ev, w_prev=w0, w_now=w1,
                               delta=delta, certainty=cert, value=strength * cert * float(ir_w[etf])))
    ev = pd.DataFrame(events, columns=["etf", "manager", "ticker", "event", "w_prev", "w_now",
                                       "delta", "certainty", "value"])
    if ev.empty:
        return ev, pd.DataFrame(columns=["ticker", "flow_raw", "n_adds", "n_removals", "has_init", "flow_notes"])
    pm_ = ev.groupby(["ticker", "manager"])["value"].sum().clip(-1.5, 1.5)
    agg = pd.DataFrame({
        "flow_raw": pm_.groupby("ticker").sum(),
        "n_adds": (pm_ > 0).groupby("ticker").sum(),
        "n_removals": (pm_ < 0).groupby("ticker").sum(),
        "has_init": ev.assign(i=ev["event"].eq("initiation")).groupby("ticker")["i"].any(),
    })
    notes = {t: "; ".join(f"{r.etf} {r.event} {r.delta:+.1f}pt" for r in g.itertuples())
             for t, g in ev.groupby("ticker")}
    agg["flow_notes"] = pd.Series(notes)
    return ev, agg.reset_index()


# =====================  signals & scoring  =====================
RC_LOWVOL = 0.05     # low volatility's share of the risk-controlled score (kept light on purpose)
RC_N, RC_EXIT = 40, 55


def risk_controlled(df, cfg) -> pd.DataFrame:
    """Second board, built to measure skill rather than beta. Same signals and weights as the main board, except:
    every signal is ranked WITHIN its sector (so no sector can dominate by being hot), momentum is residual momentum (12-1 month return
    left after removing the market's move, per unit of risk; Blitz, Huij & Martens 2011), overlap is active-weight best ideas, and
    low volatility gets 5%. Backtest 2021-26, beta-adjusted: +0.71%/month vs the pool, t 2.15, average beta 1.12."""
    eff = df.attrs.get("eff_weights", {k: w for k, _, _, w, _ in SIGNALS})
    s = {k: df[f"sig_{k}"] for k in SIG_KEYS if eff.get(k, 0) > 0}
    if "breadth" in s and "active_best" in df.columns: s["breadth"] = _pr(df["active_best"])
    if "momentum" in s and "resmom" in df.columns and df["resmom"].notna().any(): s["momentum"] = _pr(df["resmom"])
    w = {k: eff[k] * (1 - RC_LOWVOL) / sum(eff[x] for x in s) for k in s}
    lv = df["ivol"] if "ivol" in df.columns else pd.Series(np.nan, index=df.index)
    s["_lowvol"] = _pr(lv, higher=False); w["_lowvol"] = RC_LOWVOL
    sec = df["sector"].fillna("Unknown") if "sector" in df.columns else pd.Series("Unknown", index=df.index)
    score = sum(w[k] * pd.Series(v, index=df.index).groupby(sec).rank(pct=True).fillna(0.5) for k, v in s.items()) * 100
    df["rc_score"] = score.round(2)
    order = score.sort_values(ascending=False).index; srank = pd.Series(np.arange(1, len(order) + 1), index=order)
    prev = set(getattr(cfg, "prev_rc_board", ()) or ())
    keep = [i for i in order if df.loc[i, "ticker"] in prev and srank[i] <= RC_EXIT][:RC_N]
    keep += [i for i in order if i not in set(keep)][:RC_N - len(keep)]
    rest = [i for i in order if i not in set(keep)]
    rk = pd.Series(np.arange(1, len(order) + 1), index=sorted(keep, key=lambda i: -score[i]) + rest)
    df["rc_rank"] = rk.reindex(df.index).astype(int)
    return df


def best_ideas(skilled, holdings, cfg) -> pd.DataFrame:
    """Best-ideas overlap: distinct managers holding the stock among a fund's top `best_n` positions by weight, skill-weighted
    (a manager that has not beaten its benchmark counts at cfg.unskilled_mult). One manager counts once however many of its funds agree.
    Backtest 2024-26: top-15 overlap beat counting every holder about two to four times over (Cohen, Polk & Silli 2010)."""
    by_mgr = {}
    for _, f in skilled.iterrows():
        h = holdings.get(f["etf"])
        if h is None or h.empty: continue
        top = h.sort_values("weight", ascending=False)["ticker"].head(cfg.best_n)
        w = float(f.get("ir", 1.0))
        for t in top: by_mgr.setdefault(t, {})[f["manager"]] = max(by_mgr.get(t, {}).get(f["manager"], 0.0), w)
    # active-weight best ideas (for the risk-controlled board): a fund's 15 largest weights ABOVE the peer consensus weight
    # (average weight across all funds, zeros included), so mega caps every fund holds near index weight do not count automatically
    W = pd.DataFrame({e: h.set_index("ticker")["weight"] for e, h in holdings.items() if h is not None and not h.empty}).fillna(0.0)
    act_by = {}
    if not W.empty:
        cons = W.mean(axis=1); mgr = dict(zip(skilled["etf"], skilled["manager"])); irw = dict(zip(skilled["etf"], skilled.get("ir", 1.0)))
        for e in W.columns:
            a = (W[e] - cons)[W[e] > 0].sort_values(ascending=False).head(cfg.best_n)
            for t in a.index: act_by.setdefault(t, {})[mgr.get(e, e)] = max(act_by.get(t, {}).get(mgr.get(e, e), 0.0), float(irw.get(e, 1.0)))
    tick = sorted(set(by_mgr) | set(act_by))
    return pd.DataFrame([dict(ticker=t, best_raw=sum(by_mgr.get(t, {}).values()), n_best=len(by_mgr.get(t, {})),
                              active_best=sum(act_by.get(t, {}).values())) for t in tick], columns=["ticker", "best_raw", "n_best", "active_best"])


def _pr(s, higher=True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(pct=True, method="average") * 100
    if not higher:
        r = 100 - r
    return r.fillna(50.0)


def earnings_table(tickers, provider, cfg: Config) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            e = provider.earnings(t)
        except Exception as ex:
            print(f"  ! earnings fetch failed for {t}: {ex}")
            e = {}
        sur = list(e.get("surprises") or [])[-4:]
        capped = [max(-cfg.surprise_cap, min(cfg.surprise_cap, s)) for s in sur]
        row = dict(ticker=t, n_q=len(sur), beats=sum(1 for s in sur if s > 0),
                   avg_surprise=float(np.mean(capped)) if capped else np.nan,
                   sue=(float(np.mean(capped) / max(np.std(capped), 0.02)) if len(capped) >= 3 else np.nan),
                   name=e.get("name"), sector=e.get("sector"))
        for c in EARN_COLS:
            row[c] = e.get(c)
        if row["fwd_growth"] is None:
            row["fwd_growth"] = e.get("cur_growth")
            row["cur_growth"] = None      # growth fell back to this-year; no acceleration reading
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in EARN_COLS + ["avg_surprise", "sue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def momentum_12_1(hist: pd.DataFrame) -> pd.Series:
    out = {}
    for t in hist.columns:
        s = hist[t].dropna()
        if len(s) >= 13:
            out[t] = s.iloc[-2] / s.iloc[-13] - 1
    return pd.Series(out, dtype=float)


def score_signals(df: pd.DataFrame, cfg: Config, flow_available: bool) -> pd.DataFrame:
    df = df.copy()
    sig = pd.DataFrame(index=df.index)
    sig["flow"] = 50 + 50 * np.tanh(df["flow_raw"].fillna(0)) if flow_available else 50.0
    sig["earn_growth"] = _pr(df["fwd_growth"])
    sig["earn_surprise"] = pd.concat([_pr(df["avg_surprise"]), _pr(df["sue"])], axis=1).mean(axis=1)   # four quarters, equal (audit: beat the fresh-weighted version on IC)
    sig["earn_accel"] = _pr(df["fwd_growth"] - df["cur_growth"])
    sig["earn_consistency"] = _pr(df["beats"] / df["n_q"].clip(lower=1))
    sig["earn_revisions"] = _pr(df["rev_30d"])
    sig["breadth"] = (100 * df["best_raw"] / df["best_raw"].max()) if "best_raw" in df.columns and df["best_raw"].max() > 0 else 100 * df["overlap_raw"] / df["overlap_raw"].max()
    sig["active_weight"] = _pr(df["avg_active_weight"])
    sig["momentum"] = _pr(df["mom_12_1"])
    sig["earn_sales"] = pd.concat([_pr(df["rev_growth"]), _pr(df["rev_surprise"])], axis=1).mean(axis=1)
    sig["earn_reaction"] = _pr(df["reaction"])
    sig["earn_guidance"] = (100 * df["guidance"]).clip(0, 100).fillna(50.0)
    sig["analyst_drift"] = _pr(df["pt_drift"])
    sig["quality"] = pd.concat([_pr(df["gross_margin"]), _pr(df["roe"]), _pr(df["fcf_margin"]), _pr(-df["share_chg"])], axis=1).mean(axis=1)
    peg = (df["fwd_pe"] / (df["fwd_growth"] * 100)).where((df["fwd_growth"] > 0) & (df["fwd_pe"] > 0))
    sig["valuation"] = pd.concat([_pr(peg, higher=False), _pr(df["fcf_yield"])], axis=1).mean(axis=1)
    sig["theme"] = (100 * df["tailwind"] / 3).fillna(0)
    sig["crowding"] = pd.concat([_pr(df["short_pct"], higher=False), _pr(df["short_chg"], higher=False)], axis=1).mean(axis=1)
    sig["insider"] = (50 + 50 * np.minimum(pd.to_numeric(df["insider_buyers"], errors="coerce"), 3) / 3).fillna(50.0)
    sig["beta"] = (np.clip((2 - df["beta"]) / 1.0, 0, 1) * 100).fillna(50)
    sig["earn_dispersion"] = _pr(df["dispersion"], higher=False)
    sig["filing_sim"] = _pr(df["filing_sim"])
    sig["put_skew"] = _pr(df["put_skew"], higher=False)
    sig["customer_mom"] = _pr(df["customer_mom"])
    sig["ivol"] = _pr(df["ivol"], higher=False)
    sig["profitability"] = _pr(df["gross_prof"])
    sig["accruals"] = _pr(df["accruals"], higher=False)
    sig["asset_growth"] = _pr(df["asset_growth"], higher=False)
    sig["issuance"] = _pr(df["net_issuance"], higher=False)
    inactive = set()
    if not flow_available:
        inactive.add("flow")
    if df["rev_30d"].notna().sum() == 0:
        inactive.add("earn_revisions")
    if (df["fwd_growth"] - df["cur_growth"]).notna().sum() == 0:
        inactive.add("earn_accel")
    for k, cols in {"earn_sales": ["rev_growth", "rev_surprise"], "earn_reaction": ["reaction"], "earn_guidance": ["guidance"],
                    "analyst_drift": ["pt_drift"], "insider": ["insider_buyers"], "profitability": ["gross_prof"], "accruals": ["accruals"],
                    "asset_growth": ["asset_growth"], "issuance": ["net_issuance"], "crowding": ["short_pct", "short_chg"],
                    "earn_dispersion": ["dispersion"], "filing_sim": ["filing_sim"], "put_skew": ["put_skew"], "customer_mom": ["customer_mom"], "ivol": ["ivol"]}.items():
        if df[cols].notna().sum().sum() == 0:
            inactive.add(k)
    live = sum(w for k, w in cfg.weights.items() if k not in inactive)
    eff = {k: (0.0 if k in inactive else cfg.weights[k] * 100 / live) for k in SIG_KEYS}
    df.attrs["inactive"] = sorted(inactive)
    df.attrs["eff_weights"] = eff
    for k in SIG_KEYS:
        df[f"sig_{k}"] = sig[k].astype(float)
        df[f"pts_{k}"] = eff[k] * df[f"sig_{k}"] / 100
    df["score"] = df[[f"pts_{k}" for k in SIG_KEYS]].sum(axis=1)
    earn_keys = [k for k in SIG_KEYS if k.startswith("earn_")]
    df["earn_score"] = df[[f"pts_{k}" for k in earn_keys]].sum(axis=1) / sum(eff[k] for k in earn_keys) * 100
    return df


def _drivers(r, cfg: Config, n=3) -> str:
    labels = {k: lbl for k, lbl, *_ in SIGNALS}
    contrib = {k: (r[f"sig_{k}"] - 50) * cfg.weights[k] for k in SIG_KEYS}
    top = sorted(contrib, key=contrib.get, reverse=True)[:n]
    return ", ".join(labels[k] for k in top if contrib[k] > 0)


def _thesis(r, flow_available: bool) -> str:
    parts = []
    if flow_available:
        parts.append(f"{int(r['n_adds'])} manager add(s), {int(r['n_removals'])} removal(s)"
                     if (r["n_adds"] or r["n_removals"]) else "no manager changes")
    parts.append(f"held by {int(r['n_managers'])} managers")
    parts.append(f"fwd EPS {r['fwd_growth']*100:+.0f}%, beat {int(r['beats'])}/{int(r['n_q'])} "
                 f"(avg {r['avg_surprise']*100:+.1f}%)")
    if pd.notna(r["rev_30d"]):
        parts.append(f"30d revisions {r['rev_30d']*100:+.1f}%")
    if isinstance(r.get("theme"), str) and r["theme"]:
        parts.append(f"theme: {r['theme']}")
    return " | ".join(parts)


# =====================  orchestration  =====================
def load_themes(path: Path | None = None) -> pd.DataFrame:
    p = path or HERE / "themes.csv"
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "theme", "tailwind"])
    t = pd.read_csv(p)
    t["ticker"] = t["ticker"].str.strip().str.upper()
    return t


def run_week(provider, universe, cfg: Config, themes: pd.DataFrame, prev_hold, log=print) -> dict:
    log("[1/6] Skill: comparing active ETFs with their benchmarks...")
    all_funds = sorted(set(universe["etf"]) | set(universe["benchmark"]))
    if hasattr(provider, "skill_table"):          # providers that only have trailing returns
        skill = provider.skill_table(universe, cfg)
    else:
        skill = S.skill_table(universe, provider.prices(all_funds), cfg)
    # Skill is a weight, not a gate: every active fund counts toward overlap, and funds that have not beaten
    # their benchmark count at cfg.unskilled_mult. Set cfg.skill_gate=True to use skilled funds only.
    skilled = skill.copy()
    skilled["ir"] = np.where(skilled["skilled"], 1.0, cfg.unskilled_mult)
    if cfg.skill_gate:
        skilled = skilled[skilled["skilled"]]
    log(f"      {int(skill['skilled'].sum())} of {len(skill)} ETFs beat their benchmark; "
        f"{len(skilled)} count toward overlap")

    log("[2/6] Holdings...")
    holdings = {e: provider.holdings(e) for e in skilled["etf"]}
    benches = {b: provider.holdings(b) for b in skilled["benchmark"].unique()}
    partial = [b for b, h in benches.items() if not S._bench_lookup(h)[1]]
    if partial:
        log(f"      WARNING: partial benchmark holdings for {partial}; add full CSVs to holdings/ to verify "
            f"benchmark weight and active weight")
    held = sorted({t for h in holdings.values() for t in h["ticker"]})
    px = provider.last_prices(held + list(skilled["etf"]) + ["SPY"])
    cur = build_holdings_long(skilled, holdings, px, px)

    ages = {e: provider.holdings_age(e) for e in skilled["etf"]} if hasattr(provider, "holdings_age") else {}
    flow_funds = skilled[[ages.get(e) is None or ages[e] <= cfg.flow_max_age for e in skilled["etf"]]] if ages else skilled
    kind = dict(zip(universe["etf"], universe["kind"])) if "kind" in universe.columns else {}
    skilled_ov = skilled.copy()
    skilled_ov["ir"] = [float(r) * age_factor(ages.get(e), cfg) * (cfg.sector_mult if kind.get(e) in ("sector", "thematic") else 1.0)
                        for e, r in zip(skilled_ov["etf"], skilled_ov["ir"])]
    flow_funds = skilled_ov[skilled_ov["etf"].isin(flow_funds["etf"])]
    n_sec = sum(1 for e in skilled["etf"] if kind.get(e) in ("sector", "thematic"))
    if n_sec: log(f"      {n_sec} sector/thematic funds count at {cfg.sector_mult:g}x in overlap and flow")
    if ages:
        old_n = sum(1 for e in skilled["etf"] if (ages.get(e) or 0) > cfg.stale_full_days)
        log(f"      flow uses {len(flow_funds)} funds with daily holdings; {old_n} funds older than {cfg.stale_full_days} days count less in overlap")

    log("[3/6] Manager flow vs last week's snapshot...")
    flow_available = prev_hold is not None and len(prev_hold) > 0
    if flow_available:
        events, fagg = compute_flow(flow_funds, cur, prev_hold, cfg)
        flow_available = (len(events) > 0) or (prev_hold["etf"].isin(flow_funds["etf"]).any())
        log(f"      {len(events)} position changes ({(events['value'] > 0).sum() if len(events) else 0} adds, "
            f"{(events['value'] < 0).sum() if len(events) else 0} removals)")
    else:
        events, fagg = pd.DataFrame(), pd.DataFrame(columns=["ticker", "flow_raw", "n_adds", "n_removals", "has_init", "flow_notes"])
        log("      no previous snapshot yet: flow is neutral this week (builds from next week)")

    ov = S.overlap_table(skilled_ov, holdings, benches)
    if ov.empty:
        return dict(skill=skill, cur=cur, events=events, all=pd.DataFrame(), ranking=pd.DataFrame(),
                    flow_available=flow_available, partial=partial, spy=px.get("SPY"))
    ov = ov.merge(best_ideas(skilled_ov, holdings, cfg), on="ticker", how="left")
    ov["best_raw"] = ov["best_raw"].fillna(0.0); ov["n_best"] = ov["n_best"].fillna(0).astype(int); ov["active_best"] = ov["active_best"].fillna(0.0)
    ov = ov.merge(fagg, on="ticker", how="left")
    ov["flow_raw"] = ov["flow_raw"].fillna(0.0)
    ov["n_adds"] = ov["n_adds"].fillna(0).astype(int)
    ov["n_removals"] = ov["n_removals"].fillna(0).astype(int)
    ov["has_init"] = ov["has_init"].fillna(False).astype(bool)
    ov["flow_notes"] = ov["flow_notes"].fillna("")

    # starting point: held by several active-fund managers. Then a real index name held above benchmark weight.
    keep = ov["n_managers"] >= cfg.min_managers
    bw_ok, aw_ok = ov["bench_weight"] >= cfg.min_bench_weight, ov["avg_active_weight"] > cfg.min_active_weight
    if cfg.allow_niche:
        keep &= aw_ok | ~ov["bench_verified"]
    elif cfg.strict_benchmark:
        keep &= bw_ok & aw_ok
    else:
        keep &= (bw_ok | ~ov["bench_verified"]) & (aw_ok | ~ov["bench_verified"])
    cand = ov[keep].copy()
    log(f"[4/6] {len(ov)} stocks held; {len(cand)} pass manager/benchmark gates. Earnings data...")

    earn = earnings_table(cand["ticker"].tolist(), provider, cfg) if len(cand) else pd.DataFrame()
    df = cand.merge(earn, on="ticker", how="inner")
    # Missing inputs lower the data-coverage score; they do not disqualify. Sanity gates apply only where data exists.
    df = df[((df["n_q"] > 0) | df["fwd_growth"].notna())
            & ((df["n_q"] < 2) | (df["beats"] >= cfg.min_beats))
            & (df["fwd_growth"].isna() | (df["fwd_growth"] >= cfg.min_fwd_growth))].copy()
    if cfg.max_beta is not None:
        df = df[df["beta"].isna() | (df["beta"] <= cfg.max_beta)]
    if df.empty:
        return dict(skill=skill, cur=cur, events=events, all=df, ranking=df, flow_available=flow_available,
                    partial=partial, spy=px.get("SPY"))

    log(f"[5/6] Momentum, themes, scoring {len(df)} names...")
    if hasattr(provider, "momentum"):
        mom = provider.momentum(df["ticker"].tolist())
    else:
        mom = momentum_12_1(provider.stock_history(df["ticker"].tolist()))
    df["mom_12_1"] = df["ticker"].map(mom)
    th = themes.drop_duplicates("ticker").set_index("ticker")
    df["theme"] = df["ticker"].map(th["theme"]).fillna("")
    df["tailwind"] = df["ticker"].map(th["tailwind"])
    df["price"] = df["ticker"].map(px)
    gaap = load_gaap_only()
    df["gaap_only"] = df["ticker"].isin(gaap)
    for c in ("avg_surprise", "sue"):
        if c in df.columns:
            df.loc[df["gaap_only"], c] = df.loc[df["gaap_only"], c].clip(upper=cfg.gaap_surprise_cap)
    df = score_signals(df, cfg, flow_available)
    df = risk_controlled(df, cfg)
    import diagnostics as _D                        # lazy: diagnostics imports this module
    df["data_coverage"] = _D.coverage(df, cfg, flow_available).round(0)
    df["score_raw"] = df["score"]
    df["score"] = df["score_raw"] * (1 - cfg.coverage_penalty * (1 - df["data_coverage"] / 100))
    df["bucket"] = df["ticker"].map(_D.BUCKET).fillna("Other")
    df = df.sort_values(["score", "sig_flow", "fwd_growth"], ascending=False).reset_index(drop=True)
    df["score_rank"] = np.arange(1, len(df) + 1)   # rank by score alone, before the buffer and theme cap
    kept, n_in = [], {}
    fits = lambda b: b == "Other" or n_in.get(b, 0) < cfg.max_per_bucket
    # 1) incumbents keep their seat while they rank exit_rank or better (enter at top_n, leave below exit_rank: less churn)
    for i, r in df.iterrows():
        if r["ticker"] in set(cfg.prev_board) and r["score_rank"] <= cfg.exit_rank and len(kept) < cfg.top_n and fits(r["bucket"]):
            kept.append(i); n_in[r["bucket"]] = n_in.get(r["bucket"], 0) + 1
    # 2) fill the remaining seats in score order, theme cap applied
    for i, r in df.iterrows():
        if i in kept: continue
        if len(kept) < cfg.top_n and fits(r["bucket"]):
            kept.append(i); n_in[r["bucket"]] = n_in.get(r["bucket"], 0) + 1
    kept = sorted(kept, key=lambda i: -df.loc[i, "score"])
    pushed = [i for i in df.index if i not in set(kept)]
    df["held_by_buffer"] = False
    df.loc[[i for i in kept if df.loc[i, "score_rank"] > cfg.top_n], "held_by_buffer"] = True
    df["bucket_capped"] = False
    df.loc[[i for i in pushed if df.loc[i, "score"] >= df.loc[kept[-1], "score"] and not df.loc[i, "score_rank"] > cfg.top_n] if kept else [], "bucket_capped"] = True
    df = df.loc[kept + pushed].reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["drivers"] = df.apply(lambda r: _drivers(r, cfg), axis=1)
    df["thesis"] = df.apply(lambda r: _thesis(r, flow_available), axis=1)
    log(f"[6/6] Ranked {len(df)} names; top {min(cfg.top_n, len(df))} go on the board")
    return dict(skill=skill, cur=cur, events=events, all=df, ranking=df.head(cfg.top_n),
                flow_available=flow_available, partial=partial, spy=px.get("SPY"))


def save_week(week: date, res: dict, cfg: Config, ddir: Path | None = None) -> Path:
    wdir = (ddir or data_dir()) / "weeks" / week.isoformat()
    wdir.mkdir(parents=True, exist_ok=True)
    res["skill"].to_csv(wdir / "skill.csv", index=False)
    res["cur"].to_csv(wdir / "holdings.csv", index=False)
    res["events"].to_csv(wdir / "flow_events.csv", index=False)
    res["all"].to_csv(wdir / "ranking_all.csv", index=False)
    res["ranking"].to_csv(wdir / "ranking.csv", index=False)
    meta = dict(week=week.isoformat(), created=datetime.now().isoformat(timespec="seconds"),
                spy_price=res["spy"], flow_available=bool(res["flow_available"]),
                benchmarks_partial=res["partial"], n_ranked=int(len(res["ranking"])),
                top_n=cfg.top_n, weights=cfg.weights, signals=[dict(key=k, label=l, group=g, desc=d) for k, l, g, _, d in SIGNALS])
    (wdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return wdir


def generate(provider, universe, cfg: Config, themes: pd.DataFrame, asof: date,
             force: bool = False, ddir: Path | None = None, log=print) -> dict:
    ddir = ddir or data_dir()
    week = week_start(asof)
    if (ddir / "weeks" / week.isoformat() / "ranking.csv").exists() and not force:
        raise FileExistsError(f"Week {week} already exists; the board is frozen for the week. Use --force to recompute.")
    prev = previous_week(week, ddir)
    prev_hold = pd.read_csv(ddir / "weeks" / prev / "holdings.csv") if prev else None
    res = run_week(provider, universe, cfg, themes, prev_hold, log=log)
    res["week"] = week
    if len(res["ranking"]):
        save_week(week, res, cfg, ddir)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", help="date YYYY-MM-DD (default today)")
    ap.add_argument("--force", action="store_true", help="recompute the week even if it exists")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--strict-benchmark", action="store_true")
    a = ap.parse_args()
    cfg = replace(Config(), top_n=a.top, strict_benchmark=a.strict_benchmark)
    asof = date.fromisoformat(a.asof) if a.asof else date.today()
    from data_sources import LiveProvider
    res = generate(LiveProvider(), S.load_universe(), cfg, load_themes(), asof, force=a.force)
    r = res["ranking"]
    if r.empty:
        print("\nNo stocks passed the gates. Check out/ and loosen thresholds, or add ETFs / full holdings CSVs.")
        return
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print(f"\nWeek of {res['week']}: top {len(r)}")
    print(r[["rank", "ticker", "score", "sig_flow", "earn_score", "n_managers", "fwd_growth", "beats", "theme"]]
          .round(2).to_string(index=False))
    if len(r) < cfg.top_n:
        print(f"\nOnly {len(r)} names passed all gates (target {cfg.top_n}); the board is not padded.")


if __name__ == "__main__":
    main()


def flow_selftest():
    """A rally with no trading must not count as buying; a real trim must."""
    cfg = Config(); sk = pd.DataFrame({"etf": ["F"], "manager": ["M"], "ir": [1.0]})
    prev = pd.DataFrame({"etf": "F", "ticker": ["A", "B", "C"], "weight": [40.0, 30, 30], "source": "url", "price": [100.0, 100, 100], "etf_price": np.nan})
    a = 40 * 1.2 / (40 * 1.2 + 60) * 100; r = (100 - a) / 2
    cur = pd.DataFrame({"etf": "F", "ticker": ["A", "B", "C"], "weight": [a, r, r], "source": "url", "price": [120.0, 100, 100], "etf_price": np.nan})
    ev, _ = compute_flow(sk, cur, prev, cfg); assert len(ev) == 0, ev
    cur.loc[1, "weight"] -= 3; cur.loc[0, "weight"] += 3
    ev, _ = compute_flow(sk, cur, prev, cfg); assert ("B", "trim") in set(zip(ev.ticker, ev.event)), ev
    assert age_factor(30, cfg) == 1.0 and age_factor(120, cfg) == 0.5 and 0.5 < age_factor(80, cfg) < 1.0
    print("weekly flow selftest ok")
