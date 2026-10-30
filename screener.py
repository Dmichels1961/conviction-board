"""Conviction screener: what do skilled active managers own, and are earnings backing it up?

Pipeline
  1. SKILL      Rank the active ETFs in etf_universe.csv by excess return and
                information ratio vs their benchmark; keep the skilled ones.
  2. OVERLAP    Pull holdings of the skilled ETFs; count how many distinct
                managers hold each stock. Compare each position with its weight
                in the benchmark: must be a meaningful index name (liquid,
                big presence) AND held above benchmark weight (a real active
                bet, not just owning the index).
  3. EARNINGS   For the survivors: forward EPS growth, beat count over the last
                4 quarters, and average size of the beats.
  4. RANK       Composite score -> top N -> picks.csv for the dashboard.

Usage
  python screener.py            # live data
  python screener.py --top 25 --no-write
Edit the CONFIG block (or pass flags) to change thresholds.
Not investment advice; a screen is a starting point for your own diligence.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


# =====================  CONFIG  =====================
@dataclass
class Config:
    top_n: int = 25

    # --- 1. skill ---
    min_months: int = 12             # need at least this much monthly history
    skilled_top_fraction: float = 0.5  # keep the top X of ETFs (by info ratio) with positive excess return

    # --- 2. overlap / not-a-beta-play ---
    min_managers: int = 2            # distinct fund managers that must hold it
    min_bench_weight: float = 0.20   # % - must be a real index constituent of this size
    min_active_weight: float = 0.0   # % - avg (ETF weight - benchmark weight) must exceed this
    strict_benchmark: bool = False   # True: drop names whose benchmark weight can't be verified
    max_beta: float | None = None    # e.g. 1.5 to exclude high-beta names; None = off

    # --- 3. earnings ---
    min_beats: int = 3               # beats in the last 4 reported quarters
    min_fwd_growth: float = 0.10     # next-fiscal-year consensus EPS growth (0.10 = 10%)
    surprise_cap: float = 0.50       # cap each quarter's surprise at 50% (tiny-estimate outliers)

    # --- 4. composite weights (sum to 1) ---
    w_overlap: float = 0.35
    w_growth: float = 0.25
    w_consistency: float = 0.20
    w_surprise: float = 0.20


# =====================  1. SKILL  =====================
def skill_table(universe: pd.DataFrame, prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-ETF excess return, tracking error, info ratio, and a `skilled` flag."""
    rets = prices.pct_change()
    rows = []
    for _, u in universe.iterrows():
        etf, bench = u["etf"], u["benchmark"]
        row = dict(etf=etf, manager=u["manager"], benchmark=bench, months=0,
                   excess_ann=np.nan, excess_1y=np.nan, te=np.nan, ir=np.nan)
        if etf in rets.columns and bench in rets.columns:
            pair = rets[[etf, bench]].dropna().tail(36)
            n = len(pair)
            row["months"] = n
            if n >= cfg.min_months:
                cum_e = (1 + pair[etf]).prod()
                cum_b = (1 + pair[bench]).prod()
                row["excess_ann"] = (cum_e / cum_b) ** (12 / n) - 1
                last = pair.tail(12)
                row["excess_1y"] = (1 + last[etf]).prod() / (1 + last[bench]).prod() - 1
                active = pair[etf] - pair[bench]
                row["te"] = active.std() * math.sqrt(12)
                row["ir"] = row["excess_ann"] / row["te"] if row["te"] and row["te"] > 0 else np.nan
        rows.append(row)
    sk = pd.DataFrame(rows)
    eligible = sk[(sk["months"] >= cfg.min_months) & (sk["excess_ann"] > 0) & sk["ir"].notna()]
    n_keep = max(1, math.ceil(len(eligible) * cfg.skilled_top_fraction)) if len(eligible) else 0
    keep = set(eligible.sort_values("ir", ascending=False).head(n_keep)["etf"])
    sk["skilled"] = sk["etf"].isin(keep)
    return sk.sort_values("ir", ascending=False, na_position="last").reset_index(drop=True)


# =====================  2. OVERLAP  =====================
def _bench_lookup(bench_holdings: pd.DataFrame):
    """(series ticker->weight%, is_full) for a benchmark's holdings."""
    s = bench_holdings.set_index("ticker")["weight"] if len(bench_holdings) else pd.Series(dtype=float)
    return s, len(s) >= 200 or (len(s) >= 25 and float(s.sum()) >= 95)   # full = the whole index: 200+ names, or a small index (SOXX has 30) adding to ~100%


def overlap_table(skilled: pd.DataFrame, holdings: dict, bench_holdings: dict) -> pd.DataFrame:
    """One row per stock: who holds it, how many managers, active weight vs benchmark."""
    bench = {b: _bench_lookup(h) for b, h in bench_holdings.items()}
    rows = []
    for _, e in skilled.iterrows():
        h = holdings.get(e["etf"])
        if h is None or h.empty:
            continue
        series, full = bench.get(e["benchmark"], (pd.Series(dtype=float), False))
        for tkr, w in zip(h["ticker"], h["weight"]):
            if tkr in series.index:
                b = float(series[tkr])
            elif full:
                b = 0.0           # full benchmark list and not in it -> not in the benchmark
            else:
                b = np.nan        # benchmark only partially known -> can't tell
            rows.append(dict(etf=e["etf"], manager=e["manager"], benchmark=e["benchmark"],
                             ticker=tkr, weight=float(w), bench_w=b,
                             active_w=(w - b) if not np.isnan(b) else np.nan, ir=e["ir"]))
    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame()
    g = long.groupby("ticker")
    agg = pd.DataFrame({
        "n_etfs": g["etf"].nunique(),
        "n_managers": g["manager"].nunique(),
        "avg_weight": g["weight"].mean(),
        "avg_active_weight": g["active_w"].mean(),
        "bench_weight": g["bench_w"].mean(),
        "holders": g["etf"].apply(lambda s: ", ".join(sorted(set(s)))),
    })
    # skill-weighted manager count: each manager counts once, scaled by its best holder's info ratio
    best = long.groupby(["ticker", "manager"])["ir"].max().clip(lower=0.1)
    agg["overlap_raw"] = best.groupby("ticker").sum()
    ow = long[long["active_w"] > 0].groupby("ticker")["manager"].nunique()
    agg["n_overweight"] = ow.reindex(agg.index).fillna(0).astype(int)   # managers holding it above benchmark weight
    agg["bench_verified"] = agg["bench_weight"].notna()
    return agg.reset_index()


def filter_overlap(ov: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if ov.empty:
        return ov
    keep = ov["n_managers"] >= cfg.min_managers
    bw_ok = ov["bench_weight"] >= cfg.min_bench_weight
    aw_ok = ov["avg_active_weight"] > cfg.min_active_weight
    if cfg.strict_benchmark:
        keep &= bw_ok & aw_ok
    else:  # unverifiable benchmark weights pass but stay flagged
        keep &= (bw_ok | ~ov["bench_verified"]) & (aw_ok | ~ov["bench_verified"])
    return ov[keep].copy()


# =====================  3. EARNINGS  =====================
def earnings_table(tickers, provider, cfg: Config) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            e = provider.earnings(t)
        except Exception as ex:
            print(f"  ! earnings fetch failed for {t}: {ex}")
            e = {}
        sur = [s for s in (e.get("surprises") or [])][-4:]
        capped = [max(-cfg.surprise_cap, min(cfg.surprise_cap, s)) for s in sur]
        fwd = e.get("fwd_growth")
        if fwd is None:
            fwd = e.get("cur_growth")
        rows.append(dict(
            ticker=t,
            fwd_growth=fwd,
            n_q=len(sur),
            beats=sum(1 for s in sur if s > 0),
            avg_surprise=float(np.mean(capped)) if capped else np.nan,
            beta=e.get("beta"),
        ))
    return pd.DataFrame(rows)


# =====================  4. RANK  =====================
def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average").fillna(0.0)


def rank_candidates(ov: pd.DataFrame, earn: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = ov.merge(earn, on="ticker", how="inner")
    df = df[(df["beats"] >= cfg.min_beats) & (df["fwd_growth"].notna())
            & (df["fwd_growth"] >= cfg.min_fwd_growth)].copy()
    if cfg.max_beta is not None:
        df = df[df["beta"].isna() | (df["beta"] <= cfg.max_beta)]
    if df.empty:
        return df
    df["consistency"] = df["beats"] / df["n_q"].clip(lower=1)
    df["score"] = 100 * (
        cfg.w_overlap * _pct_rank(df["overlap_raw"])
        + cfg.w_growth * _pct_rank(df["fwd_growth"])
        + cfg.w_consistency * _pct_rank(df["consistency"])
        + cfg.w_surprise * _pct_rank(df["avg_surprise"])
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def to_picks(ranked: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    top = ranked.head(cfg.top_n).copy()
    if top.empty:
        return pd.DataFrame(columns=["ticker", "conviction", "weight", "entry_price", "target_price", "thesis"])
    pr = top["score"].rank(pct=True, method="first")
    top["conviction"] = np.ceil(pr * 5).astype(int).clip(1, 5)

    def thesis(r):
        act = "" if pd.isna(r["avg_active_weight"]) else f", avg {r['avg_active_weight']:+.1f}pt vs benchmark"
        return (f"Held by {int(r['n_managers'])} managers ({r['holders']}){act}; "
                f"fwd EPS growth {r['fwd_growth']*100:+.0f}%; beat {int(r['beats'])}/{int(r['n_q'])} qtrs, "
                f"avg surprise {r['avg_surprise']*100:+.1f}%")

    return pd.DataFrame({
        "ticker": top["ticker"],
        "conviction": top["conviction"],
        "weight": top["score"].round(1),
        "entry_price": np.nan,
        "target_price": np.nan,
        "thesis": top.apply(thesis, axis=1),
    })


# =====================  ORCHESTRATION  =====================
def run(provider, universe: pd.DataFrame, cfg: Config, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    all_tickers = sorted(set(universe["etf"]) | set(universe["benchmark"]))
    log(f"[1/4] Prices for {len(all_tickers)} funds...")
    prices = provider.prices(all_tickers)
    skill = skill_table(universe, prices, cfg)
    skilled = skill[skill["skilled"]]
    log(f"      {len(skilled)} skilled of {len(skill)} ETFs "
        f"({(skill['months'] < cfg.min_months).sum()} lacked {cfg.min_months} months of history)")

    log("[2/4] Holdings + overlap...")
    holdings = {etf: provider.holdings(etf) for etf in skilled["etf"]}
    benches = {b: provider.holdings(b) for b in skilled["benchmark"].unique()}
    partial = [b for b, h in benches.items() if len(h) < 50]
    if partial:
        log(f"      WARNING: benchmark holdings incomplete for {partial}. Download the full holdings CSV into "
            f"holdings/<TICKER>.csv so 'big presence in benchmark' and active weight can be verified.")
    ov = overlap_table(skilled, holdings, benches)
    cand = filter_overlap(ov, cfg)
    log(f"      {len(ov)} distinct stocks held; {len(cand)} pass overlap/benchmark filters")

    log(f"[3/4] Earnings data for {len(cand)} candidates...")
    earn = earnings_table(cand["ticker"].tolist(), provider, cfg) if len(cand) else pd.DataFrame()

    log("[4/4] Ranking...")
    ranked = rank_candidates(cand, earn, cfg) if len(cand) and len(earn) else pd.DataFrame()
    picks = to_picks(ranked, cfg)
    return dict(skill=skill, overlap=ov, candidates=cand, earnings=earn, ranked=ranked, picks=picks)


def load_universe(path: Path | None = None) -> pd.DataFrame:
    u = pd.read_csv(path or HERE / "etf_universe.csv")
    for c in ("etf", "benchmark"):
        u[c] = u[c].str.strip().str.upper()
    u["manager"] = u["manager"].str.strip()
    return u


def write_outputs(res: dict, out_dir: Path, write_picks: bool = True):
    out_dir.mkdir(exist_ok=True)
    res["skill"].to_csv(out_dir / "skilled_etfs.csv", index=False)
    if len(res["ranked"]):
        res["ranked"].to_csv(out_dir / "ranked_candidates.csv", index=False)
    if write_picks and len(res["picks"]):
        target = HERE / "picks.csv"
        if target.exists():
            target.replace(HERE / "picks_backup.csv")
        res["picks"].to_csv(target, index=False)
        print(f"Wrote {len(res['picks'])} picks to picks.csv (previous saved as picks_backup.csv)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, help="number of picks (default 25)")
    ap.add_argument("--min-managers", type=int)
    ap.add_argument("--strict-benchmark", action="store_true",
                    help="drop names whose benchmark weight can't be verified")
    ap.add_argument("--no-write", action="store_true", help="don't overwrite picks.csv")
    args = ap.parse_args()

    cfg = Config()
    if args.top:
        cfg = replace(cfg, top_n=args.top)
    if args.min_managers:
        cfg = replace(cfg, min_managers=args.min_managers)
    if args.strict_benchmark:
        cfg = replace(cfg, strict_benchmark=True)

    from data_sources import LiveProvider
    res = run(LiveProvider(), load_universe(), cfg)

    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print("\nSkilled ETFs:")
    print(res["skill"][["etf", "manager", "benchmark", "months", "excess_ann", "te", "ir", "skilled"]]
          .round(3).to_string(index=False))
    if len(res["ranked"]):
        print(f"\nTop {min(cfg.top_n, len(res['ranked']))} of {len(res['ranked'])} qualifying stocks:")
        print(res["ranked"].head(cfg.top_n)[["ticker", "score", "n_managers", "avg_active_weight",
                                              "bench_weight", "fwd_growth", "beats", "avg_surprise"]]
              .round(3).to_string(index=False))
        if len(res["ranked"]) < cfg.top_n:
            print(f"\nOnly {len(res['ranked'])} names passed all filters (target was {cfg.top_n}). "
                  f"Loosen thresholds in Config or add more ETFs to etf_universe.csv rather than padding the list.")
    else:
        print("\nNo stocks passed all filters. See skilled_etfs.csv and loosen thresholds or add ETFs.")
    write_outputs(res, HERE / "out", write_picks=not args.no_write)


if __name__ == "__main__":
    main()
