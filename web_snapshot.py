"""Provider that runs the weekly pipeline on data collected from public web pages (data_web/*.csv)
instead of live APIs. Useful when the machine running the pipeline has no network, or to reproduce a
past run. Same interface as data_sources.LiveProvider.

Limits baked in: holdings are top-N lists (not full holdings), so a name missing from a fund's list is
"not shown", not "not held"; there are no prices (snapshot prices are blank, momentum is neutral); skill
comes from trailing 1Y/3Y returns rather than monthly return series.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(__file__).parent / "data_web"


class WebSnapshotProvider:
    def __init__(self, folder: Path | None = None, include_stale: bool = False):
        self.folder = Path(folder or FOLDER)
        h = pd.read_csv(self.folder / "holdings.csv")
        b = pd.read_csv(self.folder / "benchmarks.csv")
        self.h = pd.concat([h, b])
        if not include_stale:
            self.h = self.h[self.h["fresh"].astype(str) == "True"]
        ad = pd.to_datetime(self.h.get("asof"), errors="coerce") if "asof" in self.h.columns else pd.Series(dtype="datetime64[ns]")
        self._asof = ad.groupby(self.h["etf"]).max() if len(ad) else pd.Series(dtype="datetime64[ns]")
        self._ref = self._asof.max() if len(self._asof) else None      # newest file in the set stands in for "today"
        self.returns = pd.read_csv(self.folder / "returns.csv")
        e = self.folder / "earnings.csv"
        self.e = pd.read_csv(e).set_index("ticker") if e.exists() else pd.DataFrame()
        a = self.folder / "auto.csv"                  # daily machine-pulled values win over the hand-collected ones
        if a.exists():
            au = pd.read_csv(a).set_index("ticker")
            self.e = au.combine_first(self.e) if not self.e.empty else au

    # ----- interface used by weekly.run_week -----
    def holdings(self, fund: str) -> pd.DataFrame:
        d = self.h[self.h["etf"] == fund][["ticker", "weight"]].reset_index(drop=True)
        cov = d["weight"].sum()
        d.attrs["source"] = "url" if cov >= 85 else "web_top_n"   # >=85% of net assets listed: absence means not held
        return d

    def holdings_age(self, fund: str):
        """Days between this fund's holdings date and the newest holdings date in the set (None if unknown)."""
        d = self._asof.get(fund)
        return None if d is None or pd.isna(d) or self._ref is None else int((self._ref - d).days)

    def skill_table(self, universe: pd.DataFrame, cfg) -> pd.DataFrame:
        rows = []
        for _, u in universe.iterrows():
            r = self._ret(u["etf"], None)
            src = r["source"] if r is not None else None
            b = self._ret(u["benchmark"], src)
            row = dict(etf=u["etf"], manager=u["manager"], benchmark=u["benchmark"], months=0,
                       excess_ann=np.nan, excess_1y=np.nan, te=np.nan, ir=np.nan, skilled=False)
            if r is not None and b is not None:
                if pd.notna(r["ret_1y"]) and pd.notna(b["ret_1y"]):
                    row["excess_1y"] = (r["ret_1y"] - b["ret_1y"]) / 100
                    row["excess_ann"], row["months"] = row["excess_1y"], 12
                if pd.notna(r["ret_3y"]) and pd.notna(b["ret_3y"]):
                    row["excess_ann"], row["months"] = (r["ret_3y"] - b["ret_3y"]) / 100, 36
                row["skilled"] = bool(pd.notna(row["excess_ann"]) and row["excess_ann"] > 0)
            rows.append(row)
        return pd.DataFrame(rows)

    px: dict = {}                                       # set by the caller from the day's closes

    def last_prices(self, tickers) -> dict:
        return {k: self.px[k] for k in tickers if k in self.px}

    def stock_history(self, tickers) -> pd.DataFrame:
        return pd.DataFrame()

    def momentum(self, tickers) -> pd.Series:
        """12-1 momentum from earnings.csv column mom_12_1 (derived from the 52-week change and the last month)."""
        if self.e.empty or "mom_12_1" not in self.e.columns:
            return pd.Series(dtype=float)
        return self.e["mom_12_1"].reindex(tickers).dropna()

    def next_earnings(self) -> dict:
        if self.e.empty or "next_earnings" not in self.e.columns:
            return {}
        return self.e["next_earnings"].dropna().to_dict()

    def earnings(self, t: str) -> dict:
        if t not in self.e.index:
            return {}
        r = self.e.loc[t]
        g = lambda k: (None if k not in r.index or pd.isna(r[k]) else float(r[k]))  # noqa: E731
        sur = [g(k) for k in ("s1", "s2", "s3", "s4")]
        fresh = None                                   # latest surprise, scaled down as the report ages (full for 60 days, a quarter after 180)
        lr = r["last_report"] if "last_report" in r.index and isinstance(r["last_report"], str) else None
        if lr and sur[-1] is not None:
            age = (pd.Timestamp.today().normalize() - pd.Timestamp(lr)).days
            fresh = max(-0.5, min(0.5, sur[-1])) * (1.0 if age <= 60 else max(0.25, 1 - 0.75 * (age - 60) / 120))
        return dict(fwd_growth=g("fwd_growth"), cur_growth=g("cur_growth"), surprises=[s for s in sur if s is not None],
                    rev_30d=g("rev_30d"), beta=g("beta"), rev_growth=g("rev_growth"), rev_surprise=g("rev_surprise"),
                    reaction=g("reaction"), guidance=g("guidance"), pt_drift=g("pt_drift"), short_chg=g("short_chg"),
                    share_chg=g("share_chg"), implied_move=g("implied_move"), fwd_pe=g("fwd_pe"), gross_margin=g("gross_margin"),
                    roe=g("roe"), fcf_margin=g("fcf_margin"), fcf_yield=g("fcf_yield"), short_pct=g("short_pct"), insider_buyers=g("insider_buyers"),
                    gross_prof=g("gross_prof"), accruals=g("accruals"), asset_growth=g("asset_growth"), net_issuance=g("net_issuance"),
                    dispersion=g("dispersion"), fresh_surprise=fresh, resmom=g("resmom"), filing_sim=g("filing_sim"), put_skew=g("put_skew"), customer_mom=g("customer_mom"), ivol=g("ivol"),
                    name=(r["name"] if "name" in r.index and pd.notna(r["name"]) else None),
                    sector=(r["sector"] if "sector" in r.index and isinstance(r["sector"], str) else None))

    # ----- helpers -----
    def _ret(self, fund, prefer_source):
        d = self.returns[self.returns["fund"] == fund]
        if d.empty:
            return None
        if prefer_source is not None and (d["source"] == prefer_source).any():
            return d[d["source"] == prefer_source].iloc[0]
        for s in ("yahoo", "etfdb"):              # live Yahoo returns first, the older hand-collected figures as fallback
            if (d["source"] == s).any(): return d[d["source"] == s].iloc[0]
        return d.iloc[0]
