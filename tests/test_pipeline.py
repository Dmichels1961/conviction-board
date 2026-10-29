"""Offline test: runs the whole pipeline on synthetic data with known right answers.

Run from the project folder:  python tests/test_pipeline.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import screener as S  # noqa: E402
from data_sources import read_holdings_csv  # noqa: E402


def hold(d):
    df = pd.DataFrame(list(d.items()), columns=["ticker", "weight"])
    df.attrs["source"] = "fake"
    return df


class FakeProvider:
    def __init__(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-10-01", periods=37, freq="MS")
        bench_r = rng.normal(0.01, 0.04, len(idx))

        def series(excess_ann, noise=0.01):
            r = bench_r + excess_ann / 12 + rng.normal(0, noise, len(idx))
            return 100 * np.cumprod(1 + r)

        p = pd.DataFrame({"B": 100 * np.cumprod(1 + bench_r),
                          "A1": series(0.06), "A2": series(0.05), "M1": series(0.04), "N1": series(0.08),
                          "W1": series(-0.06)}, index=idx)
        p["Y1"] = series(0.10)
        p.loc[p.index[:30], "Y1"] = np.nan          # only 7 months of history -> not eligible
        self._prices = p

        bench = {f"BX{i:02d}": 0.5 for i in range(205)}
        bench.update({"STK1": 2.0, "STK2": 1.5, "STK3": 7.0, "STK5": 1.0, "STK6": 2.0, "STK7": 1.0, "STK8": 1.2})
        self._h = {
            "B": hold(bench),
            "A1": hold({"STK1": 6, "STK2": 5, "STK3": 3, "STK8": 5}),
            "A2": hold({"STK2": 6, "STK3": 3}),
            "M1": hold({"STK1": 5, "STK4": 4, "STK5": 4, "STK6": 5, "STK8": 4}),
            "N1": hold({"STK1": 4, "STK4": 5, "STK5": 4, "STK6": 4, "STK8": 6}),
            "W1": hold({"STK7": 9}), "Y1": hold({"STK7": 9}),
        }

        def e(g, s):
            return dict(fwd_growth=g, cur_growth=g, surprises=s, beta=1.0)
        self._e = {
            "STK1": e(0.25, [0.06, 0.08, 0.05, 0.10]),
            "STK5": e(0.30, [0.05, -0.02, 0.04, -0.01]),   # only 2/4 beats
            "STK6": e(0.05, [0.05, 0.05, 0.05, 0.05]),     # growth too low
            "STK8": e(0.40, [0.15, 0.20, 0.10, 0.30]),     # best
        }

    def prices(self, tickers, years=3):
        return self._prices[[t for t in tickers if t in self._prices.columns]]

    def holdings(self, fund):
        return self._h.get(fund, hold({}))

    def earnings(self, t):
        return self._e[t]


universe = pd.DataFrame([
    ("A1", "MgrA", "B"), ("A2", "MgrA", "B"), ("M1", "MgrM", "B"),
    ("N1", "MgrN", "B"), ("W1", "MgrW", "B"), ("Y1", "MgrY", "B"),
], columns=["etf", "manager", "benchmark"])

cfg = replace(S.Config(), skilled_top_fraction=1.0)
res = S.run(FakeProvider(), universe, cfg, verbose=False)

skilled = set(res["skill"].query("skilled")["etf"])
assert skilled == {"A1", "A2", "M1", "N1"}, skilled          # W1 negative excess, Y1 too short
cand = set(res["candidates"]["ticker"])
# STK2: one manager only; STK3: underweight vs benchmark; STK4: not in benchmark; STK7: held by unskilled only
assert cand == {"STK1", "STK5", "STK6", "STK8"}, cand
ranked = res["ranked"]
assert list(ranked["ticker"]) == ["STK8", "STK1"], list(ranked["ticker"])   # STK5 beats<3, STK6 growth<10%
picks = res["picks"]
assert list(picks.columns) == ["ticker", "conviction", "weight", "entry_price", "target_price", "thesis"]
assert picks["conviction"].between(1, 5).all()
print(picks[["ticker", "conviction", "weight", "thesis"]].to_string(index=False))

# strict benchmark with partial benchmark list should drop unverifiable names
class PartialBench(FakeProvider):
    def holdings(self, fund):
        h = super().holdings(fund)
        return h.head(10) if fund == "B" else h
res2 = S.run(PartialBench(), universe, replace(cfg, strict_benchmark=True), verbose=False)
assert len(res2["picks"]) <= len(picks)

# holdings CSV parser: iShares-style preamble, % signs, class filter
tmp = Path(__file__).parent / "_tmp_holdings.csv"
tmp.write_text('Fund Holdings as of,"Oct 14, 2026"\nInception Date,"May 15, 2000"\n\n'
               'Ticker,Name,Sector,Asset Class,Weight (%),Price\n'
               'BRK.B,Berkshire,Financials,Equity,"1.50",500\nAAPL,Apple,IT,Equity,6.2,200\nUSD,Cash,Cash,Cash,0.3,1\n')
h = read_holdings_csv(tmp)
tmp.unlink()
assert set(h["ticker"]) == {"BRK-B", "AAPL"} and abs(h.set_index("ticker").loc["AAPL", "weight"] - 6.2) < 1e-9
print("\nALL TESTS PASSED")
