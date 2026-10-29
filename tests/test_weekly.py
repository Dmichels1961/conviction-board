"""Offline test of the weekly engine on two synthetic weeks with known answers.

    python tests/test_weekly.py
Set DASH_DATA_DIR to keep the generated snapshots (used for the dashboard smoke test).
"""
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import weekly as W  # noqa: E402


def hold(d, source="csv"):
    df = pd.DataFrame(list(d.items()), columns=["ticker", "weight"])
    df.attrs["source"] = source
    return df


BENCH = {f"BX{i:02d}": 0.5 for i in range(205)}
BENCH.update({f"S{i}": 1.0 for i in range(1, 9)})

WEEK1 = {
    "A1": hold({"S1": 6, "S2": 5, "S3": 4, "S4": 3, "S8": 3}),
    "M1": hold({"S1": 5, "S2": 4, "S5": 4, "S4": 3}),
    "N1": hold({"S1": 4, "S3": 5, "S5": 3, "S6": 2, "S7": 2}, "yahoo_top10"),
    "W1": hold({"S7": 3, "S6": 3}),                       # lags its benchmark: counts at half weight
}
S3_W2 = 4 * 1.30 / 1.05          # S3 rallied 30%, A1 +5%: unchanged share count -> weight drifts up
WEEK2 = {
    "A1": hold({"S1": 6, "S3": S3_W2, "S4": 1, "S6": 3, "S8": 3}),  # exit S2, trim S4, initiate S6
    "M1": hold({"S1": 5, "S5": 4, "S7": 3, "S4": 3}),               # exit S2, initiate S7
    "N1": hold({"S1": 4, "S3": 5, "S8": 2, "S6": 2, "S7": 2}, "yahoo_top10"),  # S5 leaves top 10, S8 enters
    "W1": hold({"S7": 3, "S6": 3}),
}


class Fake:
    def __init__(self, week):
        self.week = week
        rng = np.random.default_rng(1)
        idx = pd.date_range("2023-10-01", periods=37, freq="MS")
        b = rng.normal(0.01, 0.04, len(idx))

        def ser(x):
            return 100 * np.cumprod(1 + b + x / 12 + rng.normal(0, 0.01, len(idx)))
        self._p = pd.DataFrame({"B": 100 * np.cumprod(1 + b), "A1": ser(.06), "M1": ser(.05), "N1": ser(.07), "W1": ser(-.06)}, index=idx)
        good = [0.06, 0.08, 0.05, 0.10]
        self._e = {}
        for i, g in zip(range(1, 9), [.25, .20, .22, .18, .15, .35, .30, .28]):
            self._e[f"S{i}"] = dict(fwd_growth=g, cur_growth=g, surprises=good, beta=1.1, rev_30d=0.02 * i / 4,
                                    gross_margin=.5, roe=.2, fcf_margin=.15, fcf_yield=.03, fwd_pe=25,
                                    short_pct=.02, name=f"Stock {i}", sector="Tech")
        self._e["S6"]["surprises"] = [0.15, 0.20, 0.12, 0.25]
        self._e["S7"]["surprises"] = [0.10, 0.12, 0.10, 0.20]

    def prices(self, tickers, years=3):
        return self._p[[t for t in tickers if t in self._p.columns]]

    def holdings(self, f):
        if f == "B":
            return hold(BENCH)
        return (WEEK1 if self.week == 1 else WEEK2)[f]

    def last_prices(self, tickers):
        px = {t: 100.0 for t in tickers}
        etf = {"A1": 50.0, "M1": 40.0, "N1": 30.0, "W1": 20.0, "SPY": 500.0}
        px.update(etf)
        if self.week == 2:
            px["S3"], px["A1"], px["SPY"] = 130.0, 52.5, 505.0
            for t in tickers:
                if t in ("S6", "S7"):
                    px[t] = 100.0
        return {t: px[t] for t in tickers}

    def stock_history(self, tickers):
        idx = pd.date_range("2024-10-01", periods=24, freq="MS")
        return pd.DataFrame({t: np.linspace(80, 100 + i * 4, len(idx)) for i, t in enumerate(tickers)}, index=idx)

    def earnings(self, t):
        return self._e[t]


universe = pd.DataFrame([("A1", "MgrA", "B"), ("M1", "MgrM", "B"), ("N1", "MgrN", "B"), ("W1", "MgrW", "B")],
                        columns=["etf", "manager", "benchmark"])
themes = pd.DataFrame([("S1", "AI compute", 3), ("S6", "AI power", 3)], columns=["ticker", "theme", "tailwind"])
cfg = W.Config(skilled_top_fraction=1.0)

ddir = Path(os.environ.get("DASH_DATA_DIR") or tempfile.mkdtemp())
if (ddir / "weeks").exists():
    import shutil
    shutil.rmtree(ddir / "weeks")
quiet = lambda *a, **k: None  # noqa: E731

# ---------- week 1: no baseline -> flow neutral ----------
r1 = W.generate(Fake(1), universe, cfg, themes, date(2026, 10, 7), ddir=ddir, log=quiet)
assert r1["week"] == date(2026, 10, 5)
assert not r1["flow_available"] and (r1["all"]["sig_flow"] == 50).all()
try:
    W.generate(Fake(1), universe, cfg, themes, date(2026, 10, 9), ddir=ddir, log=quiet)
    raise AssertionError("week should be frozen")
except FileExistsError:
    pass

# ---------- week 2: flow vs week 1 ----------
r2 = W.generate(Fake(2), universe, cfg, themes, date(2026, 10, 14), ddir=ddir, log=quiet)
ev = r2["events"].set_index(["etf", "ticker"])
assert ev.loc[("A1", "S2"), "event"] == "exit"
assert ev.loc[("A1", "S6"), "event"] == "initiation"
assert ev.loc[("A1", "S4"), "event"] == "trim"
assert ("A1", "S3") not in ev.index, "price drift alone must not register as an add"
assert ev.loc[("M1", "S7"), "event"] == "initiation"
assert ev.loc[("N1", "S8"), "event"] == "initiation"
assert ev.loc[("N1", "S5"), "certainty"] == 0.5 and ev.loc[("N1", "S8"), "certainty"] == 0.5
assert ev.loc[("A1", "S2"), "value"] < 0 < ev.loc[("A1", "S6"), "value"]

a = r2["all"].set_index("ticker")
skill_flags = dict(zip(r2["skill"]["etf"], r2["skill"]["skilled"]))
assert r2["flow_available"]
assert a.loc["S6", "sig_flow"] > 50 > a.loc["S4", "sig_flow"]
assert "S2" not in a.index                       # exited by everyone -> no longer held
assert "S5" not in a.index                       # only one manager left holding it -> fails the overlap start
assert skill_flags == {"A1": True, "M1": True, "N1": True, "W1": False}
assert a.loc["S1", "n_managers"] == 3 and a.loc["S1", "sig_breadth"] == 100   # most overlap = full marks
assert a.loc["S7", "n_managers"] == 3 and a.loc["S7", "sig_breadth"] < 100   # W1 counts at half weight
assert a["sig_breadth"].corr(a["n_managers"]) > 0.5                          # more overlap, higher score
cfg_gate = W.replace(cfg, skill_gate=True) if hasattr(W, "replace") else None
rg = W.run_week(Fake(2), universe, cfg_gate, themes, pd.read_csv(ddir / "weeks" / "2026-10-05" / "holdings.csv"), log=quiet)
assert rg["all"].set_index("ticker").loc["S7", "n_managers"] == 2             # skill_gate drops W1
ranks = r2["ranking"]["rank"].tolist()
assert ranks == list(range(1, len(ranks) + 1))
assert r2["ranking"]["score"].is_monotonic_decreasing
assert abs(sum(cfg.weights.values()) - 100) < 1e-9
assert abs(a["score_raw"] - a[[c for c in a.columns if c.startswith("pts_")]].sum(axis=1)).max() < 1e-9
assert (a["score"] <= a["score_raw"] + 1e-9).all()   # coverage penalty only ever lowers a score
assert a["score"].between(0, 100).all()
assert a.loc["S6", "score"] > a.loc["S4", "score"]   # fresh initiation + strong earnings beats a trimmed name
assert list(W.list_weeks(ddir)) == ["2026-10-05", "2026-10-12"]
for f in ("ranking.csv", "ranking_all.csv", "holdings.csv", "flow_events.csv", "meta.json", "skill.csv"):
    assert (ddir / "weeks" / "2026-10-12" / f).exists(), f

print(r2["ranking"][["rank", "ticker", "score", "sig_flow", "earn_score", "n_adds", "n_removals", "drivers"]]
      .round(1).to_string(index=False))
print("\nDATA DIR:", ddir)
print("ALL TESTS PASSED")
