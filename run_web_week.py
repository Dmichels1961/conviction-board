"""Run the weekly board on the web-collected snapshot and print diagnostics. python run_web_week.py [--out DIR]"""
import sys, tempfile
from datetime import date
from pathlib import Path
import pandas as pd
import weekly as W, diagnostics as D
from web_snapshot import WebSnapshotProvider

asof = date(2026, 10, 22)
prov = WebSnapshotProvider(); cfg = W.Config()
res = W.generate(prov, pd.read_csv("etf_universe.csv"), cfg, W.load_themes(), asof,
                 ddir=Path(tempfile.mkdtemp()), log=lambda *a: None)
res["all"].to_csv("data_web/pool_ranked_2026_10_22.csv", index=False)
rk = res["all"].copy()
rk["data_coverage"] = D.coverage(rk, cfg, res["flow_available"]).round(0)
rk["days_to_earnings"] = D.event_risk(rk, prov.next_earnings(), asof)
rk["bucket"] = rk["ticker"].map(D.BUCKET).fillna("Other")
st = D.rank_stability(rk)
rk = rk.merge(st[["median_rank", "rank_p10", "rank_p90", "p_top10"]], left_on="ticker", right_index=True)
rk.to_csv("data_web/ranking_2026_10_22.csv", index=False)
pd.set_option("display.width", 220)
print(rk[["rank", "ticker", "score", "n_managers", "data_coverage", "median_rank", "rank_p10", "rank_p90",
          "p_top10", "days_to_earnings", "bucket"]].round(2).to_string(index=False))
print(D.concentration(rk[rk["rank"] <= 25]).to_string())
print("avg coverage", rk.data_coverage.mean().round(1), "| score range", rk.score.min().round(1), rk.score.max().round(1))
