"""Merge the live-collected earnings inputs (data_web/live/e*.csv, g*.csv) into data_web/earnings.csv.
Rules: Yahoo estimate data is used only if its page date is on/after 2026-09-24 (or has no date but a time = today);
growth is cross-checked between stockanalysis and Yahoo and averaged when they agree within 25 pts."""
import glob, numpy as np, pandas as pd
import os
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/"
E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(ROOT + "live/e*.csv"))], ignore_index=True).set_index("ticker")
_TXT = {"ticker", "y_asof", "y_cur_label", "next_report_date", "notes"}
for _c in E.columns:
    if _c not in _TXT:
        E[_c] = pd.to_numeric(E[_c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(ROOT + "live/g*.csv"))], ignore_index=True).set_index("ticker")
import os, shutil
if not os.path.exists(ROOT + "earnings_base.csv"): shutil.copy(ROOT + "earnings.csv", ROOT + "earnings_base.csv")   # idempotent: always rebuild from the base
base = pd.read_csv(ROOT + "earnings_base.csv").set_index("ticker")
FRESH_FROM = pd.Timestamp("2026-09-24")

def fresh(a):
    if pd.isna(a): return True                       # time-of-day only = current session
    d = pd.Timestamp(str(a).replace("2025-09-25", "2026-09-25"))   # one page printed the wrong year
    return d >= FRESH_FROM

def pct(a, b):
    return (a / b - 1) if pd.notna(a) and pd.notna(b) and b not in (0,) and a > 0 and b > 0 else np.nan

rows = []
for t in sorted(set(E.index) | set(G.index)):
    e = E.loc[t] if t in E.index else None
    g = G.loc[t] if t in G.index else None
    r = dict(ticker=t)
    yok = e is not None and fresh(e["y_asof"]) and pd.notna(e["y_cur_eps"]) and pd.notna(e["y_next_eps"])
    sa_g = base["fwd_growth"].get(t, np.nan)
    if yok:
        yg = pct(e["y_next_eps"], e["y_cur_eps"])
        # revisions: average 30-day change in current- and next-FY consensus
        revs = [x for x in (pct(e["y_cur_eps"], e["y_cur_eps_30d_ago"]), pct(e["y_next_eps"], e["y_next_eps_30d_ago"])) if pd.notna(x)]
        r["rev_30d"] = float(np.mean(revs)) if revs else np.nan
        r["rev_90d"] = pct(e["y_next_eps"], e["y_next_eps_90d_ago"])
        r["y_n_analysts"] = e["y_n_analysts"]
        if pd.notna(yg) and pd.notna(sa_g) and abs(yg - sa_g) <= 0.25: r["fwd_growth"] = (yg + sa_g) / 2
        elif pd.notna(sa_g): r["fwd_growth"] = sa_g
        elif pd.notna(yg) and yg < 1.5: r["fwd_growth"] = yg
        r["growth_check"] = "agree" if pd.notna(yg) and pd.notna(sa_g) and abs(yg - sa_g) <= 0.25 else ("conflict" if pd.notna(yg) and pd.notna(sa_g) else "single")
    else:
        r["growth_check"] = "single"
    if e is not None:
        # revenue: beat share + average beat over the last 4 reports; consensus growth next FY
        sur = []
        for i in range(1, 5):
            est, act = e[f"mb_rev_q{i}_est_b"], e[f"mb_rev_q{i}_act_b"]
            if pd.notna(est) and pd.notna(act) and est > 0: sur.append(act / est - 1)
        r["rev_surprise"] = float(np.mean(sur)) if len(sur) >= 3 else np.nan
        r["rev_growth"] = e["sa_rev_growth_next_pct"] / 100 if pd.notna(e["sa_rev_growth_next_pct"]) else np.nan
        mv = [e[f"tr_move{i}_pct"] for i in range(1, 5) if pd.notna(e[f"tr_move{i}_pct"])]
        r["reaction"] = float(np.mean(mv)) / 100 if len(mv) >= 3 else np.nan
        r["implied_move"] = e["os_implied_move_pct"] / 100 if pd.notna(e["os_implied_move_pct"]) else np.nan
        r["pt_drift"] = pct(e["mb_pt_now"], e["mb_pt_3m"])
        r["net_upgrades"] = (e["mb_upgrades_30d"] or 0) - (e["mb_downgrades_30d"] or 0)
        r["short_chg"] = pct(e["sa_short_now_m"], e["sa_short_prev_m"])
        r["share_chg"] = e["sa_share_chg_yoy_pct"] / 100 if pd.notna(e["sa_share_chg_yoy_pct"]) else np.nan
        r["next_earnings"] = e["next_report_date"]
        r["insider_sales_usd_m"] = e.get("insider_sales_usd_m", np.nan)
    if g is not None:
        v, c = g["verdict"], g["vs_consensus"]
        score = {"raised": 1.0, "maintained": 0.5, "lowered": 0.1, "new": 0.5}.get(v, np.nan)
        if v == "new": score += {"above": 0.3, "below": -0.3}.get(c, 0)
        elif v in ("raised", "lowered") and c in ("above", "below"): score += {"above": 0.0, "below": -0.25}[c] if v == "raised" else 0
        r["guidance"] = score
        r["guidance_note"] = f"{v} ({g['guide_metric']} {g['guide_period']}; vs consensus {c})"
    rows.append(r)
new = pd.DataFrame(rows).set_index("ticker")
out = base.copy().astype(object)
for c in new.columns:
    if c not in out.columns: out[c] = pd.Series(np.nan, index=out.index, dtype=object)
for t, r in new.iterrows():
    if t not in out.index: out.loc[t] = np.nan
    for c, v in r.items():
        if pd.notna(v) or c in ("rev_30d",): out.loc[t, c] = v if pd.notna(v) else out.loc[t, c]
out = out.infer_objects(); out.reset_index().to_csv(ROOT + "earnings.csv", index=False)
chk = out.loc[sorted(new.index), ["fwd_growth", "growth_check", "rev_30d", "rev_growth", "rev_surprise", "reaction", "guidance", "pt_drift", "short_chg", "share_chg", "implied_move", "next_earnings"]]
pd.set_option("display.width", 250); print(chk.to_string())
