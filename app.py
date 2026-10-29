"""Weekly Conviction Board: rank 1 (highest conviction) to 25.

    streamlit run app.py

Reads the weekly snapshots written by `python weekly.py` (data/weeks/<Monday>/...).
Each week the board resets; earlier weeks stay available for the track record.
For research and tracking only, not investment advice.
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from weekly import SIGNALS, SIG_KEYS, data_dir, list_weeks

st.set_page_config(page_title="Weekly Conviction Board", page_icon="📈", layout="wide")

LABEL = {k: lbl for k, lbl, *_ in SIGNALS}
GROUP = {k: g for k, _, g, _, _ in SIGNALS}
CATEGORIES = ["Thesis wrong (fundamentals)", "Right thesis, valuation compressed", "Theme / factor drawdown",
              "Crowding unwind", "One-off shock", "Manager exited / signal faded", "Bad or stale data", "Too early (timing)"]


# ---------------- data loading ----------------
@st.cache_data(ttl=30)
def load_week(week: str):
    d = data_dir() / "weeks" / week
    rk = pd.read_csv(d / "ranking.csv")
    allr = pd.read_csv(d / "ranking_all.csv")
    ev = pd.read_csv(d / "flow_events.csv") if (d / "flow_events.csv").stat().st_size > 5 else pd.DataFrame()
    import json
    meta = json.loads((d / "meta.json").read_text())
    return rk, allr, ev, meta


@st.cache_data(ttl=60)
def live_prices(tickers: tuple) -> dict:
    override = os.environ.get("DASH_PRICES_CSV")
    if override and Path(override).exists():
        d = pd.read_csv(override)
        return dict(zip(d["ticker"], d["price"]))
    try:  # Polygon / Finnhub if keys are set (see .env.example), otherwise Yahoo
        from data_sources import live_quotes
        return live_quotes(tuple(sorted(set(tickers) | {"SPY"})))
    except Exception:
        return {}


def pct(a, b):
    return (a / b - 1) * 100 if pd.notna(a) and pd.notna(b) and b else np.nan


def with_returns(df: pd.DataFrame, meta: dict, live: dict) -> pd.DataFrame:
    df = df.copy()
    df["live"] = df["ticker"].map(live)
    df["since"] = [pct(l, p) for l, p in zip(df["live"], df["price"])]
    spy = pct(live.get("SPY"), meta.get("spy_price"))
    df["vs_spy"] = df["since"] - spy if pd.notna(spy) else np.nan
    return df, spy


# ---------------- sidebar ----------------
weeks = list_weeks()
st.title("📈 Weekly Conviction Board")
if not weeks:
    st.info("No weekly snapshots yet. Run `python weekly.py` to build this week's ranking, "
            "then refresh this page.")
    st.stop()

wk = st.sidebar.selectbox("Week of (Monday)", weeks[::-1], index=0)
prev_wk = next((w for w in reversed(weeks) if w < wk), None)
refresh = st.sidebar.select_slider("Live price refresh (s)", [30, 60, 120, 300], value=60)
st.sidebar.caption("The board is frozen each week when `weekly.py` runs. Past weeks stay selectable so you can "
                   "see how earlier lists performed.")

rk, allr, ev, meta = load_week(wk)
prev_rk = load_week(prev_wk)[0] if prev_wk else None
if not meta.get("flow_available"):
    st.warning("Manager flow is neutral this week: there is no earlier holdings snapshot to compare against. "
               "It starts working from the second weekly run.")
if meta.get("benchmarks_partial"):
    st.caption(f"⚠ Benchmark holdings incomplete for {', '.join(meta['benchmarks_partial'])}; "
               "benchmark weight / active weight not fully verified. Add full CSVs to holdings/.")

# rank change vs prior week
if prev_rk is not None:
    pr = dict(zip(prev_rk["ticker"], prev_rk["rank"]))
    rk["change"] = [("NEW" if t not in pr else ("–" if pr[t] == r else (f"▲{pr[t]-r}" if pr[t] > r else f"▼{r-pr[t]}")))
                    for t, r in zip(rk["ticker"], rk["rank"])]
else:
    rk["change"] = "NEW"

tab_rank, tab_sig, tab_moves, tab_track, tab_post = st.tabs(
    ["Ranking", "Signals & scorecards", "Manager moves", "Track record", "Signal health & post-mortems"])


# ---------------- 1. ranking (live) ----------------
@st.fragment(run_every=refresh)
def ranking_view():
    live = live_prices(tuple(rk["ticker"]))
    df, spy = with_returns(rk, meta, live)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Names ranked", len(df))
    c2.metric("Avg score", f"{df['score'].mean():.1f}")
    c3.metric("New this week", int((df["change"] == "NEW").sum()))
    if df["since"].notna().any():
        c4.metric("Eq-wt return since snapshot", f"{df['since'].mean():+.2f}%",
                  None if pd.isna(spy) else f"{df['since'].mean() - spy:+.2f} pts vs SPY")
    else:
        c4.metric("Live prices", "unavailable")
    view = df[["rank", "change", "ticker", "name", "theme", "score", "sig_flow", "earn_score", "n_managers",
               "fwd_growth", "beats", "since", "vs_spy", "drivers", "thesis"]]
    st.dataframe(
        view, hide_index=True, use_container_width=True, height=min(900, 38 * (len(view) + 1)),
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "change": st.column_config.TextColumn("Δ wk", width="small"),
            "ticker": "Ticker", "name": "Name", "theme": "Theme",
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "sig_flow": st.column_config.ProgressColumn("Manager flow", min_value=0, max_value=100, format="%.0f",
                                                        help="50 = no change. Above 50: net adds. Below 50: net trims/exits."),
            "earn_score": st.column_config.ProgressColumn("Earnings", min_value=0, max_value=100, format="%.0f"),
            "n_managers": st.column_config.NumberColumn("Managers", width="small"),
            "fwd_growth": st.column_config.NumberColumn("Fwd EPS g", format="percent"),
            "beats": st.column_config.NumberColumn("Beats /4", width="small"),
            "since": st.column_config.NumberColumn("Since snapshot", format="%+.2f%%"),
            "vs_spy": st.column_config.NumberColumn("vs SPY", format="%+.2f%%"),
            "drivers": st.column_config.TextColumn("Top drivers", width="medium"),
            "thesis": st.column_config.TextColumn("Why it's here", width="large"),
        })
    st.caption(f"Snapshot taken {meta['created']}; prices refresh every {refresh}s ({datetime.now():%H:%M:%S}).")


with tab_rank:
    ranking_view()

# ---------------- 2. signals ----------------
with tab_sig:
    st.subheader("What drives the score")
    wdf = pd.DataFrame([dict(Signal=l, Group=g, Weight=meta["weights"][k], What=d) for k, l, g, _, d in SIGNALS])
    gsum = wdf.groupby("Group")["Weight"].sum()
    st.caption(" · ".join(f"{g}: {w:.0f} pts" for g, w in gsum.items()) + "  (score = weighted sum of 0-100 signal scores)")
    st.dataframe(wdf, hide_index=True, use_container_width=True,
                 column_config={"Weight": st.column_config.ProgressColumn("Weight (pts)", min_value=0, max_value=30, format="%d"),
                                "What": st.column_config.TextColumn("How it's scored", width="large")})
    hm = rk[["rank", "ticker"] + [f"sig_{k}" for k in SIG_KEYS]].copy()
    fig = go.Figure(go.Heatmap(
        z=hm[[f"sig_{k}" for k in SIG_KEYS]].values, x=[LABEL[k] for k in SIG_KEYS],
        y=[f"{r}. {t}" for r, t in zip(hm["rank"], hm["ticker"])], zmin=0, zmax=100,
        colorscale="RdYlGn", text=hm[[f"sig_{k}" for k in SIG_KEYS]].round(0).values, texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}: %{z:.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top", tickangle=-35)
    fig.update_layout(height=max(500, 26 * len(hm) + 140), margin=dict(l=10, r=10, t=120, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Scorecard")
    pick = st.selectbox("Stock", rk["ticker"], format_func=lambda t: f"{int(rk.loc[rk.ticker == t, 'rank'].iloc[0])}. {t}")
    r = rk[rk["ticker"] == pick].iloc[0]
    sc = pd.DataFrame({"Signal": [LABEL[k] for k in SIG_KEYS], "Group": [GROUP[k] for k in SIG_KEYS],
                       "Points": [r[f"pts_{k}"] for k in SIG_KEYS], "Max": [meta["weights"][k] for k in SIG_KEYS],
                       "Signal score": [r[f"sig_{k}"] for k in SIG_KEYS]})
    cA, cB = st.columns([3, 2])
    bar = px.bar(sc, x="Points", y="Signal", color="Group", orientation="h", height=420,
                 title=f"{pick}: {r['score']:.1f} / 100 (rank {int(r['rank'])})")
    bar.update_yaxes(autorange="reversed")
    cA.plotly_chart(bar, use_container_width=True)
    cB.dataframe(sc.round(1), hide_index=True, use_container_width=True)
    st.write(r["thesis"])
    if isinstance(r.get("holders"), str):
        st.caption(f"Held by: {r['holders']}  ·  avg weight vs benchmark: "
                   f"{'n/a' if pd.isna(r['avg_active_weight']) else format(r['avg_active_weight'], '+.1f') + ' pts'}")

# ---------------- 3. manager moves ----------------
with tab_moves:
    st.subheader("Skilled-manager position changes since last week")
    if ev.empty:
        st.info("No changes to show (first week, or no manager changed a position).")
    else:
        ev = ev.copy()
        ev["on_board"] = ev["ticker"].map(dict(zip(rk["ticker"], rk["rank"])))
        ev["note"] = np.where(ev["certainty"] < 1, "top-10 list only: partial certainty", "")
        cols = ["etf", "manager", "ticker", "event", "w_prev", "w_now", "delta", "on_board", "note"]
        fmt = {"w_prev": st.column_config.NumberColumn("Prev wt %", format="%.2f"),
               "w_now": st.column_config.NumberColumn("Now wt %", format="%.2f"),
               "delta": st.column_config.NumberColumn("Δ pts (drift-adj.)", format="%+.2f"),
               "on_board": st.column_config.NumberColumn("Board rank")}
        a, b = st.columns(2)
        a.markdown("**Adds & initiations**")
        a.dataframe(ev[ev["value"] > 0].sort_values("value", ascending=False)[cols], hide_index=True,
                    use_container_width=True, column_config=fmt)
        b.markdown("**Trims & exits**")
        b.dataframe(ev[ev["value"] < 0].sort_values("value")[cols], hide_index=True,
                    use_container_width=True, column_config=fmt)
        st.caption("Weight changes exclude price drift: a position that only grew because the stock rallied is not an add.")
    if prev_rk is not None:
        dropped = prev_rk[~prev_rk["ticker"].isin(rk["ticker"])][["rank", "ticker", "thesis"]]
        st.markdown("**Fell off the board since last week**")
        st.dataframe(dropped.rename(columns={"rank": "Last wk rank"}), hide_index=True, use_container_width=True) \
            if len(dropped) else st.caption("Nothing dropped off.")


# ---------------- 4. track record ----------------
@st.cache_data(ttl=60)
def track_table(weeks_: tuple, _n: int):
    rows, frames = [], {}
    for w in weeks_:
        rk_, all_, _, meta_ = load_week(w)
        live = live_prices(tuple(all_["ticker"]))
        d, spy_ = with_returns(rk_, meta_, live)
        frames[w] = with_returns(all_, meta_, live)[0]
        if d["since"].notna().any():
            rows.append(dict(week=w, n=len(d), top25=d["since"].mean(), top10=d.nsmallest(10, "rank")["since"].mean(),
                             top5=d.nsmallest(5, "rank")["since"].mean(), bottom5=d.nlargest(5, "rank")["since"].mean(),
                             spy=spy_, hit=(d["vs_spy"] > 0).mean() * 100))
    return pd.DataFrame(rows), frames


with tab_track:
    st.subheader("How each week's board has done since its snapshot")
    tt, frames = track_table(tuple(weeks), len(weeks))
    if tt.empty:
        st.info("Live prices unavailable, so returns can't be computed right now.")
    else:
        tt["excess"] = tt["top25"] - tt["spy"]
        st.dataframe(tt, hide_index=True, use_container_width=True, column_config={
            "week": "Week", "n": "Names", "top25": st.column_config.NumberColumn("Board (eq-wt)", format="%+.2f%%"),
            "top10": st.column_config.NumberColumn("Ranks 1-10", format="%+.2f%%"),
            "top5": st.column_config.NumberColumn("Ranks 1-5", format="%+.2f%%"),
            "bottom5": st.column_config.NumberColumn("Ranks 21-25", format="%+.2f%%"),
            "spy": st.column_config.NumberColumn("SPY", format="%+.2f%%"),
            "excess": st.column_config.NumberColumn("Board − SPY", format="%+.2f%%"),
            "hit": st.column_config.NumberColumn("% beating SPY", format="%.0f%%")})
        st.caption("The test of the ranking is whether top ranks beat bottom ranks and the index repeatedly, over many "
                   "weeks. A few weeks of 25 stocks is mostly noise, so avoid changing the model on one week.")

# ---------------- 5. signal health & post-mortems ----------------
with tab_post:
    st.subheader("Signal health: does each signal predict returns?")
    ics = []
    for w, f in frames.items():
        f = f.dropna(subset=["since"])
        if len(f) < 8:
            continue
        for k in SIG_KEYS:
            x, y = f[f"sig_{k}"], f["since"]
            if x.nunique() > 1:
                ics.append(dict(week=w, signal=LABEL[k], ic=x.rank().corr(y.rank())))
    if ics:
        ic = pd.DataFrame(ics)
        piv = ic.pivot(index="signal", columns="week", values="ic")
        piv["mean IC"] = piv.mean(axis=1)
        st.dataframe(piv.sort_values("mean IC", ascending=False).round(2), use_container_width=True)
        st.caption("Rank correlation between each signal's score and the return since the snapshot, across all scored "
                   "candidates each week. Persistently ≤ 0 over many weeks = the signal may be decaying. Needs 8+ names "
                   "with prices; weigh it only after a quarter or more of weeks.")
    else:
        st.info("Needs at least one week with 8+ scored names and live prices.")

    st.subheader("Post-mortems")
    pm_path = data_dir() / "postmortems.csv"
    pm = pd.read_csv(pm_path) if pm_path.exists() else pd.DataFrame(columns=["week", "ticker", "rank", "excess_vs_spy", "category", "note"])
    pw = st.selectbox("Week to review", weeks[::-1], key="pm_week")
    if pw in frames:
        rkw = load_week(pw)[0]
        live = live_prices(tuple(rkw["ticker"]))
        d, spy_ = with_returns(rkw, load_week(pw)[3], live)
        lag = d[d["vs_spy"] < 0][["rank", "ticker", "since", "vs_spy"]].copy()
        if lag.empty:
            st.caption("No pick is lagging SPY for that week.")
        else:
            done = pm[pm["week"] == pw].set_index("ticker")
            lag["category"] = lag["ticker"].map(done["category"]) if len(done) else None
            lag["note"] = lag["ticker"].map(done["note"]) if len(done) else ""
            edited = st.data_editor(lag, hide_index=True, use_container_width=True, key=f"ed_{pw}",
                                    disabled=["rank", "ticker", "since", "vs_spy"],
                                    column_config={"category": st.column_config.SelectboxColumn("Why it lagged", options=CATEGORIES),
                                                   "since": st.column_config.NumberColumn("Since", format="%+.2f%%"),
                                                   "vs_spy": st.column_config.NumberColumn("vs SPY", format="%+.2f%%")})
            if st.button("Save post-mortems"):
                new = edited.dropna(subset=["category"]).assign(week=pw).rename(columns={"vs_spy": "excess_vs_spy"})
                new = new[["week", "ticker", "rank", "excess_vs_spy", "category", "note"]]
                pm = pd.concat([pm[pm["week"] != pw], new]).reset_index(drop=True)
                pm_path.parent.mkdir(parents=True, exist_ok=True)
                pm.to_csv(pm_path, index=False)
                st.success(f"Saved {len(new)} post-mortems.")
    if len(pm):
        cnt = pm["category"].value_counts().rename_axis("category").reset_index(name="count")
        st.plotly_chart(px.bar(cnt, x="count", y="category", orientation="h", height=320,
                               title="Why picks lagged (all weeks)"), use_container_width=True)
        st.caption("Change the model only when one cause keeps recurring across many picks, not for a single stock.")

st.caption("Research and tracking only, not investment advice. Data: Yahoo Finance and issuer holdings; may be delayed.")
