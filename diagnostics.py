"""Critique tools for a weekly ranking: how much of each score is real data, how stable the ranks are to the
(untested) weights, what the 25 names have in common, and which report earnings soon."""
from __future__ import annotations
import numpy as np
import pandas as pd
import weekly as W

# which input column proves a signal had real data for a name
_INPUT = {"flow": None, "earn_accel": "cur_growth", "earn_sales": "rev_growth", "earn_reaction": "reaction",
          "earn_guidance": "guidance", "analyst_drift": "pt_drift", "earn_growth": "fwd_growth", "earn_surprise": "avg_surprise", "earn_revisions": "rev_30d",
          "breadth": "best_raw", "active_weight": "avg_active_weight", "momentum": "mom_12_1",
          "crowding": "short_pct", "theme": "tailwind", "beta": "beta", "insider": "insider_buyers",
          "profitability": "gross_prof", "accruals": "accruals", "asset_growth": "asset_growth", "issuance": "net_issuance",
          "earn_dispersion": "dispersion", "filing_sim": "filing_sim", "put_skew": "put_skew", "customer_mom": "customer_mom", "ivol": "ivol"}

BUCKET = {**dict.fromkeys(["AVGO", "AMD", "ASML", "AMAT", "LRCX", "TSM", "WDC", "STX", "APH", "MU", "NVDA", "KLAC", "MRVL", "INTC", "TXN"], "AI hardware / semis"),
          **dict.fromkeys(["META", "GOOGL", "ORCL", "SNOW", "CRWD", "AMZN", "MSFT", "NET", "NOW"], "AI platforms / software"),
          **dict.fromkeys(["MA", "V", "JPM", "WFC"], "Financials"), **dict.fromkeys(["LLY", "ABBV"], "Health care"),
          **dict.fromkeys(["ANET", "VRT", "GLW", "PWR"], "AI infrastructure (network/power)"), **dict.fromkeys(["AMGN", "ISRG"], "Health care"), **dict.fromkeys(["GE", "RTX", "UNP", "HWM"], "Industrials"), **dict.fromkeys(["WELL", "CSCO", "PM", "NFLX", "TJX", "BKNG", "SPOT"], "Other")}


def eff_weights(rk: pd.DataFrame) -> dict:
    """Weights actually used this week (inactive signals are 0, the rest are rescaled to 100)."""
    out = {}
    for k in W.SIG_KEYS:
        m = rk[f"sig_{k}"] > 1
        out[k] = float((rk.loc[m, f"pts_{k}"] / rk.loc[m, f"sig_{k}"] * 100).median()) if m.any() else 0.0
    return out


def coverage(rk: pd.DataFrame, cfg: W.Config, flow_available: bool) -> pd.Series:
    """Share of the 100 signal points that rest on real data (not neutral-filled)."""
    out = pd.Series(0.0, index=rk.index)
    ew = eff_weights(rk)
    for k in W.SIG_KEYS:
        if k == "flow":
            ok = pd.Series(flow_available, index=rk.index)
        elif k == "earn_consistency":
            ok = rk["n_q"] >= 3
        elif k == "quality":
            ok = rk[["gross_margin", "roe", "fcf_margin"]].notna().sum(axis=1) >= 2
        elif k == "valuation":
            ok = rk["fwd_pe"].notna() & rk["fwd_growth"].notna()
        else:
            ok = rk[_INPUT[k]].notna()
        out += ew[k] * ok.astype(float)
    return out


def rank_stability(rk: pd.DataFrame, n=1000, sigma=0.4, seed=0) -> pd.DataFrame:
    """Re-score with every weight multiplied by lognormal noise; report the rank range per name."""
    rng = np.random.default_rng(seed)
    ew = eff_weights(rk)
    base = np.array([ew[k] for k in W.SIG_KEYS], float)
    S = rk[[f"sig_{k}" for k in W.SIG_KEYS]].to_numpy()
    ranks = np.empty((n, len(rk)))
    for i in range(n):
        w = base * rng.lognormal(0, sigma, len(base)); w *= 100 / w.sum()   # inactive stay 0
        sc = S @ w / 100
        ranks[i] = (-sc).argsort().argsort() + 1
    r = pd.DataFrame(ranks, columns=rk["ticker"])
    return pd.DataFrame({"base_rank": rk.set_index("ticker")["rank"], "median_rank": r.median(),
                         "rank_p10": r.quantile(.1), "rank_p90": r.quantile(.9),
                         "p_top10": (r <= 10).mean()}).sort_values("base_rank")


def concentration(rk: pd.DataFrame) -> pd.DataFrame:
    g = rk.assign(bucket=rk["ticker"].map(BUCKET).fillna("Other")).groupby("bucket")
    return pd.DataFrame({"names": g.size(), "share_%": (100 * g.size() / len(rk)).round(0),
                         "avg_beta": g["beta"].mean().round(2), "avg_rank": g["rank"].mean().round(1)}).sort_values("names", ascending=False)


def event_risk(rk: pd.DataFrame, nxt: dict, asof) -> pd.Series:
    d = pd.to_datetime(rk["ticker"].map(nxt)); days = (d - pd.Timestamp(asof)).dt.days
    return days

BUCKET.update({'MU': 'AI hardware / semis', 'NVDA': 'AI hardware / semis', 'LITE': 'AI hardware / semis', 'CIEN': 'AI hardware / semis', 'CLS': 'AI hardware / semis', 'SNPS': 'AI hardware / semis', 'SNDK': 'AI hardware / semis', 'CRDO': 'AI hardware / semis', 'MTSI': 'AI hardware / semis', 'MSFT': 'AI platforms / software', 'PLTR': 'AI platforms / software', 'APP': 'AI platforms / software', 'CRM': 'AI platforms / software', 'TWLO': 'AI platforms / software', 'RDDT': 'AI platforms / software', 'GEV': 'AI infrastructure (network/power)', 'CEG': 'AI infrastructure (network/power)', 'ETN': 'AI infrastructure (network/power)', 'MS': 'Financials', 'GS': 'Financials', 'BAC': 'Financials', 'SCHW': 'Financials', 'CB': 'Financials', 'PGR': 'Financials', 'MMC': 'Financials', 'HOOD': 'Financials', 'RKT': 'Financials', 'ALNY': 'Health care', 'ABT': 'Health care', 'TMO': 'Health care', 'DHR': 'Health care', 'VRTX': 'Health care', 'KYMR': 'Health care', 'NTRA': 'Health care', 'ASND': 'Health care', 'UNH': 'Health care', 'CAT': 'Industrials', 'TDG': 'Industrials', 'ODFL': 'Industrials', 'AAPL': 'Consumer / internet', 'TSLA': 'Consumer / internet', 'AMZN': 'Consumer / internet', 'DASH': 'Consumer / internet', 'SHOP': 'Consumer / internet', 'CVNA': 'Consumer / internet', 'SBUX': 'Consumer / internet', 'CMG': 'Consumer / internet', 'LYV': 'Consumer / internet', 'RCL': 'Consumer / internet', 'HLT': 'Consumer / internet', 'COST': 'Consumer / internet', 'SE': 'Consumer / internet', 'RBLX': 'Consumer / internet', 'W': 'Consumer / internet', 'UBER': 'Consumer / internet'})
