"""Append the new_earnings_h*.csv pulls (quarterly est/act, EPS cur/next, quality, price moves) to earnings_base.csv in the
base schema. Idempotent: rows already in the base keep priority only if they were not re-pulled in this batch set."""
import glob, numpy as np, pandas as pd
import os
R = os.path.dirname(os.path.abspath(__file__)) + "/"
base = pd.read_csv(R + "earnings_base.csv")
new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "new_earnings_h*.csv"))], ignore_index=True)
rows = []
for _, r in new.iterrows():
    s = []
    for i in range(1, 5):
        e, a = r.get(f"q{i}_est"), r.get(f"q{i}_act")
        s.append(round((a - e) / abs(e), 4) if pd.notna(e) and pd.notna(a) and abs(e) >= 0.05 else np.nan)
    g = r.eps_next / r.eps_cur - 1 if pd.notna(r.eps_cur) and pd.notna(r.eps_next) and r.eps_cur > 0 and r.eps_next > 0 else np.nan
    mom = (1 + r.r52w) / (1 + r.r1m) - 1 if pd.notna(r.r52w) and pd.notna(r.r1m) else np.nan
    rows.append(dict(ticker=r.ticker, fwd_growth=g, cur_growth=np.nan, s1=s[0], s2=s[1], s3=s[2], s4=s[3], rev_30d=np.nan,
                     beta=r.beta, fwd_pe=r.fwd_pe, roe=r.roe, gross_margin=r.gross_margin, fcf_margin=r.fcf_margin,
                     fcf_yield=r.fcf_yield, short_pct=r.short_pct, mom_12_1=mom, next_earnings=r.next_earnings))
add = pd.DataFrame(rows)
out = pd.concat([base[~base.ticker.isin(add.ticker)], add], ignore_index=True)
out.to_csv(R + "earnings_base.csv", index=False)
print(len(base), "+", len(add), "->", len(out))
