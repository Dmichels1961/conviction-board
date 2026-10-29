"""Merge holdings_raw/*.csv (full lists read from issuer PDFs / SEC filings / iShares CSV) into data_web/holdings.csv
and benchmarks.csv. Where a fund has a newer top-N list than its full list, the newer weights overwrite the old ones
(names outside the top-N keep their older weight). Provenance and ages are written to data_web/holdings_provenance.csv."""
import pandas as pd, re
R = "holdings_raw/"
# fund -> (full-list file, full as-of, full source, [(overlay top-N file/list, as-of, source)])
FULL = {
 "TCHP": ("TCHP.csv", "2026-09-30", "T. Rowe Price quarterly portfolio PDF"),
 "TGRW": ("TGRW.csv", "2026-09-30", "T. Rowe Price quarterly portfolio PDF"),
 "TCAF": ("TCAF.csv", "2026-09-30", "T. Rowe Price quarterly portfolio PDF"),
 "FBCG": ("FBCG.csv", "2026-08-31", "Fidelity portfolio holdings listing PDF"),
 "JGRO": ("JGRO.csv", "2026-03-31", "SEC N-PORT"),
 "CGUS": ("CGUS.csv", "2026-05-31", "SEC N-PORT"),
 "CGDV": ("CGDV.csv", "2026-05-31", "SEC N-PORT"),
 "CGGR": ("CGGR.csv", "2026-05-31", "SEC N-PORT (80 of ~90 rows)"),
 "BAI": ("BAI.csv", "2026-10-21", "ishares.com latest-holdings.csv"),
 "ILDR": ("ILDR.csv", "2026-10-21", "ftportfolios.com holdings page"),
 "MFSV": ("MFSV.csv", "2026-10-10", "marketbeat top 25 of 64 (partial)"),
 "TDVG": ("TDVG.csv", "2026-10-20", "tipranks top 10 of 94 (partial)"),
}
# newer top-N weights from fund pages (fetched 2026-10-15)
TOP = {
 "CGGR": ("2026-09-30", "capitalgroup.com top 10", {"META": 6.6, "GOOGL": 6.4, "NVDA": 5.0, "MSFT": 4.6, "AVGO": 4.4,
          "TSLA": 4.0, "MU": 3.9, "V": 3.1, "AAPL": 2.5, "NET": 2.5}),
 "CGUS": ("2026-09-30", "capitalgroup.com top 10", {"NVDA": 7.8, "MSFT": 7.0, "AVGO": 5.2, "GOOGL": 5.1, "AMZN": 4.7,
          "META": 4.6, "LLY": 3.8, "TSM": 2.8, "AAPL": 2.5, "AMAT": 2.0}),
}
def load(f):
    d = pd.read_csv(R + f); d["ticker"] = d["ticker"].astype(str).str.strip().str.upper().replace({"GOOG": "GOOGL"}).str.replace(" ", "-").str.replace(".", "-", regex=False)
    d = d[d["ticker"].str.match(r"^[A-Z][A-Z0-9\-]*$") & ~d["ticker"].str.endswith("-TO") & (d["ticker"] != "NAN")]
    return d.groupby("ticker", as_index=False)["weight"].sum()
rows, prov = [], []
for f, (fn, asof, src) in FULL.items():
    d = load(fn); d["w"] = d["weight"]
    if f in TOP:
        a2, s2, top = TOP[f]
        for t, w in top.items(): d.loc[d.ticker == t, "w"] = w
        for t, w in top.items():
            if t not in set(d.ticker): d = pd.concat([d, pd.DataFrame([dict(ticker=t, weight=0, w=w)])])
        src += f" + {s2} ({a2})"
    d = d[d.w > 0]
    rows.append(d.assign(etf=f, weight=d.w, source=src, asof=asof, fresh=True)[["etf", "ticker", "weight", "source", "asof", "fresh"]])
    prov.append(dict(etf=f, rows=len(d), weight_sum=round(d.w.sum(), 1), full_asof=asof, source=src))
pd.concat(rows).to_csv("data_web/holdings.csv", index=False)
b = []
for name, fn, asof, src in [("IWF", "IWF.csv", "2026-10-14", "ishares.com holdings CSV"),
                            ("SPY", "SP500.csv", "2026-10-13", "ishares IVV holdings CSV (S&P 500 proxy for SPY; 443 of ~503 rows)")]:
    d = load(fn); b.append(d.assign(etf=name, source=src, asof=asof, fresh=True)[["etf", "ticker", "weight", "source", "asof", "fresh"]])
    prov.append(dict(etf=name, rows=len(d), weight_sum=round(d.weight.sum(), 1), full_asof=asof, source=src))
pd.concat(b).to_csv("data_web/benchmarks.csv", index=False)
pd.DataFrame(prov).to_csv("data_web/holdings_provenance.csv", index=False)
print(pd.DataFrame(prov).to_string(index=False))
