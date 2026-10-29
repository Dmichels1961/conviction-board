"""Holdings, benchmark weights and trailing returns collected from public web pages on 2026-10-15.

Each entry is (source, as_of_date, "TICKER weight%, ...") and later sources in a fund's list only fill names the
earlier (fresher) ones did not show. Weights are as published; none of this is verified against issuer files.
Lists dated before 2026-07-17 are written with fresh=False and are ignored by default.

    python data_web/collected_2026_10_15.py      # rewrites holdings.csv, benchmarks.csv, returns.csv here
"""
import csv
from pathlib import Path

OUT = Path(__file__).parent
FRESH_AFTER = "2026-07-17"          # 90 days before collection


def parse(s):
    return [(t.split()[0], float(t.split()[1])) for t in s.replace("\n", " ").split(",") if t.strip()]


HOLDINGS = {
    "CGGR": [("stockanalysis.com", "2026-10-05",
              "META 6.63, NVDA 5.02, MSFT 4.68, AVGO 4.47, TSLA 4.22, MU 3.80, GOOG 3.23, V 3.17, GOOGL 3.09, NET 2.53, "
              "AAPL 2.49, AMZN 2.36, LLY 2.07, VRTX 1.87, MRNA 1.80, TDG 1.69, SHOP 1.58, TSM 1.58, WELL 1.58, INTC 1.37, "
              "MSTR 1.36, ISRG 1.20, ASML 1.08, APH 1.00")],
    "CGUS": [("etfdb.com", "2026-10-15",
              "NVDA 7.49, MSFT 7.28, AVGO 5.13, AMZN 4.80, LLY 3.90, META 4.61, GOOGL 3.18, TSM 2.59, AAPL 2.60, RTX 1.85, "
              "CSCO 1.90, JPM 1.76, GOOG 1.91, VRTX 1.76, MA 1.77"),
             ("marketbeat.com", "2026-09-14", "AMAT 2.09, SBUX 1.85, BLK 1.67, LIN 1.51, UBER 1.46, ADP 1.43, WFC 1.38, NFLX 1.37"),
             ("stockanalysis.com", "2026-08-11", "RCL 1.40, GE 1.39, ANET 1.37, STX 1.36")],
    "CGDV": [("stockanalysis.com", "2026-10-14",
              "MSFT 8.18, NVDA 6.72, META 5.36, AVGO 4.52, GOOGL 3.71, ORCL 2.95, RTX 2.89, GE 2.84, CSCO 2.72, AAPL 2.41, "
              "CMCSA 2.27, CARR 2.24, UNP 2.23, TSM 2.21, AMZN 2.19, SBUX 2.18, JPM 2.09, LIN 1.96, LLY 1.91, AMAT 1.83, "
              "RCL 1.79, ABBV 1.71, PM 1.59")],
    "TCHP": [("etfdb.com", "2026-10-14",
              "NVDA 17.42, MSFT 6.12, GOOG 6.00, GOOGL 5.56, AVGO 5.42, AAPL 5.10, META 4.57, AMD 3.64, AMZN 3.46, LLY 2.99, "
              "CVNA 2.85, V 2.72, TSLA 2.24, MA 1.98, LRCX 1.67"),
             ("stockanalysis.com", "2026-06-30",
              "MU 2.61, AMAT 2.50, GE 1.96, ASML 1.61, NFLX 1.46, GEV 1.11, TSM 1.06, MRVL 1.03, CB 0.98, BKNG 0.92")],
    "TGRW": [("tipranks.com", "2026-10-13",
              "NVDA 16.58, GOOGL 10.65, AAPL 6.46, MSFT 5.48, AVGO 5.37, LLY 2.99, AMD 4.20, META 3.97, MU 3.15, TSLA 2.40"),
             ("etfdb.com", "2026-10-01", "V 2.20, MA 1.78, GE 1.58, AMZN 1.45, LRCX 1.59"),
             ("stockanalysis.com", "2026-08-31",
              "AMAT 1.31, NFLX 1.30, SNOW 1.20, CRWD 1.14, TSM 1.09, APH 0.95, NOW 0.86, CVNA 0.85, ASML 0.84"),
             ("frenzycap.com", "2026-06-30",
              "CAT 1.38, ISRG 1.02, HWM 0.88, SNDK 0.84, GEV 0.82, KLAC 0.80, BKNG 0.70, VRT 0.64, SPOT 0.63, MRVL 0.62, "
              "APP 0.59, HLT 0.58, WELL 0.57, LIN 0.56, DASH 0.54, SHOP 0.54, BA 0.53, MS 0.53, ORCL 0.51, CTAS 0.51")],
    "TCAF": [("stockanalysis.com", "2026-07-30",
              "NVDA 5.11, AAPL 4.75, MSFT 4.65, AMZN 4.49, GOOGL 4.27, META 3.97, AVGO 3.88, CNP 3.10, COR 2.99, MA 2.27, "
              "KDP 2.01, AMD 1.99, TSM 1.86, AEE 1.80, V 1.75, GOOG 1.71, DHR 1.67, PPL 1.65, WAT 1.54, RVTY 1.53, NI 1.53, "
              "ASND 1.51, JPM 1.48, CNQ 1.44, AJG 1.35")],
    "FBCG": [("etfdb.com", "2026-10-13",
              "NVDA 15.68, AAPL 9.23, GOOGL 8.65, AMZN 4.97, MSFT 5.56, AVGO 4.11, LLY 2.65, META 2.42, MU 2.37, WDC 1.38, "
              "SNDK 1.84, TSLA 1.47, TSM 1.13, AMD 1.26, MRVL 1.06"),
             ("stockanalysis.com", "2026-07-31",
              "APP 1.16, MPWR 0.94, MA 0.85, SN 0.80, GEV 0.77, UNH 0.68, CLS 0.66, STX 0.64, ALAB 0.64, WELL 0.62")],
    "JGRO": [("stockanalysis.com", "2026-09-15",
              "NVDA 12.58, GOOG 10.94, AAPL 5.93, AVGO 4.33, MSFT 3.60, AMD 2.80, META 2.47, MU 2.46, LLY 2.40, PANW 1.73, "
              "TSLA 1.66, MA 1.55, NTRA 1.51, CRWD 1.50, PM 1.39, AMZN 1.35, JNJ 1.21, GS 1.19, TWLO 0.90, TSM 0.87, "
              "LRCX 0.87, GE 0.87, SNOW 0.86, MRK 0.86")],
    "DFAC": [("frenzycap.com", "2026-10-13",
              "NVDA 5.82, AAPL 5.67, MSFT 4.93, AMZN 2.58, META 2.31, GOOGL 1.69, AVGO 1.23, MU 1.16, GOOG 1.13, JPM 1.09, "
              "LLY 1.06, XOM 0.88, JNJ 0.80, V 0.76, BRK-B 0.76, CVX 0.66, MA 0.64, CSCO 0.58, PG 0.51, CAT 0.50, WMT 0.48, "
              "KO 0.44, ABBV 0.44, MRK 0.42, WDC 0.40, BAC 0.37, COP 0.35, HD 0.35, AMGN 0.34, GS 0.33, TXN 0.32, WFC 0.32, "
              "QCOM 0.32, COST 0.31, IBM 0.31, GE 0.30, INTC 0.28, UNH 0.28, ORCL 0.27, KLAC 0.27, MPC 0.27, RTX 0.27, "
              "GILD 0.27, PEP 0.27, LRCX 0.26, UNP 0.26, MS 0.25, DE 0.25, AXP 0.25")],
    "DFUS": [("etfdb.com", "2026-10-09",
              "NVDA 7.43, AAPL 6.85, MSFT 5.42, AMZN 3.48, GOOGL 2.84, AVGO 2.35, GOOG 2.32, META 2.24, MU 1.65, TSLA 1.55, "
              "JPM 1.24, BRK-B 1.34, LLY 1.31, AMD 1.43, XOM 0.98"),
             ("stockanalysis.com", "2026-09-10",
              "JNJ 0.92, V 0.90, INTC 0.67, WMT 0.66, MA 0.66, ABBV 0.64, CSCO 0.61, BAC 0.60, COST 0.57, CVX 0.56")],
}

BENCHMARKS = {   # top-60 lists only: absence from a list does NOT mean "not in the index"
    "IWF": [("ishares.com holdings CSV", "2026-10-14",
             "NVDA 15.30, AAPL 7.96, GOOGL 6.00, MSFT 5.87, GOOG 4.84, AVGO 4.77, META 3.93, TSLA 3.18, MU 3.05, LLY 2.86, "
             "AMD 2.75, V 1.97, MA 1.44, PLTR 1.32, LRCX 1.06, AMAT 1.05, CAT 1.01, GE 0.94, PANW 0.90, NFLX 0.87, TXN 0.76, "
             "GEV 0.75, CRWD 0.74, KLAC 0.69, ORCL 0.66, HD 0.65, MRVL 0.63, AMZN 0.61, SNDK 0.61, ANET 0.60, APH 0.60, "
             "COST 0.56, AMGN 0.48, ISRG 0.44, NOW 0.43, WDC 0.37, BKNG 0.37, SNOW 0.33, FTNT 0.33, GLW 0.33, MCK 0.32, "
             "LMT 0.31, TT 0.31, NET 0.31, PWR 0.29, BA 0.27, CDNS 0.27, DDOG 0.26, HWM 0.26, VRT 0.25, SPOT 0.25, MPC 0.22, "
             "HLT 0.21, TJX 0.21, CTAS 0.21, LITE 0.21, ABNB 0.20, DASH 0.20, ADBE 0.20")],
    "SPY": [("stockanalysis.com", "2026-10-14",
             "NVDA 8.05, AAPL 7.53, MSFT 5.96, AMZN 3.75, GOOGL 3.07, AVGO 2.48, GOOG 2.46, META 2.45, TSLA 1.56, MU 1.55, "
             "BRK-B 1.47, LLY 1.45, AMD 1.40, JPM 1.34, XOM 1.09, V 1.00, JNJ 0.92, WMT 0.78, ABBV 0.75, MA 0.73, INTC 0.73, "
             "PLTR 0.70, CSCO 0.67, CVX 0.66, COST 0.65"),
            ("slickcharts.com (S&P 500 index proxy, undated)", "2026-10-15",
             "ORCL 0.58, KO 0.56, UNH 0.52, PG 0.52, BAC 0.51, MRK 0.51, LRCX 0.49, AMAT 0.48, CAT 0.48, GE 0.46, PM 0.44, "
             "HD 0.44, PANW 0.43, NFLX 0.43, DELL 0.43, MS 0.41, GS 0.37, TMO 0.37, TXN 0.36, RTX 0.36, CRWD 0.36, WFC 0.35, "
             "GEV 0.35, AMGN 0.33, ANET 0.33, KLAC 0.32, LIN 0.32, IBM 0.31, MRVL 0.31, AXP 0.30, C 0.29, CRM 0.29, "
             "SNDK 0.29, APH 0.28, ADI 0.28")],
}

# trailing total returns in %, (fund, source, 1y, 3y annualized, ytd). Same-source pairs keep fund vs benchmark comparable.
RETURNS = [
    ("IWF", "etfdb", 7.20, 22.77, 5.23), ("SPY", "etfdb", 17.47, 22.75, 13.74),
    ("CGGR", "etfdb", 8.19, 24.61, 6.53), ("CGUS", "etfdb", 15.10, 22.07, 12.04), ("CGDV", "etfdb", 19.19, 24.59, 13.59),
    ("TCHP", "etfdb", 5.10, 22.30, 2.17), ("TGRW", "etfdb", 6.19, 21.57, 5.54), ("FBCG", "etfdb", 18.69, 28.64, 15.63),
    ("JGRO", "etfdb", 6.58, 20.64, 5.14), ("DFAC", "etfdb", 19.92, 21.27, 15.36), ("DFUS", "etfdb", 16.15, 23.05, 14.45),
    ("TCAF", "stockanalysis", 14.29, None, None), ("SPY", "stockanalysis", 17.80, None, None),
    ("IWF", "stockanalysis", 7.25, None, None),
]


def main():
    with open(OUT / "holdings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["etf", "ticker", "weight", "source", "asof", "fresh"])
        for etf, lists in HOLDINGS.items():
            seen = set()
            for src, asof, s in lists:
                for t, wt in parse(s):
                    if t not in seen:
                        seen.add(t)
                        w.writerow([etf, t, wt, src, asof, asof >= FRESH_AFTER])
    with open(OUT / "benchmarks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["etf", "ticker", "weight", "source", "asof", "fresh"])
        for etf, lists in BENCHMARKS.items():
            seen = set()
            for src, asof, s in lists:
                for t, wt in parse(s):
                    if t not in seen:
                        seen.add(t)
                        w.writerow([etf, t, wt, src, asof, True])
    with open(OUT / "returns.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fund", "source", "ret_1y", "ret_3y", "ytd"])
        w.writerows(RETURNS)
    print("wrote holdings.csv, benchmarks.csv, returns.csv in", OUT)


if __name__ == "__main__":
    main()
