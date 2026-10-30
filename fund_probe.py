"""Find working daily-holdings files for candidate active ETFs, from the GitHub runner (issuer sites are not reachable elsewhere).

  python fund_probe.py      tries every URL below, follows any holdings/download links found on issuer pages (one hop),
                            parses what comes back and writes data/fund_probe.json + data/fund_probe.txt. Changes no rankings.
A URL counts as working when it parses to 20+ holdings that add up to 30%+ of the fund and is dated within a week.
"""
import json, re, sys, urllib.parse
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
CANDIDATES = {
    "TMSL": ["https://www.troweprice.com/financial-intermediary/us/en/investments/etfs/small-mid-cap-etf.html",
             "https://www.troweprice.com/personal-investing/tools/fund-research/TMSL",
             "https://www.troweprice.com/content/fai/us/en/investments/etfs/small-mid-cap-etf.html"],
    "FFSM": ["https://fundresearch.fidelity.com/mutual-funds/composition/316092600",
             "https://www.fidelity.com/etfs/fidelity-fundamental-small-mid-cap-etf",
             "https://institutional.fidelity.com/app/funds-and-products/ETF/FFSM.html"],
    "TSME": ["https://www.thrivent.com/investing/etfs/small-mid-cap-equity-etf", "https://www.thrivent.com/etfs/tsme"],
    "TSCV": ["https://www.thrivent.com/investing/etfs/small-cap-value-etf", "https://www.thrivent.com/etfs/tscv"],
    "TMVE": ["https://www.thrivent.com/investing/etfs/mid-cap-value-etf", "https://www.thrivent.com/etfs/tmve"],
    "BSMC": ["https://www.brandes.com/etfs/brandes-u-s-small-mid-cap-value-etf", "https://www.brandes.com/etfs/bsmc",
             "https://www.brandes.com/funds/fund/brandes-u-s-small-mid-cap-value-etf"],
    "JPSV": ["https://am.jpmorgan.com/FundsMarketingHandler/excel?type=dailyETFHoldings&cusip=46654Q708&country=us&role=adv&fundType=N_ETF&locale=en-US&isUnderlyingHolding=false&isProxyHolding=false",
             "https://am.jpmorgan.com/us/en/asset-management/adv/products/jpmorgan-active-small-cap-value-etf-etf-shares-46654q708"],
    "FRTY": ["https://www.alger.com/pages/Products.aspx?productCode=2490", "https://www.alger.com/Pages/ETFs.aspx"],
}
LINK = re.compile(r"""(?:href|data-url|data-href)\s*=\s*["']([^"']*(?:holding|Holding|HOLDING|download|Download|excel|csv|xlsx)[^"']*)["']""")


def try_parse(raw, today):
    import refresh as R
    txt = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    for fmt in ("xlsx", "csv", "html"):
        try:
            if fmt == "xlsx":
                if not (isinstance(raw, bytes) and raw[:2] == b"PK"): continue
                d, asof = R.parse_holdings_xlsx(raw)
            elif fmt == "csv":
                d = R.parse_holdings_csv(txt); asof = R.csv_asof(txt)
            else:
                d = R.parse_holdings_html(txt); asof = None
            if len(d) >= 20 and d.weight.sum() >= 30:
                return fmt, len(d), round(float(d.weight.sum()), 1), asof
        except Exception:
            pass
    return None


def main():
    import refresh as R
    today = date.today(); found, lines = {}, []
    for etf, urls in CANDIDATES.items():
        seen, queue = set(), list(urls)
        while queue:
            u = queue.pop(0)
            if u in seen or len(seen) > 14: continue
            seen.add(u)
            try:
                raw = R.get_bytes(u, timeout=40)
            except Exception as e:
                lines.append(f"{etf} {u} -> {type(e).__name__} {str(e)[:80]}"); continue
            res = try_parse(raw, today)
            lines.append(f"{etf} {u} -> {len(raw)} bytes, parsed: {res}")
            if res and etf not in found:
                fmt, n, w, asof = res
                if asof is None or (today - pd.Timestamp(asof).date()).days <= 7:
                    found[etf] = dict(url=u, format=fmt, rows=n, weight=w, asof=asof)
            if u in urls and not res:                         # one hop: holdings/download links on the issuer page
                txt = raw.decode("utf-8", "replace")
                for link in LINK.findall(txt)[:12]:
                    link = urllib.parse.urljoin(u, link.replace("&amp;", "&"))
                    if link.startswith("http") and link not in seen: queue.append(link)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/fund_probe.json").write_text(json.dumps(dict(date=str(today), found=found), indent=1))
    (ROOT / "data/fund_probe.txt").write_text("\n".join(lines))
    print("\n".join(lines)); print("FOUND:", json.dumps(found, indent=1))


if __name__ == "__main__":
    main()
