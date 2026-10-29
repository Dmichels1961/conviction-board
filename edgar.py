"""Free full-portfolio holdings for any US mutual fund / ETF from SEC EDGAR N-PORT filings.

  latest filing  : EDGAR company browse (accepts a fund ticker) -> newest NPORT-P -> <folder>/primary_doc.xml
  parse          : equity lines (assetCat EC/EP, long), % of net assets, identifiers (ticker if the filer gives one, else ISIN/CUSIP)
  ticker mapping : OpenFIGI (free, no key needed; ~25 requests/min, 10 ids each) cached in data/cusip_map.csv

SEC requires a descriptive User-Agent with a contact: set SEC_USER_AGENT="Your Name your@email". Public N-PORT data lags: a fund's filing
for any month becomes public 60 days after its fiscal quarter-end, so mutual-fund holdings here are 1-4 months old (ETFs with daily
disclosure should use the issuer file instead). Nothing here has been run against the live SEC site yet: see selftest().
"""
import json, os, re, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEC_UA = os.environ.get("SEC_USER_AGENT", "ConvictionBoard research set-SEC_USER_AGENT@example.com")


def _get(url, ua=None, data=None, headers=None, timeout=60):
    h = {"User-Agent": ua or SEC_UA, "Accept-Encoding": "identity"}; h.update(headers or {})
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h), timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def latest_nport_xml_url(fund_ticker):
    entries, _ = nport_entries(fund_ticker, 5)
    hrefs = [h for e in entries for h in re.findall(r"<filing-href>(.*?)</filing-href>", e)]
    if not hrefs: raise ValueError(f"no NPORT-P filings found for {fund_ticker}")
    return hrefs[0].rsplit("/", 1)[0] + "/primary_doc.xml"


def _t(el, name):
    for c in el.iter():
        if c.tag.split("}")[-1] == name: return c
    return None


def parse_nport(xml_text):
    root = ET.fromstring(xml_text)
    rep = _t(root, "repPdDate"); series = _t(root, "seriesName")
    rows = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "invstOrSec": continue
        g = lambda n: (_t(el, n).text if _t(el, n) is not None and _t(el, n).text else "")
        if g("assetCat") not in ("EC", "EP"): continue
        if g("payoffProfile") not in ("", "Long"): continue
        ident = _t(el, "identifiers"); tk = isin = ""
        if ident is not None:
            for c in ident:
                n = c.tag.split("}")[-1]
                if n == "ticker": tk = c.get("value", "")
                if n == "isin": isin = c.get("value", "")
        try: pct = float(g("pctVal"))
        except ValueError: continue
        rows.append(dict(name=g("name"), cusip=g("cusip"), isin=isin, ticker=tk, weight=pct))
    d = pd.DataFrame(rows, columns=["name", "cusip", "isin", "ticker", "weight"])
    return d[d.weight > 0].reset_index(drop=True), (rep.text if rep is not None else ""), (series.text if series is not None else "")


def figi_map(ids, cache_path=ROOT / "data/cusip_map.csv"):
    """ids: list of ISINs/CUSIPs -> {id: ticker}. Prefers US-listed composite tickers; caches results (including misses)."""
    cache = pd.read_csv(cache_path).set_index("id").ticker.to_dict() if cache_path.exists() else {}
    todo = [i for i in dict.fromkeys(ids) if i and i not in cache]
    for k in range(0, len(todo), 10):
        batch = todo[k:k + 10]
        jobs = [{"idType": "ID_ISIN" if len(i) == 12 else "ID_CUSIP", "idValue": i, "marketSecDes": "Equity"} for i in batch]
        try:
            res = json.loads(_get("https://api.openfigi.com/v3/mapping", ua="ConvictionBoard", data=json.dumps(jobs).encode(),
                                  headers={"Content-Type": "application/json"}))
        except Exception:
            time.sleep(30); continue
        for i, r in zip(batch, res):
            data = r.get("data") or []
            us = [x for x in data if x.get("exchCode") in ("US", "UN", "UW", "UQ", "UA", "UR")] or data
            cache[i] = us[0]["ticker"] if us else ""
        time.sleep(2.6)
    pd.DataFrame({"id": list(cache), "ticker": list(cache.values())}).to_csv(cache_path, index=False)
    return cache


def fetch_fund(fund_ticker):
    xml_url = latest_nport_xml_url(fund_ticker)
    d, asof, series = parse_nport(_get(xml_url))
    key = d.isin.where(d.isin != "", d.cusip)
    mp = figi_map([k for k, t in zip(key, d.ticker) if not t])
    d["ticker"] = [t if t else mp.get(k, "") for t, k in zip(d.ticker, key)]
    d = d[d.ticker != ""]
    d["ticker"] = d.ticker.str.upper().str.replace(".", "-", regex=False).str.replace(" ", "-", regex=False).replace({"GOOG": "GOOGL"})
    return d.groupby("ticker", as_index=False).agg(weight=("weight", "sum"), name=("name", "first")), asof, series, xml_url


_MF = None


def mf_ids(ticker):
    """Ticker -> (cik, seriesId, classId) from the SEC's fund ticker file (covers mutual funds and ETFs)."""
    global _MF
    if _MF is None:
        j = json.loads(_get("https://www.sec.gov/files/company_tickers_mf.json"))
        _MF = {r[3].upper(): (r[0], r[1], r[2]) for r in j["data"]}
    return _MF.get(ticker.upper())


def nport_entries(fund_ticker, n):
    """Atom entries of NPORT-P filings. EDGAR resolves some fund tickers directly; otherwise try the fund's series id, then class id."""
    ids = None
    try: ids = mf_ids(fund_ticker)
    except Exception: pass
    for key in [fund_ticker] + ([ids[1], ids[2]] if ids else []):
        q = urllib.parse.urlencode({"action": "getcompany", "CIK": key, "type": "NPORT-P", "dateb": "", "owner": "include", "count": str(n), "output": "atom"})
        entries = re.findall(r"<entry>(.*?)</entry>", _get("https://www.sec.gov/cgi-bin/browse-edgar?" + q), flags=re.S)
        if entries: return entries, key
    return [], None


def history(fund_ticker, n=12, out_dir=None, log=print):
    """Pull the last n public N-PORT filings for one fund and save each as data/holdings_history/<fund>/<period>.csv.
    The filing date (when the holdings became public) is recorded in filed.csv: the backtest only uses a snapshot after that date,
    so it cannot see holdings before the market could have. Already-saved periods are skipped."""
    out = Path(out_dir) if out_dir else ROOT / "data/holdings_history" / fund_ticker; out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "filed.csv"
    idx = pd.read_csv(idx_path) if idx_path.exists() else pd.DataFrame(columns=["period", "filed"])
    entries, used = nport_entries(fund_ticker, n)
    if not entries: log(f"history {fund_ticker}: no NPORT-P filings found by ticker, series or class id")
    for e in entries:
        href = re.search(r"<filing-href>(.*?)</filing-href>", e); fdate = re.search(r"<filing-date>(.*?)</filing-date>", e)
        if not href or not fdate: continue
        try:
            d, period, _ = parse_nport(_get(href.group(1).rsplit("/", 1)[0] + "/primary_doc.xml"))
            if not period or period in set(idx.period.astype(str)) : continue
            key = d.isin.where(d.isin != "", d.cusip)
            mp = figi_map([k for k, t in zip(key, d.ticker) if not t])
            d["ticker"] = [t if t else mp.get(k, "") for t, k in zip(d.ticker, key)]
            d = d[d.ticker != ""]
            d["ticker"] = d.ticker.str.upper().str.replace(".", "-", regex=False).str.replace(" ", "-", regex=False).replace({"GOOG": "GOOGL"})
            d.groupby("ticker", as_index=False).agg(weight=("weight", "sum")).to_csv(out / f"{period}.csv", index=False)
            idx = pd.concat([idx, pd.DataFrame([dict(period=period, filed=fdate.group(1))])], ignore_index=True)
            log(f"history {fund_ticker}: {period} filed {fdate.group(1)}, {len(d)} rows")
        except Exception as ex:
            log(f"history {fund_ticker}: {ex}")
        time.sleep(0.2)
    idx.drop_duplicates("period").to_csv(idx_path, index=False)
    return idx


def selftest():
    x = '''<edgarSubmission xmlns="http://www.sec.gov/edgar/nport"><formData><genInfo><seriesName>Test Fund</seriesName></genInfo>
    <genInfo2><repPdDate>2026-08-31</repPdDate></genInfo2><invstOrSecs>
    <invstOrSec><name>NVIDIA CORP</name><cusip>67066G104</cusip><identifiers><isin value="US67066G1040"/></identifiers><pctVal>7.5</pctVal><assetCat>EC</assetCat><payoffProfile>Long</payoffProfile></invstOrSec>
    <invstOrSec><name>APPLE INC</name><cusip>037833100</cusip><identifiers><ticker value="AAPL"/></identifiers><pctVal>4.0</pctVal><assetCat>EC</assetCat></invstOrSec>
    <invstOrSec><name>US TREASURY</name><pctVal>3.0</pctVal><assetCat>DBT</assetCat></invstOrSec>
    <invstOrSec><name>SHORT X</name><pctVal>-1.0</pctVal><assetCat>EC</assetCat><payoffProfile>Short</payoffProfile></invstOrSec>
    </invstOrSecs></formData></edgarSubmission>'''
    d, asof, series = parse_nport(x)
    assert list(d.name) == ["NVIDIA CORP", "APPLE INC"] and asof == "2026-08-31" and series == "Test Fund", d
    assert d.loc[1, "ticker"] == "AAPL" and d.loc[0, "isin"] == "US67066G1040"
    print("edgar selftest ok")


if __name__ == "__main__":
    selftest()
