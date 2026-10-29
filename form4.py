"""Insider open-market buying from SEC Form 4 filings (free). A cluster of different insiders buying with their own money is one
of the few insider signals with decent academic support; routine sales are ignored (mostly pre-planned or tax related).

  insider_buyers(ticker, today) = number of distinct insiders with a code-P (open-market purchase) in the last 90 days.

Filings are cached in data/form4_cache.csv, so each daily run only downloads new ones. Needs SEC_USER_AGENT (see edgar.py).
Untested against the live site: see selftest() for the parser check.
"""
import re, time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import edgar as EG

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data/form4_cache.csv"
COLS = ["acc", "ticker", "owner", "code", "date", "shares", "price"]


def _txt(el, path):
    cur = el
    for p in path.split("/"):
        nxt = None
        for c in cur:
            if c.tag.split("}")[-1] == p: nxt = c; break
        if nxt is None: return ""
        cur = nxt
    return (cur.text or "").strip()


def parse_form4(xml_text):
    root = ET.fromstring(xml_text); owners = []
    for o in root.iter():
        if o.tag.split("}")[-1] == "reportingOwner": owners.append(_txt(o, "reportingOwnerId/rptOwnerName"))
    owner = " / ".join(x for x in owners if x)
    out = []
    for tx in root.iter():
        if tx.tag.split("}")[-1] != "nonDerivativeTransaction": continue
        code = _txt(tx, "transactionCoding/transactionCode")
        sh = _txt(tx, "transactionAmounts/transactionShares/value"); pr = _txt(tx, "transactionAmounts/transactionPricePerShare/value")
        out.append(dict(owner=owner, code=code, date=_txt(tx, "transactionDate/value"),
                        shares=float(sh) if sh else 0.0, price=float(pr) if pr else 0.0))
    return out


def filing_xml_url(index_html, folder):
    for m in re.findall(r'href="([^"]+\.xml)"', index_html):
        if "xsl" in m.lower(): continue
        return m if m.startswith("http") else "https://www.sec.gov" + m if m.startswith("/") else folder + "/" + m
    return None


def fetch_new(ticker, cache, per_ticker=40):
    q = EG.urllib.parse.urlencode({"action": "getcompany", "CIK": ticker, "type": "4", "dateb": "", "owner": "include", "count": str(per_ticker), "output": "atom"})
    atom = EG._get("https://www.sec.gov/cgi-bin/browse-edgar?" + q)
    rows = []
    for href in re.findall(r"<filing-href>(.*?)</filing-href>", atom):
        acc = href.rsplit("/", 1)[-1].replace("-index.htm", "")
        if acc in set(cache.acc): continue
        try:
            xml_url = filing_xml_url(EG._get(href), href.rsplit("/", 1)[0])
            tx = parse_form4(EG._get(xml_url)) if xml_url else []
        except Exception:
            continue
        rows += [dict(acc=acc, ticker=ticker, **t) for t in tx] or [dict(acc=acc, ticker=ticker, owner="", code="", date="", shares=0, price=0)]
        time.sleep(0.15)
    return pd.DataFrame(rows, columns=COLS)


def refresh(tickers, today=None, log=print):
    today = today or date.today()
    cache = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=COLS)
    for t in tickers:
        try:
            new = fetch_new(t, cache)
            if len(new): cache = pd.concat([cache, new], ignore_index=True)
        except Exception as e:
            log(f"form4 {t}: {e}")
    CACHE.parent.mkdir(exist_ok=True); cache.to_csv(CACHE, index=False)
    return buyers_from_cache(cache, today)


def buyers_from_cache(cache, today, days=90):
    c = cache[(cache.code == "P") & (cache.shares * cache.price > 0)].copy()
    c["date"] = pd.to_datetime(c.date, errors="coerce")
    c = c[c.date >= pd.Timestamp(today - timedelta(days=days))]
    return c.groupby("ticker").owner.nunique()


def selftest():
    x = '''<ownershipDocument><reportingOwner><reportingOwnerId><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId></reportingOwner>
    <nonDerivativeTable><nonDerivativeTransaction><transactionDate><value>2026-10-01</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>25.5</value></transactionPricePerShare></transactionAmounts>
    </nonDerivativeTransaction><nonDerivativeTransaction><transactionDate><value>2026-10-02</value></transactionDate>
    <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>50</value></transactionShares><transactionPricePerShare><value>26</value></transactionPricePerShare></transactionAmounts>
    </nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>'''
    tx = parse_form4(x); assert [t["code"] for t in tx] == ["P", "S"] and tx[0]["owner"] == "DOE JANE" and tx[0]["shares"] == 1000
    cache = pd.DataFrame([dict(acc="a", ticker="XYZ", **tx[0]), dict(acc="b", ticker="XYZ", **{**tx[0], "owner": "ROE RICK"}),
                          dict(acc="c", ticker="XYZ", **tx[1])], columns=COLS)
    assert buyers_from_cache(cache, date(2026, 10, 22))["XYZ"] == 2
    assert buyers_from_cache(cache, date(2027, 3, 1)).get("XYZ", 0) == 0     # purchases older than 90 days drop out
    idx = '<a href="/Archives/edgar/data/1/2/xslF345X05/form4.xml">x</a><a href="/Archives/edgar/data/1/2/wk-form4_1.xml">y</a>'
    assert filing_xml_url(idx, "f").endswith("wk-form4_1.xml")
    print("form4 selftest ok")


if __name__ == "__main__":
    selftest()
