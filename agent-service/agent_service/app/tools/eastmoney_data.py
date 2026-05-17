"""EastMoney data fetcher for China A-Share stocks.

Primary data source for CN stocks. Uses the EastMoney push API for real-time
quotes (price, change, PE, PB, market cap, volume, turnover) and supplements
with the K-line API for 52-week range.

Uses http.client directly (stdlib) to avoid proxy interference from requests
library, which caches proxy env vars at import time.
"""

import json as _json
import ssl
from http.client import HTTPSConnection
from urllib.parse import urlencode

from langchain_core.tools import tool

from agent_service.app.state import ToolOutput


EASTMONEY_HOST = "push2.eastmoney.com"
EASTMONEY_KLINE_HOST = "push2his.eastmoney.com"
EMWEB_HOST = "emweb.securities.eastmoney.com"


def _is_cn(symbol: str) -> bool:
    """Check if a symbol is a China A-Share stock."""
    s = symbol.strip().upper()
    if ".SZ" in s:
        return True
    if ".SH" in s:
        return True
    if s.isdigit() and len(s) == 6:
        return s.startswith(("0", "2", "3", "6"))
    return False


def _to_secid(symbol: str) -> str:
    """Convert a symbol to EastMoney secid format: 0.{code} or 1.{code}."""
    s = symbol.strip().upper()
    if "." in s:
        code, market = s.split(".")
        if market in ("SZ", "SHE"):
            return f"0.{code}"
        if market in ("SH", "SHG"):
            return f"1.{code}"
        return f"0.{code}"
    if s.startswith("6"):
        return f"1.{s}"
    return f"0.{s}"


def _http_get(host: str, path: str, timeout: int = 10) -> str | None:
    """Perform a direct HTTPS GET (no proxy) and return the response body."""
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        conn = HTTPSConnection(host, timeout=timeout, context=ctx)
        conn.request("GET", path, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        })
        resp = conn.getresponse()
        if resp.status != 200:
            conn.close()
            return None
        body = resp.read().decode("utf-8")
        conn.close()
        return body
    except Exception:
        return None


def _fetch_eastmoney(symbol: str) -> tuple[str, dict] | None:
    """Fetch CN stock data from EastMoney. Returns (text, fields) or None."""
    secid = _to_secid(symbol)

    try:
        # Real-time quote
        fields = (
            "f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,"
            "f116,f117,f162,f167,f168,f169,f170,f171"
        )
        path = (
            "/api/qt/stock/get?"
            + urlencode({
                "secid": secid,
                "fields": fields,
                "ut": "fa5fd1943c7b386f172d6893dbbf6613",
            })
        )
        body = _http_get(EASTMONEY_HOST, path)
        if not body:
            return None

        data = _json.loads(body)
        d = data.get("data")
        if not d:
            return None

        # Parse fields (all values are in centi-units: /100 for display)
        def _f(key):
            val = d.get(key)
            if val is None or val == "-":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        code = d.get("f57", symbol)
        name = d.get("f58", code)
        price = _f("f43")
        high = _f("f44")
        low = _f("f45")
        open_p = _f("f46")
        volume = _f("f47")
        turnover = _f("f48")
        vol_ratio = _f("f50")
        prev_close = _f("f60")
        market_cap = _f("f116")
        circ_market_cap = _f("f117")
        pe_dynamic = _f("f162")
        pb = _f("f167")
        turnover_rate = _f("f168")
        change_amt = _f("f169")
        change_pct = _f("f170")
        amplitude = _f("f171")

        if price is None:
            return None

        # Normalize from centi-units to standard units
        price = price / 100
        if high is not None:
            high = high / 100
        if low is not None:
            low = low / 100
        if open_p is not None:
            open_p = open_p / 100
        if prev_close is not None:
            prev_close = prev_close / 100
        if change_amt is not None:
            change_amt = change_amt / 100
        if change_pct is not None:
            change_pct = change_pct / 100
        if pe_dynamic is not None:
            pe_dynamic = pe_dynamic / 100
        if pb is not None:
            pb = pb / 100
        if vol_ratio is not None:
            vol_ratio = vol_ratio / 100
        if turnover_rate is not None:
            turnover_rate = turnover_rate / 100
        if amplitude is not None:
            amplitude = amplitude / 100

        # Format text output
        lines = [
            f"=== EastMoney Real-Time Data: {name} ({code}) ===",
            "Source: EastMoney (东方财富) — real-time CN market data",
            "",
            "[Price]",
        ]
        if change_amt is not None and change_pct is not None:
            sign = "+" if change_amt >= 0 else ""
            lines.append(
                f"Current: {price:.2f}  Change: {sign}{change_amt:.2f} ({sign}{change_pct:.2f}%)"
            )
        else:
            lines.append(f"Current: {price:.2f}")
        if open_p is not None:
            lines.append(f"Open: {open_p:.2f}")
        if high is not None:
            lines.append(f"High: {high:.2f}")
        if low is not None:
            lines.append(f"Low: {low:.2f}")
        if prev_close is not None:
            lines.append(f"Prev Close: {prev_close:.2f}")

        lines.append("")
        lines.append("[Volume]")
        if volume is not None:
            lines.append(f"Volume: {volume:,.0f} lots")
        if turnover is not None:
            lines.append(f"Turnover: {turnover:,.2f} CNY")
        if turnover_rate is not None:
            lines.append(f"Turnover Rate: {turnover_rate:.2f}%")
        if vol_ratio is not None:
            lines.append(f"Volume Ratio: {vol_ratio:.2f}")

        lines.append("")
        lines.append("[Valuation]")
        if pe_dynamic is not None:
            lines.append(f"P/E (Dynamic): {pe_dynamic:.2f}")
        if pb is not None:
            lines.append(f"P/B: {pb:.2f}")
        if market_cap is not None:
            lines.append(f"Market Cap: ¥{market_cap / 1e8:.2f} 亿")
        if circ_market_cap is not None:
            lines.append(f"Circulating Cap: ¥{circ_market_cap / 1e8:.2f} 亿")
        if amplitude is not None:
            lines.append(f"Amplitude: {amplitude:.2f}%")

        # Build structured fields
        out_fields: dict = {
            "source": "eastmoney",
            "currency": "CNY",
            "name": name,
            "current_price": price,
        }
        if pe_dynamic is not None:
            out_fields["pe"] = pe_dynamic
        if pb is not None:
            out_fields["pb"] = pb
        if change_pct is not None:
            out_fields["change_pct"] = change_pct
        if change_amt is not None:
            out_fields["change"] = change_amt
        if market_cap is not None:
            if market_cap >= 1e12:
                out_fields["market_cap"] = f"{market_cap / 1e12:.2f}T"
            elif market_cap >= 1e8:
                out_fields["market_cap"] = f"{market_cap / 1e8:.2f}B"
            else:
                out_fields["market_cap"] = f"{market_cap:,.0f}"

        # Fetch 52-week range from weekly K-line
        try:
            kline_path = (
                "/api/qt/stock/kline/get?"
                + urlencode({
                    "secid": secid,
                    "klt": "101",
                    "fqt": "1",
                    "lmt": "52",
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                    "ut": "fa5fd1943c7b386f172d6893dbbf6613",
                })
            )
            kline_body = _http_get(EASTMONEY_KLINE_HOST, kline_path)
            if kline_body:
                kline_data = _json.loads(kline_body)
                klines = kline_data.get("data", {}).get("klines", [])
                if klines:
                    highs = []
                    lows = []
                    for kl in klines:
                        parts = kl.split(",")
                        if len(parts) >= 4:
                            try:
                                highs.append(float(parts[3]))
                                lows.append(float(parts[4]))
                            except ValueError:
                                pass
                    if highs and lows:
                        high_52w = max(highs)
                        low_52w = min(lows)
                        out_fields["52w_high"] = high_52w
                        out_fields["52w_low"] = low_52w
                        lines.append("")
                        lines.append("[52-Week Range]")
                        lines.append(f"High: {high_52w:.2f}")
                        lines.append(f"Low: {low_52w:.2f}")
        except Exception:
            pass

        # Fetch sector/industry from EastMoney company profile
        try:
            profile_prefix = "SH" if secid.startswith("1.") else "SZ"
            profile_path = (
                "/PC_HSF10/CompanySurvey/CompanySurveyAjax?"
                + urlencode({"code": f"{profile_prefix}{code}"})
            )
            profile_body = _http_get(EMWEB_HOST, profile_path)
            if profile_body:
                profile_data = _json.loads(profile_body)
                jbzl = profile_data.get("jbzl", {})
                sector = jbzl.get("HY", "") or jbzl.get("Industry", "")
                if sector:
                    out_fields["sector"] = sector
                    out_fields["industry"] = sector
                    lines.insert(2, f"Sector: {sector} | Country: China (CN A-Share)")
        except Exception:
            pass

        return "\n".join(lines), out_fields

    except Exception:
        return None


@tool
def fetch_eastmoney_data(symbol: str) -> ToolOutput:
    """东方财富实时行情 — Real-time CN A-share market data from EastMoney.

    Provides real-time price, change, PE, PB, market cap, volume, turnover rate,
    and 52-week range for China A-share stocks.

    Args:
        symbol: Ticker symbol (e.g. 002608.SZ, 600519.SH)
    """
    result = _fetch_eastmoney(symbol)
    if result:
        text, fields = result
        return {
            "text": text,
            "fields": fields,
            "source": "eastmoney",
            "freshness": "realtime",
            "warnings": [],
        }
    return {
        "text": f"EastMoney: No data for {symbol}",
        "fields": {},
        "source": "eastmoney",
        "freshness": "realtime",
        "warnings": ["EastMoney returned no data"],
    }
