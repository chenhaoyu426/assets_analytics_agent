"""Tool 1/3: 市场基础数据 (Structured Market Data)

Data source priority by market:
  CN (China A-Share): EastMoney → Futu OpenD → yfinance
  HK / US / others:   Futu OpenD → yfinance
"""

from langchain_core.tools import tool

from agent_service.app.tools.market_utils import detect_market
from agent_service.app.state import ToolOutput


# ── Futu helpers (reused from futu_data.py) ─────────────────────

def _resolve_futu_codes(symbol: str) -> list[str]:
    s = symbol.strip().upper()
    if "." in s:
        return [s]

    market = detect_market(symbol)
    prefixes = market.get("futu_prefixes", ["US", "HK"])

    stripped = s.lstrip("0") or "0"
    return [f"{p}.{stripped}" for p in prefixes]


def _try_futu(symbol: str) -> tuple[str, dict] | None:
    """Try Futu OpenD for real-time data. Returns (text, fields) or None."""
    try:
        from futu import OpenQuoteContext
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    except ImportError:
        return None
    except Exception:
        return None

    try:
        codes = _resolve_futu_codes(symbol)
        snapshot_text = None
        snapshot_fields: dict = {}

        for code in codes:
            ret, data = ctx.get_market_snapshot([code])
            if ret == 0 and data is not None and not data.empty:
                row = data.iloc[0]
                snapshot_text = _format_snapshot(code, row)
                snapshot_fields = _extract_snapshot_fields(code, row)
                break

        if snapshot_text is None:
            ctx.close()
            return None

        parts = [snapshot_text]
        fields = dict(snapshot_fields)

        market_str = codes[0].split(".")[0] if snapshot_text else ""
        from futu import Market
        market_map = {
            "US": Market.US, "HK": Market.HK, "SH": Market.SH, "SZ": Market.SZ,
            "JP": Market.JP, "SG": Market.SG, "AU": Market.AU, "MY": Market.MY,
            "CA": Market.CA,
        }
        futu_market = market_map.get(market_str)
        if futu_market is not None:
            ret, info = ctx.get_stock_basicinfo(futu_market, code_list=[codes[0]])
            if ret == 0 and info is not None and not info.empty:
                parts.append(_format_basicinfo(codes[0], info.iloc[0]))

        ctx.close()
        return "\n\n".join(parts), fields
    except Exception:
        try:
            ctx.close()
        except Exception:
            pass
        return None


def _extract_snapshot_fields(code: str, row) -> dict:
    """Extract structured fields from a Futu market snapshot row."""
    import pandas as pd

    def _vf(key):
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    prefix = code.split(".")[0] if "." in code else ""
    currency_map = {
        "US": "USD", "HK": "HKD", "SH": "CNY", "SZ": "CNY",
        "JP": "JPY", "KR": "KRW", "TW": "TWD", "SG": "SGD",
        "AU": "AUD", "CA": "CAD", "UK": "GBP",
    }
    currency = currency_map.get(prefix, "USD")

    last = _vf("last_price")
    prev = _vf("prev_close_price")
    change = last - prev if (last is not None and prev is not None) else None
    change_pct = (change / prev * 100) if (change is not None and prev and prev != 0) else None

    fields: dict = {
        "source": "futu",
        "currency": currency,
    }
    if last is not None:
        fields["current_price"] = last
    if change is not None:
        fields["change"] = change
    if change_pct is not None:
        fields["change_pct"] = change_pct
    pe = _vf("pe_ratio") or _vf("pe_ttm_ratio")
    if pe is not None:
        fields["pe"] = pe
    pb = _vf("pb_ratio")
    if pb is not None:
        fields["pb"] = pb
    eps = _vf("earning_per_share")
    if eps is not None:
        fields["eps"] = eps
    mc = _vf("total_market_val")
    if mc is not None:
        if mc >= 1e12:
            fields["market_cap"] = f"{mc / 1e12:.2f}T"
        elif mc >= 1e9:
            fields["market_cap"] = f"{mc / 1e9:.2f}B"
        elif mc >= 1e6:
            fields["market_cap"] = f"{mc / 1e6:.2f}M"
        else:
            fields["market_cap"] = f"{mc:,.0f}"
    high52 = _vf("highest52weeks_price")
    if high52 is not None:
        fields["52w_high"] = high52
    low52 = _vf("lowest52weeks_price")
    if low52 is not None:
        fields["52w_low"] = low52
    div_yield = _vf("dividend_ratio_ttm")
    if div_yield is not None:
        # Futu returns dividend_ratio_ttm as a percentage (e.g. 1.23 = 1.23%).
        # Normalize to decimal ratio for consistency with the yfinance path.
        if div_yield > 1.0:
            div_yield = div_yield / 100.0
        fields["dividend_yield"] = div_yield
    volume = _vf("volume")
    if volume is not None:
        fields["volume"] = volume
    return fields


def _format_snapshot(code: str, row) -> str:
    import pandas as pd

    def _vf(key):
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _vs(key):
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return str(val)

    # Derive currency symbol from Futu code prefix
    prefix = code.split(".")[0] if "." in code else ""
    currency_map = {
        "US": "$", "HK": "HK$", "SH": "¥", "SZ": "¥",
        "JP": "¥", "KR": "₩", "TW": "NT$", "SG": "S$",
        "AU": "A$", "CA": "C$", "UK": "£",
    }
    currency_symbol = currency_map.get(prefix, "$")

    name = _vs("name") or code
    update = _vs("update_time")
    lines = [f"=== Futu Real-Time Data: {name} ({code}) ==="]
    if update:
        lines.append(f"As of: {update}")

    last = _vf("last_price")
    prev = _vf("prev_close_price")
    change = last - prev if (last is not None and prev is not None) else None
    change_pct = (change / prev * 100) if (change is not None and prev and prev != 0) else None

    lines.append("\n[Price]")
    if last is not None:
        sign = ""
        if change is not None and change_pct is not None:
            sign = "+" if change >= 0 else ""
            lines.append(f"Current: {last:.2f}  Change: {sign}{change:.2f} ({sign}{change_pct:.2f}%)")
        else:
            lines.append(f"Current: {last:.2f}")
    for f, label in [("open_price", "Open"), ("high_price", "High"), ("low_price", "Low"),
                      ("prev_close_price", "Prev Close")]:
        val = _vf(f)
        if val is not None:
            lines.append(f"{label}: {val:.2f}")

    lines.append("\n[Volume]")
    for f, label in [("volume", "Volume"), ("turnover", "Turnover"), ("turnover_rate", "Turnover Rate (%)"),
                      ("volume_ratio", "Volume Ratio")]:
        val = _vf(f)
        if val is not None:
            fmt = f"{val:.3f}" if f == "turnover_rate" else f"{val:,.0f}"
            lines.append(f"{label}: {fmt}")

    lines.append("\n[Valuation]")
    for f, label in [("pe_ratio", "P/E"), ("pe_ttm_ratio", "P/E TTM"), ("pb_ratio", "P/B"),
                      ("total_market_val", "Market Cap"), ("ey_ratio", "Earnings Yield (%)")]:
        val = _vf(f)
        if val is not None:
            fmt = _fmt_big(val, currency_symbol) if "market" in f.lower() else f"{val:.2f}"
            lines.append(f"{label}: {fmt}")

    lines.append("\n[Fundamentals]")
    for f, label in [("earning_per_share", "EPS"), ("net_asset_per_share", "Book Value/Share"),
                      ("dividend_ttm", "Dividend TTM"), ("dividend_ratio_ttm", "Dividend Yield (%)"),
                      ("issued_shares", "Issued Shares")]:
        val = _vf(f)
        if val is not None:
            fmt = f"{val:,.0f}" if f == "issued_shares" else f"{val:.2f}"
            lines.append(f"{label}: {fmt}")

    high52 = _vf("highest52weeks_price")
    low52 = _vf("lowest52weeks_price")
    if high52 or low52:
        lines.append("\n[52-Week Range]")
        if high52 is not None:
            lines.append(f"High: {high52:.2f}")
        if low52 is not None:
            lines.append(f"Low: {low52:.2f}")

    return "\n".join(lines)


def _format_basicinfo(code: str, row) -> str:
    import pandas as pd

    def _v(key):
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return val

    name = _v("name") or code
    lines = [f"=== Stock Info: {name} ({code}) ==="]
    for f, label in [("stock_type", "Type"), ("exchange_type", "Exchange"),
                      ("listing_date", "Listed"), ("lot_size", "Lot Size")]:
        val = _v(f)
        if val is not None:
            lines.append(f"{label}: {val}")
    return "\n".join(lines)


def _fmt_big(n: float, currency_symbol: str = "$") -> str:
    if abs(n) >= 1e12:
        return f"{currency_symbol}{n / 1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"{currency_symbol}{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{currency_symbol}{n / 1e6:.2f}M"
    return f"{currency_symbol}{n:,.0f}"


# ── yfinance fallback ──────────────────────────────────────────

def _fetch_yfinance_market_data(symbol: str) -> tuple[str, dict]:
    """Fallback: yfinance for profile, price, metrics + relevant index data.
    Returns (text, fields) tuple.
    """
    import yfinance as yf

    try:
        ticker = yf.Ticker(symbol.strip())
        info = ticker.get_info()

        if not info or info.get("symbol") is None:
            return f"No data found for symbol: {symbol}", {}

        name = info.get("shortName") or info.get("longName") or symbol
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        country = info.get("country", "N/A")
        market_cap = info.get("marketCap")
        description = info.get("longBusinessSummary", "")
        if description:
            sentences = description.replace("\n", " ").split(". ")
            description = ". ".join(sentences[:2]) + "."

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        currency = info.get("currency", "USD")
        change = info.get("regularMarketChange")
        change_pct = info.get("regularMarketChangePercent")

        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        eps = info.get("trailingEps")
        dividend_yield = info.get("dividendYield")
        # yfinance returns dividendYield in inconsistent formats across markets:
        # US stocks → decimal ratio (e.g. 0.0043 = 0.43%)
        # CN/HK stocks → percentage already (e.g. 1.23 = 1.23%)
        # Sanitize: values > 1.0 are already percentages, divide by 100 to normalize
        if dividend_yield is not None and dividend_yield > 1.0:
            dividend_yield = dividend_yield / 100.0
        beta = info.get("beta")
        high_52w = info.get("fiftyTwoWeekHigh")
        low_52w = info.get("fiftyTwoWeekLow")

        # Map ISO currency to symbol for display
        currency_symbol_map = {
            "USD": "$", "HKD": "HK$", "CNY": "¥", "JPY": "¥",
            "KRW": "₩", "TWD": "NT$", "SGD": "S$", "AUD": "A$",
            "CAD": "C$", "GBP": "£", "EUR": "€", "CHF": "CHF",
        }
        csym = currency_symbol_map.get(currency, "$")

        lines = [
            f"=== yfinance Market Data: {name} ({symbol}) ===",
            f"(Note: yfinance data may be delayed. Futu OpenD was unavailable.)",
            "",
            f"Sector: {sector} | Industry: {industry} | Country: {country}",
            f"Market Cap: {_fmt_big(market_cap, csym)}" if market_cap else "Market Cap: N/A",
            "",
            f"Current Price: {current_price} {currency}" if current_price else "Price: N/A",
        ]
        if change is not None and change_pct is not None:
            sign = "+" if change >= 0 else ""
            lines.append(f"Change: {sign}{change:.2f} ({sign}{change_pct:.2f}%)")

        lines.append("\n[Key Metrics]")
        lines.append(f"P/E: {pe:.2f}" if pe else "P/E: N/A")
        lines.append(f"P/B: {pb:.2f}" if pb else "P/B: N/A")
        lines.append(f"EPS: {csym}{eps:.2f}" if eps else "EPS: N/A")
        lines.append(f"Dividend Yield: {dividend_yield * 100:.2f}%" if dividend_yield else "Dividend Yield: N/A")
        lines.append(f"Beta: {beta:.2f}" if beta else "Beta: N/A")
        lines.append(f"52W High: {csym}{high_52w:.2f}" if high_52w else "52W High: N/A")
        lines.append(f"52W Low: {csym}{low_52w:.2f}" if low_52w else "52W Low: N/A")

        if description:
            lines.append(f"\n{description}")

        # Build structured fields
        fields: dict = {
            "source": "yfinance",
            "currency": currency,
            "sector": sector,
            "industry": industry,
            "country": country,
        }
        if current_price is not None:
            fields["current_price"] = current_price
        if pe is not None:
            fields["pe"] = pe
        if pb is not None:
            fields["pb"] = pb
        if eps is not None:
            fields["eps"] = eps
        if dividend_yield is not None:
            fields["dividend_yield"] = dividend_yield
        if beta is not None:
            fields["beta"] = beta
        if high_52w is not None:
            fields["52w_high"] = high_52w
        if low_52w is not None:
            fields["52w_low"] = low_52w
        if market_cap is not None:
            if market_cap >= 1e12:
                fields["market_cap"] = f"{market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9:
                fields["market_cap"] = f"{market_cap / 1e9:.2f}B"
            elif market_cap >= 1e6:
                fields["market_cap"] = f"{market_cap / 1e6:.2f}M"
            else:
                fields["market_cap"] = f"{market_cap:,.0f}"

        # Fetch a relevant market index for context
        market = info.get("market", "").lower()
        exchange = info.get("exchange", "").lower()
        index_symbol = _pick_index(market, exchange, country)
        if index_symbol:
            try:
                idx_ticker = yf.Ticker(index_symbol)
                idx_info = idx_ticker.get_info()
                idx_price = idx_info.get("currentPrice") or idx_info.get("regularMarketPrice")
                idx_change = idx_info.get("regularMarketChangePercent")
                if idx_price:
                    idx_line = f"\n[Market Index: {index_symbol} = {idx_price:.2f}"
                    if idx_change is not None:
                        sign = "+" if idx_change >= 0 else ""
                        idx_line += f" ({sign}{idx_change:.2f}%)"
                    idx_line += "]"
                    lines.append(idx_line)
            except Exception:
                pass

        return "\n".join(lines), fields
    except Exception as e:
        return f"Error fetching market data for {symbol}: {e}", {}


def _pick_index(market: str, exchange: str, country: str) -> str | None:
    idx_map = {
        "hk": "^HSI",
        "cn": "^SSEC",
        "jp": "^N225",
        "gb": "^FTSE",
        "uk": "^FTSE",
        "de": "^GDAXI",
        "fr": "^FCHI",
        "ca": "^GSPTSE",
        "au": "^AXJO",
        "sg": "^STI",
        "kr": "^KS11",
        "tw": "^TWII",
        "in": "^BSESN",
        "br": "^BVSP",
        "us": "^GSPC",
    }
    market_lower = market.lower()
    exchange_lower = exchange.lower()

    if "nasdaq" in exchange_lower:
        return "^IXIC"
    for key, idx in idx_map.items():
        if key in market_lower or key in exchange_lower or key in country.lower():
            return idx
    return "^GSPC"


# ── Main tool ──────────────────────────────────────────────────

@tool
def fetch_market_data(symbol: str) -> ToolOutput:
    """市场基础数据 — Structured market data for a ticker.

    Fetches price, volume, valuation, fundamentals, and 52-week range data.
    Data source priority by market:
      CN (China A-Share): EastMoney → Futu OpenD → yfinance
      HK / US / others:   Futu OpenD → yfinance

    Args:
        symbol: Ticker symbol (e.g. AAPL, 0700.HK, 300502.SZ)
    """
    # Detect market region for source routing
    market_info = detect_market(symbol)
    region = market_info.get("region", "")

    # CN stocks: EastMoney first (free real-time data)
    if region == "China A-Share":
        from agent_service.app.tools.eastmoney_data import _fetch_eastmoney
        em_result = _fetch_eastmoney(symbol)
        if em_result:
            text, fields = em_result
            return {
                "text": text,
                "fields": fields,
                "source": "eastmoney",
                "freshness": "realtime",
                "warnings": [],
            }

    # HK / US / others: Futu OpenD first
    futu_result = _try_futu(symbol)
    if futu_result:
        text, fields = futu_result
        return {
            "text": text,
            "fields": fields,
            "source": "futu",
            "freshness": "realtime",
            "warnings": [],
        }

    # Universal fallback: yfinance
    text, fields = _fetch_yfinance_market_data(symbol)
    warnings: list[str] = []
    if region == "China A-Share":
        warnings.append("EastMoney and Futu unavailable — using delayed yfinance data")
    else:
        warnings.append("Futu unavailable — using delayed yfinance data")
    return {
        "text": text,
        "fields": fields,
        "source": "yfinance",
        "freshness": "delayed",
        "warnings": warnings,
    }
