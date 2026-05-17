import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from agent_service.app.state import AgentState, ToolCallPlan, ToolResult, ToolOutput, ReasoningStep
from agent_service.app.llm.client_factory import create_chat_model
from agent_service.app.tools.yfinance_tools import fetch_price_history
from agent_service.app.tools.technicals import calculate_technicals
from agent_service.app.tools.market_data import fetch_market_data
from agent_service.app.tools.macro_research import fetch_macro_research
from agent_service.app.tools.sentiment_news import fetch_sentiment_news
from agent_service.app.tools.cn_market_tools import fetch_capital_flow, fetch_cn_market_sentiment
from agent_service.app.tools.us_market_tools import fetch_us_fundamentals
from agent_service.app.tools.market_utils import detect_market
from agent_service.app.prompts import (
    PLAN_PROMPT,
    OBSERVE_PROMPT,
    TOOL_REGISTRY,
    compress_tool_results,
    apply_language_instruction,
    build_synthesize_prompt,
    _now,
)
from agent_service.app.analytics.metrics import (
    compute_enriched_analytics,
    format_analytics_dashboard,
)
from agent_service.app.cache import get_cache


CORE_TOOLS = {"fetch_market_data", "fetch_macro_research", "fetch_sentiment_news"}
MAX_CORE_RETRIES = 2


def _infer_tool_meta(tool_name: str, result_text: str, ok: bool, cached: bool = False) -> dict:
    """Infer source, freshness, and warnings for a tool result."""
    source = "unknown"
    freshness = "cached" if cached else "unknown"
    warnings: list[str] = []

    if tool_name == "fetch_market_data":
        if "EastMoney" in result_text:
            source = "eastmoney"
            freshness = "realtime"
        elif "Futu Real-Time" in result_text:
            source = "futu"
            freshness = "realtime"
        elif "yfinance" in result_text:
            source = "yfinance"
            freshness = "delayed"
            warnings.append("Futu unavailable — using delayed yfinance data")
        if not ok:
            warnings.append(f"Market data collection failed: {result_text[:120]}")

    elif tool_name == "fetch_macro_research":
        source = "web_search"
        freshness = "delayed"
        if not ok:
            warnings.append(f"Macro research collection failed: {result_text[:120]}")
        elif "No macro/sector news found" in result_text:
            warnings.append("No macro/sector articles returned")

    elif tool_name == "fetch_sentiment_news":
        if "Finnhub" in result_text:
            source = "finnhub"
        elif "yfinance" in result_text:
            source = "yfinance"
        else:
            source = "web_search"
        freshness = "delayed"
        if not ok:
            warnings.append(f"News collection failed: {result_text[:120]}")

    elif tool_name == "fetch_price_history":
        source = "yfinance"
        freshness = "delayed"
        if not ok:
            warnings.append(f"Price history failed: {result_text[:120]}")

    elif tool_name == "calculate_technicals":
        source = "computed"
        freshness = "computed"
        if not ok:
            warnings.append(f"Technicals computation failed: {result_text[:120]}")

    elif tool_name in ("fetch_capital_flow", "fetch_cn_market_sentiment"):
        source = "akshare"
        freshness = "delayed"
        if not ok:
            warnings.append(f"{tool_name} failed: {result_text[:120]}")

    elif tool_name == "fetch_us_fundamentals":
        source = "yfinance"
        freshness = "delayed"
        if not ok:
            warnings.append(f"US fundamentals failed: {result_text[:120]}")

    return {"source": source, "freshness": freshness, "warnings": warnings}


def _normalize_tool_return(tool_name: str, result) -> tuple[str, dict, str, str, list[str]]:
    """Normalize a tool's return value into (text, fields, source, freshness, warnings).

    Accepts both old-style plain strings and new-style ToolOutput dicts.
    """
    if isinstance(result, dict) and "text" in result:
        text = result["text"]
        fields = result.get("fields", {})
        source = result.get("source", "unknown")
        freshness = result.get("freshness", "unknown")
        warnings = list(result.get("warnings", []))
    else:
        # Old-style: plain string — fall back to regex extraction
        text = str(result)
        fields = _extract_fields(tool_name, text)
        meta = _infer_tool_meta(tool_name, text, True)
        source = meta["source"]
        freshness = meta["freshness"]
        warnings = meta["warnings"]
    return text, fields, source, freshness, warnings


def _make_tool_result(
    tool_name: str,
    args: dict,
    result_text: str,
    ok: bool,
    call_id: str | None = None,
    cached: bool = False,
    fields: dict | None = None,
    source: str | None = None,
    freshness: str | None = None,
    warnings: list[str] | None = None,
) -> ToolResult:
    """Build a ToolResult with inferred or pre-extracted metadata."""
    summary_lines = result_text.split("\n")
    summary = summary_lines[0] if summary_lines else f"Result from {tool_name}"
    if len(summary) > 150:
        summary = summary[:147] + "..."

    # Use pre-extracted metadata if provided (from structured ToolOutput),
    # otherwise infer from the result text.
    if fields is None:
        fields = _extract_fields(tool_name, result_text) if ok else {}
    if source is None or freshness is None or warnings is None:
        meta = _infer_tool_meta(tool_name, result_text, ok, cached=cached)
        if source is None:
            source = meta["source"]
        if freshness is None:
            freshness = meta["freshness"]
        if warnings is None:
            warnings = meta["warnings"]

    result: ToolResult = {
        "tool": tool_name,
        "args": args,
        "call_id": call_id or f"{tool_name}_core",
        "summary": summary,
        "status": "ok" if ok else "error",
        "fields": fields,
        "data": {"full_result": result_text},
        "source": source,
        "freshness": freshness,
        "warnings": warnings,
    }
    return result


def _resolve_prices(state: AgentState) -> list[float] | None:
    """Extract close prices from a previous fetch_price_history result."""
    for r in state.get("tool_results", []):
        if r["tool"] == "fetch_price_history" and r.get("status") == "ok":
            closes = r.get("fields", {}).get("closes", [])
            if closes and isinstance(closes, list):
                return closes
    # Also try raw data as fallback
    for r in state.get("tool_results", []):
        if r["tool"] == "fetch_price_history" and r.get("status") == "ok":
            raw = r.get("data", {}).get("full_result", "")
            fields = _extract_fields("fetch_price_history", raw)
            closes = fields.get("closes", [])
            if closes and isinstance(closes, list):
                return closes
    return None


def _extract_fields(tool_name: str, result_text: str) -> dict:
    """Extract machine-readable fields from a tool's text output."""
    fields: dict = {}

    if tool_name == "fetch_market_data":
        is_em = "EastMoney" in result_text
        m = re.search(r"Current:\s*([\d.]+)", result_text)
        if m:
            fields["current_price"] = float(m.group(1))
        m = re.search(r"P/E(?:\s*\(Dynamic\))?(?:\s*TTM)?:\s*([\d.]+)", result_text)
        if m:
            fields["pe"] = float(m.group(1))
        m = re.search(r"P/B:\s*([\d.]+)", result_text)
        if m:
            fields["pb"] = float(m.group(1))
        m = re.search(r"EPS:\s*[¥\$]?([\d.]+)", result_text)
        if m:
            fields["eps"] = float(m.group(1))
        m = re.search(r"Market Cap:\s*[¥\$]?([\d.]+[TBM亿]?)", result_text)
        if m:
            fields["market_cap"] = m.group(1)
        m = re.search(r"Sector:\s*(\S.*?)(?:\s*[|]|\s*\n)", result_text)
        if m:
            fields["sector"] = m.group(1).strip()
        m = re.search(r"Country:\s*(\S+)", result_text)
        if m:
            fields["country"] = m.group(1).strip()
        if is_em:
            fields["source"] = "eastmoney"
        elif "Futu Real-Time" in result_text:
            fields["source"] = "futu"
        else:
            fields["source"] = "yfinance"
        m = re.search(r"As of:\s*(\S.+)", result_text)
        if m:
            fields["as_of"] = m.group(1).strip()
        # Currency extraction
        if is_em:
            fields["currency"] = "CNY"
        elif "Futu Real-Time" in result_text:
            m = re.search(r"\(([A-Z]{2})\.\d+\)", result_text)
            if m:
                prefix = m.group(1)
                currency_map = {
                    "US": "USD", "HK": "HKD", "SH": "CNY", "SZ": "CNY",
                    "JP": "JPY", "KR": "KRW", "TW": "TWD", "SG": "SGD",
                    "AU": "AUD", "CA": "CAD", "UK": "GBP",
                }
                fields["currency"] = currency_map.get(prefix, "USD")
        else:
            m = re.search(r"Current Price:\s*[\d.]+\s*(\w+)", result_text)
            if m:
                fields["currency"] = m.group(1)

    elif tool_name == "fetch_macro_research":
        m = re.search(r"Macro & Sector Research\s*\(([^)]+)\)", result_text)
        if m:
            parts = m.group(1).split("/")
            fields["region"] = parts[0].strip() if parts else ""
            fields["index"] = parts[1].strip() if len(parts) > 1 else ""
        fields["article_count"] = len(re.findall(r"^-\s*\[", result_text, re.MULTILINE))
        fields["source"] = "web_search"

    elif tool_name == "fetch_sentiment_news":
        if "Finnhub" in result_text:
            fields["source"] = "finnhub"
        elif "yfinance" in result_text:
            fields["source"] = "yfinance"
        else:
            fields["source"] = "web_search"
        m = re.search(r"(?:Articles?|Results):\s*(\d+)", result_text)
        if m:
            fields["article_count"] = int(m.group(1))
        m = re.search(r"Period:\s*(\S.+)", result_text)
        if m:
            fields["period"] = m.group(1).strip()

    elif tool_name == "fetch_price_history":
        records = []
        closes = []
        for line in result_text.split("\n"):
            m = re.match(
                r"(\d{4}-\d{2}-\d{2}):\s*O=([\d.]+)\s*H=([\d.]+)\s*L=([\d.]+)\s*C=([\d.]+)\s*V=(\d+)",
                line,
            )
            if m:
                record = {
                    "date": m.group(1),
                    "open": float(m.group(2)),
                    "high": float(m.group(3)),
                    "low": float(m.group(4)),
                    "close": float(m.group(5)),
                    "volume": int(m.group(6)),
                }
                records.append(record)
                closes.append(record["close"])
        fields["records"] = records
        fields["closes"] = closes

    elif tool_name == "calculate_technicals":
        m = re.search(r"Trend:\s*(.+)", result_text)
        if m:
            fields["trend"] = m.group(1).strip()
        m = re.search(r"RSI[^:]*:\s*([\d.]+)", result_text)
        if m:
            fields["rsi"] = float(m.group(1))
        m = re.search(r"SMA 10-day:\s*\$?([\d.]+)", result_text)
        if m:
            fields["sma_10"] = float(m.group(1))
        m = re.search(r"SMA 20-day:\s*\$?([\d.]+)", result_text)
        if m:
            fields["sma_20"] = float(m.group(1))
        m = re.search(r"Volatility[^:]*:\s*([\d.]+)%", result_text)
        if m:
            fields["volatility"] = float(m.group(1))

    elif tool_name == "fetch_capital_flow":
        fields["source"] = "akshare"
        m = re.search(r"Holdings:\s*[\d,]+\s*shares\s*\|\s*Value:\s*([\d.]+)", result_text)
        if m:
            fields["holding_value"] = float(m.group(1))
        m = re.search(r"Holding %:\s*([\d.]+)%", result_text)
        if m:
            fields["holding_pct"] = float(m.group(1))
        m = re.search(r"Holding Value Chg \(1d\):\s*([+-][\d.]+)", result_text)
        if m:
            fields["holding_chg_1d"] = float(m.group(1))

    elif tool_name == "fetch_cn_market_sentiment":
        fields["source"] = "akshare"
        m = re.search(r"LHB Appearances:\s*(\d+)", result_text)
        if m:
            fields["lhb_count"] = int(m.group(1))

    elif tool_name == "fetch_us_fundamentals":
        fields["source"] = "yfinance"
        m = re.search(r"Target Mean:\s*\$([\d.]+)\s*\(([+-][\d.]+)%", result_text)
        if m:
            fields["target_mean"] = float(m.group(1))
            fields["target_premium"] = float(m.group(2))
        m = re.search(r"percentage of shares held by institutions:\s*([\d.]+)%", result_text, re.IGNORECASE)
        if m:
            fields["inst_ownership_pct"] = float(m.group(1))
        insider_count = len(re.findall(r"^\s+\[\d{4}-\d{2}-\d{2}\]", result_text, re.MULTILINE))
        if insider_count:
            fields["insider_txn_count"] = insider_count
        m = re.search(r"Upcoming.*?(\d{4}-\d{2}-\d{2})", result_text)
        if m:
            fields["next_earnings_date"] = m.group(1)

    return fields


TOOLS_BY_NAME = {
    "fetch_market_data": fetch_market_data,
    "fetch_macro_research": fetch_macro_research,
    "fetch_sentiment_news": fetch_sentiment_news,
    "fetch_price_history": fetch_price_history,
    "calculate_technicals": calculate_technicals,
    "fetch_capital_flow": fetch_capital_flow,
    "fetch_cn_market_sentiment": fetch_cn_market_sentiment,
    "fetch_us_fundamentals": fetch_us_fundamentals,
}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("collect_core_data", collect_core_data_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("observe", observe_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("collect_core_data")
    graph.add_edge("collect_core_data", "plan")
    graph.add_conditional_edges(
        "plan",
        decide_after_plan,
        {
            "execute_tools": "execute_tools",
            "observe": "observe",
        },
    )
    graph.add_edge("execute_tools", "observe")
    graph.add_conditional_edges(
        "observe",
        decide_next,
        {
            "plan": "plan",
            "collect_core_data": "collect_core_data",
            "synthesize": "synthesize",
            "done": END,
        },
    )
    graph.add_edge("synthesize", END)

    return graph


def collect_core_data_node(state: AgentState) -> dict:
    """Deterministically run the 3 core tools in parallel, skipping cached results."""
    steps: list[ReasoningStep] = state.get("steps", [])
    existing: list[ToolResult] = list(state.get("tool_results", []))
    existing_ok = {r["tool"] for r in existing if r.get("status") == "ok"}
    missing = CORE_TOOLS - existing_ok

    if not missing:
        steps.append({
            "step_type": "tool_call",
            "status": "done",
            "message": "All core data already available (cached)",
            "detail": ", ".join(sorted(existing_ok)),
        })
        return {"steps": steps, "next_action": "plan"}

    core_retries = state.get("core_retries", 0)
    if core_retries >= MAX_CORE_RETRIES:
        steps.append({
            "step_type": "tool_call",
            "status": "done",
            "message": f"Core data still incomplete after {core_retries} retries — proceeding with available data",
            "detail": f"missing: {', '.join(sorted(missing))}",
        })
        return {"steps": steps, "next_action": "plan"}

    # Increment retry counter if this is a re-collection (not the first pass)
    if existing_ok:
        core_retries += 1

    steps.append({
        "step_type": "tool_call",
        "status": "active",
        "message": f"Collecting core data: {', '.join(sorted(missing))}...",
    })

    symbol = state["symbol"]
    finnhub_key = state.get("finnhub_api_key")

    def _run_core(tool_name: str) -> tuple[str, str, dict, str, str, list[str], bool]:
        tool_fn = TOOLS_BY_NAME[tool_name]
        args: dict = {"symbol": symbol}
        if tool_name == "fetch_sentiment_news" and finnhub_key:
            args["finnhub_api_key"] = finnhub_key
        try:
            raw = tool_fn.invoke(args)
            text, fields, source, freshness, warnings = _normalize_tool_return(tool_name, raw)
            return tool_name, text, fields, source, freshness, warnings, True
        except Exception as e:
            return tool_name, f"Error executing {tool_name}: {str(e)}", {}, "unknown", "unknown", [str(e)], False

    new_results: list[ToolResult] = []
    with ThreadPoolExecutor(max_workers=len(missing)) as executor:
        futures = {executor.submit(_run_core, name): name for name in missing}
        for future in as_completed(futures):
            tool_name, result_text, fields, source, freshness, warnings, ok = future.result()
            new_results.append(_make_tool_result(
                tool_name, {"symbol": symbol}, result_text, ok,
                call_id=f"{tool_name}_core",
                fields=fields, source=source, freshness=freshness, warnings=warnings,
            ))

    accumulated = existing + new_results

    # Auto-collect supplementary data based on market region
    try:
        market = detect_market(symbol)
        extras = []

        # CN/HK: Stock Connect capital flow
        if market["region"] in ("China A-Share", "Hong Kong"):
            try:
                cf_result = fetch_capital_flow.invoke({"symbol": symbol})
                accumulated.append(_make_tool_result(
                    "fetch_capital_flow", {"symbol": symbol}, cf_result, True,
                    call_id="fetch_capital_flow_core",
                ))
                extras.append("capital flow")
            except Exception as e:
                accumulated.append(_make_tool_result(
                    "fetch_capital_flow", {"symbol": symbol},
                    f"Error: {e}", False,
                    call_id="fetch_capital_flow_core",
                ))

        # US: Fundamentals (analyst consensus, insider trades, earnings, etc.)
        if market["region"] == "United States":
            try:
                us_result = fetch_us_fundamentals.invoke({"symbol": symbol})
                accumulated.append(_make_tool_result(
                    "fetch_us_fundamentals", {"symbol": symbol}, us_result, True,
                    call_id="fetch_us_fundamentals_core",
                ))
                extras.append("US fundamentals")
            except Exception as e:
                accumulated.append(_make_tool_result(
                    "fetch_us_fundamentals", {"symbol": symbol},
                    f"Error: {e}", False,
                    call_id="fetch_us_fundamentals_core",
                ))

        if extras:
            steps[-1]["detail"] = f"Collected {len(new_results)} core + {', '.join(extras)}"
    except Exception as e:
        steps[-1]["detail"] = f"Market detection failed: {e}"

    steps[-1]["status"] = "done"

    return {
        "tool_results": accumulated,
        "steps": steps,
        "next_action": "plan",
        "core_retries": core_retries,
    }


def _build_llm(state: AgentState):
    config = state["llm_config"]
    return create_chat_model(
        provider=config["provider"],
        model=config["model"],
        api_key=config["api_key"],
        base_url=config.get("base_url"),
    )


def _parse_plan_response(content: str, symbol: str) -> tuple[str, list[ToolCallPlan]]:
    """Parse LLM response into reasoning text and plan list."""
    reasoning = ""
    plan: list[ToolCallPlan] = []

    content = content.strip()

    # Try newline-separated format: reasoning line then JSON line
    lines = content.split("\n")
    json_str = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            json_str = stripped
        elif stripped.startswith("{"):
            json_str = stripped
        elif not json_str and stripped:
            # Lines before JSON are reasoning
            if not reasoning:
                reasoning = stripped
            else:
                reasoning += " " + stripped

    # Fallback: treat whole content as JSON
    if not json_str:
        json_str = content
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[-1] if "\n" in json_str else ""
        if json_str.endswith("```"):
            json_str = json_str[:-3].strip()

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            plan = parsed
        elif isinstance(parsed, dict):
            plan = [parsed]
    except json.JSONDecodeError:
        plan = [{"tool": "fetch_price_history", "args": {"symbol": symbol, "period": "6mo"}}]
        if not reasoning:
            reasoning = "Gathering price history for technical context"

    if not reasoning:
        reasoning = f"Planned {len(plan)} tool call(s)"

    return reasoning, plan


def plan_node(state: AgentState) -> dict:
    steps: list[ReasoningStep] = state.get("steps", [])
    iteration = state.get("iteration_count", 0) + 1
    steps.append({
        "step_type": "planning",
        "status": "active",
        "message": f"Planning analysis for {state['symbol']}...",
    })

    language = state.get("language", "en")
    llm = _build_llm(state)

    existing_results: list[ToolResult] = state.get("tool_results", [])
    available_names = [r["tool"] for r in existing_results]
    available_block = (
        "Core data already collected:\n"
        + "\n".join(f"  - {name}: {r.get('status', 'ok')}" for r in existing_results for name in [r['tool']])
        + "\n\nPlan ONLY additional tools beyond the core three. "
        "Consider: fetch_price_history (choose period: 1mo, 6mo, 1y, 5y, max) "
        "and calculate_technicals (requires price data first). "
        "If no additional data would meaningfully improve the analysis, plan []."
    )

    prompt = apply_language_instruction(
        PLAN_PROMPT.format(
            symbol=state["symbol"],
            tool_descriptions=TOOL_REGISTRY,
            current_date=_now(),
            available_data=available_block,
        ),
        language,
    )

    messages = state.get("messages", [])
    messages.append(SystemMessage(content=prompt))

    response = llm.invoke(messages)
    messages.append(HumanMessage(content="Plan my analysis"))
    messages.append(response)

    content = response.content if hasattr(response, "content") else str(response)
    reasoning, plan = _parse_plan_response(content, state["symbol"])

    steps[-1]["status"] = "done"
    steps[-1]["detail"] = reasoning
    for i, p in enumerate(plan):
        call_id = f"{p['tool']}_{i}"
        p["call_id"] = call_id
        steps.append({
            "step_type": "tool_call",
            "status": "pending",
            "message": f"Planned: {p['tool']}",
            "detail": str(p.get("args", {})),
            "call_id": call_id,
        })

    return {
        "plan": plan,
        "messages": messages,
        "steps": steps,
        "iteration_count": iteration,
        "next_action": "observe" if not plan else "execute_tools",
    }


def execute_tools_node(state: AgentState) -> dict:
    plan: list[ToolCallPlan] = state["plan"]
    steps: list[ReasoningStep] = state.get("steps", [])
    messages = state.get("messages", [])

    for step in steps:
        if step["status"] == "pending" and step["step_type"] == "tool_call":
            step["status"] = "active"
            step["message"] = f"Calling {step.get('detail', 'tool')}..."

    def _run_tool(call: ToolCallPlan) -> tuple[str, dict, str, dict, str, str, list[str], bool]:
        tool_name = call["tool"]
        args = dict(call.get("args", {}))
        tool_fn = TOOLS_BY_NAME.get(tool_name)
        if tool_fn is None:
            return tool_name, args, f"Error: Unknown tool '{tool_name}'", {}, "unknown", "unknown", [], False
        # Inject request-scoped config from state
        if tool_name == "fetch_sentiment_news" and "finnhub_api_key" not in args:
            key = state.get("finnhub_api_key") or state.get("llm_config", {}).get("finnhub_api_key")
            if key:
                args["finnhub_api_key"] = key
        # Auto-wire prices from fetch_price_history result.
        # Tools run in parallel — the price history may not have finished yet,
        # so run it inline as a fallback if not already in accumulated results.
        if tool_name == "calculate_technicals" and (
            "prices" not in args or not isinstance(args.get("prices"), list)
        ):
            prices = _resolve_prices(state)
            if not prices:
                try:
                    ph_result = fetch_price_history.invoke({
                        "symbol": state["symbol"],
                        "period": "6mo",
                    })
                    prices = _extract_fields("fetch_price_history", ph_result).get("closes", [])
                except Exception:
                    prices = []
            if prices:
                args["prices"] = prices
        try:
            raw = tool_fn.invoke(args)
            text, fields, source, freshness, warnings = _normalize_tool_return(tool_name, raw)
            return tool_name, args, text, fields, source, freshness, warnings, True
        except Exception as e:
            return tool_name, args, f"Error executing {tool_name}: {str(e)}", {}, "unknown", "unknown", [str(e)], False

    results_by_id: dict[str, tuple[str, dict, str, str, list[str], bool]] = {}
    with ThreadPoolExecutor(max_workers=min(len(plan), 5)) as executor:
        futures = {
            executor.submit(_run_tool, call): call
            for call in plan
        }
        for future in as_completed(futures):
            tool_name, args, result_text, fields, source, freshness, warnings, ok = future.result()
            call = futures[future]
            cid = call.get("call_id", tool_name)
            results_by_id[cid] = (result_text, fields, source, freshness, warnings, ok)

    tool_results: list[ToolResult] = []
    for call in plan:
        tool_name = call["tool"]
        args = call.get("args", {})
        cid = call.get("call_id", tool_name)
        entry = results_by_id.get(cid)
        if entry is None:
            result_text = f"Error: No result for '{tool_name}'"
            ok = False
            fields = {}
            source = "unknown"
            freshness = "unknown"
            warnings: list[str] = []
        else:
            result_text, fields, source, freshness, warnings, ok = entry

        for step in steps:
            if step["status"] == "active" and step["step_type"] == "tool_call":
                if step.get("call_id") == cid:
                    step["status"] = "done"
                    step["message"] = f"Completed: {tool_name}"
                    break

        tr = _make_tool_result(
            tool_name, args, result_text, ok, call_id=cid,
            fields=fields, source=source, freshness=freshness, warnings=warnings,
        )
        tool_results.append(tr)

        messages.append(HumanMessage(content=f"Tool: {tool_name}\nArgs: {json.dumps(args)}\nResult: {tr['summary']}"))

    # Merge with previously accumulated results (e.g. pre-fetched cached data)
    accumulated = list(state.get("tool_results", []))
    accumulated.extend(tool_results)

    return {
        "tool_results": accumulated,
        "steps": steps,
        "messages": messages,
        "next_action": "observe",
    }


def _parse_observe_response(content: str) -> tuple[str, list[str], str]:
    """Parse observe LLM response. Returns (decision, missing_fields, reasoning)."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1] if "\n" in content else ""
    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        data = json.loads(content)
        return (
            data.get("decision", "enough"),
            data.get("missing", []),
            data.get("reasoning", ""),
        )
    except json.JSONDecodeError:
        content_lower = content.lower()
        if content_lower.startswith("more"):
            return "more", [], content
        return "enough", [], content


def validate_coverage(tool_results: list[ToolResult]) -> dict:
    """Deterministic check: are all 3 core data types present and healthy?"""
    present = {r["tool"]: r for r in tool_results}
    missing = [t for t in CORE_TOOLS if t not in present]
    errored = [t for t in CORE_TOOLS if t in present and present[t].get("status") == "error"]
    ok = [t for t in CORE_TOOLS if t in present and present[t].get("status") == "ok"]
    return {
        "core_complete": len(ok) == 3,
        "ok": ok,
        "missing": missing,
        "errored": errored,
    }


def observe_node(state: AgentState) -> dict:
    steps: list[ReasoningStep] = state.get("steps", [])
    steps.append({
        "step_type": "evaluating",
        "status": "active",
        "message": "Evaluating collected data...",
    })

    # --- Deterministic coverage check ---
    coverage = validate_coverage(state["tool_results"])
    if not coverage["core_complete"]:
        detail_parts = []
        if coverage["missing"]:
            detail_parts.append(f"missing: {', '.join(coverage['missing'])}")
        if coverage["errored"]:
            detail_parts.append(f"errors: {', '.join(coverage['errored'])}")
        steps[-1]["status"] = "done"
        steps[-1]["detail"] = f"Incomplete core data — {'; '.join(detail_parts)}"
        if coverage["missing"]:
            missing_str = ", ".join(coverage["missing"])
            msg = f"Re-collecting core data ({missing_str})..."
        else:
            msg = "Re-collecting core data..."
        steps.append({
            "step_type": "tool_call",
            "status": "pending",
            "message": msg,
        })
        return {
            "steps": steps,
            "next_action": "collect_core_data",
        }

    # --- LLM qualitative judgment ---
    language = state.get("language", "en")
    llm = _build_llm(state)

    compressed = compress_tool_results(state["tool_results"])

    prompt = apply_language_instruction(
        OBSERVE_PROMPT.format(
            symbol=state["symbol"],
            tool_results_summary=compressed,
            current_date=_now(),
        ),
        language,
    )

    messages = state.get("messages", [])
    messages.append(SystemMessage(content=prompt))
    response = llm.invoke(messages)
    messages.append(response)

    content = response.content if hasattr(response, "content") else str(response)
    decision, missing, reasoning = _parse_observe_response(content)

    steps[-1]["status"] = "done"

    if decision == "more":
        steps[-1]["detail"] = f"Need more data — re-planning: {reasoning}"
        steps.append({
            "step_type": "planning",
            "status": "pending",
            "message": f"Re-planning{f' (need: {chr(44).join(missing)})' if missing else ''}...",
            "detail": reasoning,
        })
        return {
            "messages": messages,
            "steps": steps,
            "next_action": "plan",
        }

    steps[-1]["detail"] = f"Data sufficient — {reasoning}" if reasoning else "Data sufficient — ready to synthesize"
    return {
        "messages": messages,
        "steps": steps,
        "next_action": "synthesize",
    }


def synthesize_node(state: AgentState) -> dict:
    steps: list[ReasoningStep] = state.get("steps", [])
    steps.append({
        "step_type": "synthesizing",
        "status": "active",
        "message": "Computing analytics dashboard...",
    })

    llm = _build_llm(state)

    # Compute Bloomberg-style analytics from raw tool results
    asset_data_str = ""
    price_history_str = ""
    for r in state["tool_results"]:
        if r["tool"] in ("fetch_market_data", "fetch_asset_data"):
            asset_data_str = r.get("data", {}).get("full_result", "")
        elif r["tool"] == "fetch_price_history":
            price_history_str = r.get("data", {}).get("full_result", "")

    symbol = state["symbol"]
    language = state.get("language", "en")
    cache = get_cache()
    cache_key = f"{symbol}:{language}"
    cached = cache.get(cache_key)

    if cached and cached.get("asset_data") == asset_data_str and cached.get("price_history") == price_history_str:
        dashboard = cached["dashboard"]
    else:
        from agent_service.app.tools.market_utils import detect_market as _dm
        try:
            market_info = _dm(symbol)
        except Exception:
            market_info = None
        analytics = compute_enriched_analytics(symbol, asset_data_str, price_history_str, language, market_info)
        dashboard = format_analytics_dashboard(analytics, symbol, language)
        cache.set(cache_key, {
            "asset_data": asset_data_str,
            "price_history": price_history_str,
            "dashboard": dashboard,
        }, ttl=300)

    # Use full results for final report (not compressed)
    tool_results_full = "\n\n".join(
        r.get("data", {}).get("full_result", r["summary"])
        for r in state["tool_results"]
    )

    # Inject analytics dashboard between raw data and instructions
    enriched_data = tool_results_full + "\n\n" + dashboard

    prompt = build_synthesize_prompt(
        symbol=symbol,
        enriched_data=enriched_data,
        current_date=_now(),
        language=language,
        market_info=market_info,
    )

    steps[-1]["status"] = "done"
    steps[-1]["detail"] = "Analytics computed — writing report"
    steps.append({
        "step_type": "synthesizing",
        "status": "active",
        "message": "Writing analysis report...",
    })

    messages = state.get("messages", [])
    messages.append(SystemMessage(content=prompt))
    response = llm.invoke(messages)

    report = response.content if hasattr(response, "content") else str(response)

    steps[-1]["status"] = "done"
    steps[-1]["detail"] = "Report complete"

    return {
        "final_report": report,
        "messages": messages,
        "steps": steps,
        "next_action": "done",
    }


MAX_ITERATIONS = 3


def decide_after_plan(state: AgentState) -> Literal["execute_tools", "observe"]:
    """Skip execute_tools if the plan is empty."""
    if state.get("next_action") == "observe":
        return "observe"
    return "execute_tools"


def decide_next(state: AgentState) -> Literal["collect_core_data", "plan", "synthesize", "__end__"]:
    action = state.get("next_action", "synthesize")
    # Allow core data re-collection past MAX_ITERATIONS — collect_core_data_node
    # enforces its own retry limit via core_retries to prevent infinite loops.
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        if action == "collect_core_data" and state.get("core_retries", 0) < MAX_CORE_RETRIES:
            return "collect_core_data"
        return "synthesize"
    if action == "collect_core_data":
        return "collect_core_data"
    if action == "plan":
        return "plan"
    if action == "synthesize":
        return "synthesize"
    return "__end__"
