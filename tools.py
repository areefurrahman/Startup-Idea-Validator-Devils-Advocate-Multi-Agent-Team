import re

from crewai.tools import tool
from ddgs import DDGS

MAX_SEARCHES = 4

# Small shared state so the app can reset it per run and read the results back.
_state = {"searches": 0, "context": "", "economics": None}


def reset_tool_state(idea: str = "", market: str = "") -> None:
    """Call this once at the start of every run."""
    _state["searches"] = 0
    _state["context"] = f"{idea} {market}".strip()
    _state["economics"] = None


def get_search_count() -> int:
    return _state["searches"]


def get_last_economics():
    return _state["economics"]


def _keywords(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


def _with_context(query: str) -> str:
    """If the query lost the product context, add the start of the idea back."""
    ctx = _state["context"]
    if ctx and not (_keywords(query) & _keywords(ctx)):
        query = f"{query} {' '.join(ctx.split()[:8])}"
    return query


@tool("web_search_tool")
def web_search_tool(query: str) -> str:
    """Search the live web for competitors, pricing and market data.
    Input: a FULL search query that includes the product type and market,
    for example 'AI resume tailoring tool for graduates pricing competitors'.
    Never search a single word. You may use this tool at most 4 times."""
    if len(query.split()) < 3:
        return (
            "REJECTED: the query is too short. Write a full query with the product "
            "type and market, for example 'subscription specialty coffee box US pricing'."
        )
    if _state["searches"] >= MAX_SEARCHES:
        return (
            "SEARCH LIMIT REACHED (4 of 4 used). Do not search again. "
            "Write your report now from the results you already have."
        )
    _state["searches"] += 1
    query = _with_context(query)

    results, last_error = [], None
    for backend in ("auto", "duckduckgo", "brave"):  # fallback chain
        try:
            results = DDGS().text(query, max_results=5, backend=backend)
            if results:
                break
        except Exception as e:  # rate limit, network, bad backend
            last_error = e

    if not results:
        return (
            f"No live results for '{query}' (error: {last_error}). Continue with general "
            "knowledge and clearly label anything you could not verify as 'unverified'."
        )

    lines = [f"Search {_state['searches']}/{MAX_SEARCHES}: {query}"]
    for r in results:
        title = r.get("title", "")
        link = r.get("href", "")
        body = (r.get("body") or "")[:250]  # short, to save tokens
        lines.append(f"- {title} | {link}\n  {body}")
    return "\n".join(lines)


@tool("unit_economics_calculator")
def unit_economics_calculator(
    monthly_price: float,
    cac: float,
    monthly_fixed_cost: float,
    monthly_churn_pct: float = 5.0,
    gross_margin_pct: float = 80.0,
) -> str:
    """Calculate break-even customers, LTV:CAC ratio and a financial risk rating.
    All money is in USD. monthly_price = what one customer pays per month.
    cac = cost to win one customer. monthly_fixed_cost = salaries, tools, hosting
    per month. monthly_churn_pct (default 5) = percent of customers lost each
    month. gross_margin_pct (default 80) = percent of the price left after
    direct costs. Only change the two defaults if you have a good reason."""
    if monthly_price <= 0 or gross_margin_pct <= 0:
        _state["economics"] = {
            "breakeven": None, "ltv_cac": None, "risk": "CRITICAL",
        }
        return "RISK RATING: CRITICAL. Price or gross margin is zero or negative, so every customer loses money."

    margin = monthly_price * gross_margin_pct / 100
    churn = max(monthly_churn_pct, 0.1) / 100  # avoid divide by zero
    lifetime_months = 1 / churn
    ltv = margin * lifetime_months
    ltv_cac = ltv / cac if cac > 0 else float("inf")
    payback = cac / margin
    breakeven = monthly_fixed_cost / margin
    acquisition_outlay = breakeven * cac

    points = 0
    points += 3 if ltv_cac < 1 else 2 if ltv_cac < 3 else 1 if ltv_cac < 5 else 0
    points += 3 if payback > 18 else 2 if payback > 12 else 1 if payback > 6 else 0
    points += 2 if breakeven > 500 else 1 if breakeven > 150 else 0
    points += 1 if monthly_churn_pct >= 8 else 0
    risk = "CRITICAL" if points >= 6 else "HIGH" if points >= 4 else "MODERATE" if points >= 2 else "LOW"

    _state["economics"] = {
        "breakeven": round(breakeven),
        "ltv_cac": round(ltv_cac, 2) if ltv_cac != float("inf") else None,
        "risk": risk,
    }

    ltv_cac_text = "infinite (CAC is 0)" if ltv_cac == float("inf") else f"{ltv_cac:.2f}"
    return (
        f"Assumptions: churn {monthly_churn_pct}%/month, gross margin {gross_margin_pct}%\n"
        f"Gross profit per customer per month: ${margin:,.2f}\n"
        f"Customer lifetime: {lifetime_months:.1f} months | LTV: ${ltv:,.0f}\n"
        f"LTV:CAC ratio: {ltv_cac_text} (healthy is 3 or more)\n"
        f"CAC payback: {payback:.1f} months (healthy is under 12)\n"
        f"Break-even customers needed: {breakeven:,.0f}\n"
        f"Monthly burn with zero customers: ${monthly_fixed_cost:,.0f}\n"
        f"Cash needed to buy the break-even customers: ${acquisition_outlay:,.0f}\n"
        f"RISK RATING: {risk}"
    )
