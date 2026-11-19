"""MCP server: authoritative EGP exchange-rate tool for CapitalTrust Securities.

Run standalone over stdio (``python mcp_egp_server.py``); ``mcp_tools.py`` launches it as a
subprocess and talks to it through the MCP client. Architect: NileTech Cortex.

Modes (env ``EGP_RATE_MODE``):
  * ``mock`` (default) - deterministic, offline, clearly labelled ``CBE_MOCK``.
  * ``live``           - tries a public FX endpoint (``EGP_LIVE_URL``); on ANY failure it falls
                         back to the mock and says so in ``source`` - it never invents a rate.
"""
import datetime
import os

from mcp.server.fastmcp import FastMCP

# NOTE: the starter passed ``description=...``; FastMCP has no such argument (it is ``instructions``).
mcp = FastMCP(
    name="CapitalTrust_EGP_Oracle",
    instructions="MCP server providing authoritative financial data tools for CapitalTrust Securities.",
    log_level="WARNING",
)

# Indicative mock quotes: EGP per 1 unit of base currency. 47.85 USD is the reference value used
# in schemas/a2a_message_schema.json. They are NOT market data.
_MOCK_EGP_PER_UNIT = {"USD": 47.85, "EUR": 51.30, "GBP": 60.40, "SAR": 12.76, "AED": 13.03}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _try_live(base: str) -> float | None:
    """Best-effort live fetch. Returns None on any problem (caller falls back to the mock)."""
    url = os.getenv("EGP_LIVE_URL", "https://open.er-api.com/v6/latest/{base}").format(base=base)
    try:
        import requests

        r = requests.get(url, timeout=5)
        r.raise_for_status()
        rate = float(r.json()["rates"]["EGP"])
        return rate if rate > 0 else None
    except Exception:
        return None


@mcp.tool()
def fetch_egp_rate(base_currency: str = "USD") -> dict:
    """
    Retrieves the current exchange rate for the Egyptian Pound (EGP) against the specified base currency.
    As per Central Bank of Egypt (CBE) monitoring requirements, accuracy is critical for avoiding
    portfolio rebalancing delays.

    Args:
        base_currency (str): The currency ticker to compare against EGP (e.g., 'USD', 'EUR', 'GBP'). Defaults to 'USD'.

    Returns:
        dict: Exchange rate details including 'base', 'target', 'rate' (EGP per 1 unit of base),
        'source' and 'last_updated' timestamp.
    """
    base = (base_currency or "").strip().upper()
    if len(base) != 3 or not base.isalpha():
        raise ValueError(f"base_currency must be a 3-letter ISO code, got {base_currency!r}")
    if base == "EGP":
        raise ValueError("base_currency cannot be EGP")

    if os.getenv("EGP_RATE_MODE", "mock").lower() == "live":
        live = _try_live(base)
        if live is not None:
            return {"base": base, "target": "EGP", "rate": round(live, 4), "source": "LIVE_PUBLIC_API",
                    "last_updated": _now()}
        source_note = "CBE_MOCK (live fetch failed; fell back)"
    else:
        source_note = "CBE_MOCK"

    if base not in _MOCK_EGP_PER_UNIT:
        raise ValueError(f"No mock quote for {base}; supported: {sorted(_MOCK_EGP_PER_UNIT)}")
    return {"base": base, "target": "EGP", "rate": _MOCK_EGP_PER_UNIT[base], "source": source_note,
            "last_updated": _now()}


if __name__ == "__main__":
    # Standard entry point to run the FastMCP server via stdio
    mcp.run()
