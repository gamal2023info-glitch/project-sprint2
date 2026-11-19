"""Client side of the MCP exchange-rate tool.

Starts ``mcp_egp_server.py`` as an MCP stdio server, performs the MCP handshake / ``tools/list`` and
exposes ``fetch_egp_rate`` as a LangChain ``StructuredTool`` so that ``workflow_graph.py`` can attach
it with ``llm.bind_tools([...])``. Every call really goes through the MCP protocol (one short-lived
stdio session per call - simple and robust inside Jupyter).
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

SERVER_PATH = Path(__file__).with_name("mcp_egp_server.py")
# The MCP stdio client redirects the server's stderr to ``errlog``; its default (sys.stderr) has no real file
# descriptor inside Jupyter on Windows ("UnsupportedOperation: fileno"), so we always pass a real file.
SERVER_LOG = Path(tempfile.gettempdir()) / "mcp_egp_server_stderr.log"


def _params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])


def _run(coro_factory):
    """Run an async MCP interaction to completion from sync code, even inside a running loop (Jupyter)."""
    box: dict[str, Any] = {}

    def worker():
        try:
            box["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread
            box["error"] = exc

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    if "error" in box:
        err = box["error"]
        while isinstance(err, BaseExceptionGroup) and len(err.exceptions) == 1:   # anyio wraps errors in a TaskGroup
            err = err.exceptions[0]
        raise err
    return box["value"]


async def _alist_tools() -> list[dict]:
    with open(SERVER_LOG, "a", encoding="utf-8") as errlog:
        async with stdio_client(_params(), errlog=errlog) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                res = await session.list_tools()
                return [{"name": t.name, "description": (t.description or "").strip().splitlines()[0],
                         "input_schema": t.inputSchema} for t in res.tools]


async def _acall(name: str, args: dict) -> dict:
    with open(SERVER_LOG, "a", encoding="utf-8") as errlog:
        async with stdio_client(_params(), errlog=errlog) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                res = await session.call_tool(name, args)
                if res.isError:
                    text = " ".join(getattr(c, "text", "") for c in res.content)
                    raise RuntimeError(f"MCP tool {name} failed: {text}")
                if getattr(res, "structuredContent", None):
                    sc = res.structuredContent
                    return sc.get("result", sc) if isinstance(sc, dict) else sc
                return json.loads(res.content[0].text)


def list_mcp_tools() -> list[dict]:
    """tools/list against the live MCP server (proof the tool is registered)."""
    return _run(_alist_tools)


def fetch_egp_rate_via_mcp(base_currency: str = "USD") -> dict:
    """Call the MCP tool and return the validated payload."""
    out = _run(lambda: _acall("fetch_egp_rate", {"base_currency": base_currency}))
    for key in ("base", "target", "rate", "source", "last_updated"):
        if key not in out:
            raise ValueError(f"MCP tool returned malformed payload (missing {key!r}): {out}")
    return out


class _FxArgs(BaseModel):
    base_currency: str = Field(default="USD", description="3-letter ISO code to quote against EGP, e.g. USD")


def get_exchange_rate_tool() -> StructuredTool:
    """LangChain tool wrapping the MCP ``fetch_egp_rate`` tool - pass it to ``llm.bind_tools``."""
    return StructuredTool.from_function(
        func=lambda base_currency="USD": json.dumps(fetch_egp_rate_via_mcp(base_currency)),
        name="fetch_egp_rate",
        description=("Fetch the current EGP exchange rate (EGP per 1 unit) for a base currency such as USD "
                     "from the CapitalTrust MCP server. Returns JSON with rate, source and last_updated."),
        args_schema=_FxArgs,
    )


if __name__ == "__main__":
    print(json.dumps(list_mcp_tools(), indent=2))
    print(fetch_egp_rate_via_mcp("USD"))
    