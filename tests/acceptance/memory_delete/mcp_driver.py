"""Drive everos-mcp 0.1.1 through its own MCP client (run in its own env:
``uv run --no-project --with everos-mcp==0.1.1 python -I mcp_driver.py``).

``full <token>``: add_memory x3 (wait), print ADDED, wait for a line on
stdin, search_memory (before), forget_session, search_memory (after).
``busy``: forget_session only. Every tool result is printed as
``<TOOL>: <isError> <text>``.
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content).replace("\n", " ")


async def main(mode: str, token: str) -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "everos_mcp.server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()
        if mode == "full":
            for i in range(3):
                r = await s.call_tool(
                    "add_memory",
                    {
                        "user_message": (
                            f"Please remember {token}: my locker code"
                            f" number {i} is {token}{i}."
                        ),
                        "assistant_message": f"Noted the {token} locker code {i}.",
                        "wait": True,
                    },
                )
                print(f"ADD: {r.isError} {_text(r)}", flush=True)
            print("ADDED", flush=True)
            sys.stdin.readline()
            r = await s.call_tool("search_memory", {"query": f"{token} locker code"})
            print(f"SEARCH_BEFORE: {r.isError} {_text(r)}", flush=True)
        r = await s.call_tool("forget_session", {})
        print(f"FORGET: {r.isError} {_text(r)}", flush=True)
        if mode == "full":
            r = await s.call_tool("search_memory", {"query": f"{token} locker code"})
            print(f"SEARCH: {r.isError} {_text(r)}", flush=True)


asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))
