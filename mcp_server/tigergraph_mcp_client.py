"""Drive the agent's graph queries through the *official* TigerGraph MCP server (github.com/tigergraph/tigergraph-mcp).

    pip install tigergraph-mcp            # official server
    export TG_HOST=... TG_USERNAME=... TG_PASSWORD=...
    python mcp_server/tigergraph_mcp_client.py 3476682

The official server exposes generic tools (schema inspection, `run_query` / installed-query execution, vertex/edge
CRUD). Because every agent tool is an installed GSQL query with a stable name and parameter list, the same
investigation runs through that server unchanged: this module is a thin ``GraphStore`` that forwards each named
query via MCP and reuses ``TigerGraphStore``'s response reshaping.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from fraud_agent.store.tigergraph_store import TigerGraphStore  # noqa: E402


class MCPBackedTigerGraphStore(TigerGraphStore):
    """Same reshaping as TigerGraphStore, but query execution goes through the official TigerGraph MCP server."""

    def __init__(self, session: ClientSession, loop: asyncio.AbstractEventLoop, query_tool: str = 'run_installed_query'):
        # deliberately skip TigerGraphStore.__init__ (no direct REST connection)
        self.session, self.loop, self.query_tool = session, loop, query_tool
        self.has_vector = False
        self._text_index = None
        from fraud_agent import config
        self.memory_file = Path(config.MEMORY_FILE)

    def _q(self, name: str, **params):
        async def call():
            res = await self.session.call_tool(self.query_tool, {'graph_name': os.getenv('TG_GRAPH', 'FraudGraph'),
                                                                 'query_name': name, 'params': params})
            return json.loads(res.content[0].text)
        out = self.loop.run_until_complete(call())
        return out.get('results', out) if isinstance(out, dict) else out


async def _demo(txn_id: int):
    params = StdioServerParameters(command='tigergraph-mcp', args=[], env=dict(os.environ))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print('official TigerGraph MCP tools:', [t.name for t in tools.tools])
            query_tool = next((t.name for t in tools.tools if 'query' in t.name and 'run' in t.name), 'run_installed_query')
            loop = asyncio.get_event_loop()
            st = MCPBackedTigerGraphStore(s, loop, query_tool)
            print(json.dumps(st.get_transaction(txn_id), indent=1, default=str))


if __name__ == '__main__':
    asyncio.run(_demo(int(sys.argv[1]) if len(sys.argv) > 1 else 3476682))
