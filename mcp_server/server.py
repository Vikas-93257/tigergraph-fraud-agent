"""MCP server exposing the fraud graph tools + the investigation agent.

Two ways to use it:

1. **Bundled server (this file)** – `python -m mcp_server.server` (stdio) or `python -m mcp_server.server --http 8765`
   (streamable HTTP). Every tool wraps a named GSQL query (graph/queries/investigation.gsql) through the active backend
   (`GRAPH_BACKEND=tigergraph|local`), so an MCP client (Claude Desktop, Cursor, the LangGraph agent, the UI) can drive an
   investigation step by step, or call `investigate_case` to run the whole LangGraph workflow.

2. **Official TigerGraph MCP** – https://github.com/tigergraph/tigergraph-mcp exposes generic `run_query` /
   `get_schema` tools over the same cluster. `mcp_server/tigergraph_mcp_client.py` shows the agent calling our installed
   queries through that server instead (identical query names / params), so the two are interchangeable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as FastMCP  # noqa: E402
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP  # noqa: E402

from fraud_agent import policy as P  # noqa: E402
from fraud_agent.tools import store  # noqa: E402

mcp = FastMCP("fraud-graph", instructions=(
    "Graph tools for card-fraud investigation on the HHGOA dataset (TigerGraph FraudGraph). "
    "Start with get_transaction, then card_window / card_profile, then hub queries (device_neighbors, region_history, "
    "recurring_match, small_auth_run), then memory (similar_cases, prior_cases_for_customer) and policy retrieval. "
    "Finish with write_case. Or call investigate_case to run the full agent."))


def _j(x: Any) -> str:
    return json.dumps(x, default=str, indent=1)


# ------------------------------------------------------------------------------------------------ graph tools
@mcp.tool()
def get_transaction(txn_id: int) -> str:
    """Transaction vertex with its Card, Customer, DeviceProfile and any closed case it belongs to."""
    return _j(store().get_transaction(txn_id))


@mcp.tool()
def card_window(card_id: str, center_txn_id: int, hours_before: float = 48, hours_after: float = 48, as_of: str | None = None) -> str:
    """Transactions on the card around the flagged one (NEXT/PREV edge walk), capped at as_of."""
    return _j(store().card_window(card_id, center_txn_id, hours_before, hours_after, as_of=as_of))


@mcp.tool()
def card_profile(card_id: str, before_txn_id: int) -> str:
    """Behavioural baseline strictly before the flagged transaction: amounts, products, regions, devices, channel mix."""
    return _j(store().card_profile(card_id, before_txn_id))


@mcp.tool()
def device_neighbors(device_id: str, start: str, end: str) -> str:
    """Other cards that used the same DeviceProfile in [start, end], proxy/New shares and closed cases on the device."""
    return _j(store().device_neighbors(device_id, start, end))


@mcp.tool()
def region_history(card_id: str, region: str, before_txn_id: int, as_of: str | None = None) -> str:
    """Has this card been billed in this region before? (out_of_region_use / travel test)."""
    return _j(store().region_history(card_id, region, before_txn_id, as_of=as_of))


@mcp.tool()
def recurring_match(card_id: str, txn_id: int, as_of: str | None = None) -> str:
    """Does the amount/product recur monthly on this card (R7 recurring-charge test)?"""
    return _j(store().recurring_match(card_id, txn_id, as_of=as_of))


@mcp.tool()
def small_auth_run(card_id: str, txn_id: int, as_of: str | None = None) -> str:
    """Card-testing test (R5): small authorizations within an hour followed by a larger purchase."""
    return _j(store().small_auth_run(card_id, txn_id, as_of=as_of))


@mcp.tool()
def customer_fraud_rate(customer_id: str, as_of: str | None = None) -> str:
    """Share of the customer's prior transactions that ended in confirmed fraud cases (base rate)."""
    return _j(store().customer_fraud_rate(customer_id, as_of=as_of))


@mcp.tool()
def pattern_peers(kind: str, start: str, end: str, exclude_customer: str | None = None) -> str:
    """Cards showing the same coordinated signature (kind = sub500_burst | anon_proxy_new_device) in the window (R9)."""
    return _j(store().pattern_peers(kind, start, end, exclude_customer))


@mcp.tool()
def prior_cases_for_customer(customer_id: str) -> str:
    """Closed cases and agent cases on the customer's cards."""
    return _j(store().prior_cases_for_customer(customer_id))


@mcp.tool()
def similar_cases(query_text: str, pattern_hint: str | None = None, customer_id: str | None = None, k: int = 5) -> str:
    """Vector/keyword retrieval over closed-case narratives and agent case memory (GraphRAG)."""
    return _j(store().similar_cases(query_text, pattern_hint, customer_id, k))


@mcp.tool()
def retrieve_policy(query_text: str, k: int = 4) -> str:
    """Retrieve the fraud-policy / typology chunks relevant to the question."""
    return _j(store().retrieve_policy(query_text, k))


@mcp.tool()
def write_case(case_json: str) -> str:
    """Write an AgentCase vertex (+ INVOLVES / ON_CARD / CONNECTED_TO / USED_DEVICE edges). Returns graph case id."""
    return _j({'graph_case_id': store().write_case(json.loads(case_json))})


@mcp.tool()
def alerts_in_period(start: str, end: str, min_score: float = 0.85, limit: int = 50) -> str:
    """Monitoring helper: highest-risk transactions in a period (drives the optional monitoring track)."""
    return _j(store().alerts_in_period(start, end, min_score, limit))


# ------------------------------------------------------------------------------------------------ agent
@mcp.tool()
def investigate_case(case_id: str) -> str:
    """Run the full LangGraph investigation for a case_pack case id (e.g. HHG-006) and return the answer JSON."""
    from run_cases import load_case_pack
    from fraud_agent.agent import investigate
    rows = {r['case_id']: r for r in load_case_pack()}
    if case_id not in rows:
        return _j({'error': f'unknown case {case_id}', 'known': sorted(rows)})
    answer, _ = investigate(rows[case_id])
    return _j(answer)


@mcp.tool()
def investigate_transaction(txn_id: int, trigger_type: str = 'analyst_request', note: str = '') -> str:
    """Investigate any transaction id ad hoc (monitoring track): builds a synthetic trigger and runs the agent."""
    from fraud_agent.agent import investigate
    t = store().get_transaction(txn_id)
    if not t:
        return _j({'error': 'unknown transaction'})
    trig = {'case_id': f'ADHOC-{txn_id}', 'trigger_type': trigger_type, 'customer_id': t['customer_id'], 'card_id': t['card_id'],
            'flagged_txn_id': int(txn_id), 'opened_at': t['ts'], 'analyst_note': note or None, 'risk_score': t.get('risk_score')}
    answer, _ = investigate(trig)
    return _j(answer)


@mcp.resource("policy://fraud-policy")
def fraud_policy() -> str:
    """The bank's fraud policy, typology catalogue and routing rules."""
    from fraud_agent.documents import POLICY_CHUNKS
    return "\n\n".join(f"## {c['title']}\n{c['text']}" for c in POLICY_CHUNKS)


@mcp.resource("policy://actions")
def actions() -> str:
    """Allowed next-best-actions and their default routes."""
    return _j({'actions': sorted(P.ACTIONS), 'auto_routes': sorted(P.AUTO), 'human_routes': sorted(P.ACTIONS - P.AUTO)})


if __name__ == '__main__':
    if '--http' in sys.argv:
        port = int(sys.argv[sys.argv.index('--http') + 1]) if len(sys.argv) > sys.argv.index('--http') + 1 else 8765
        try:
            mcp.settings.host = '0.0.0.0'; mcp.settings.port = port
        except Exception:
            pass
        mcp.run(transport='streamable-http')
    else:
        mcp.run()
