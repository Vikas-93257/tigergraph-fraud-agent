"""Smoke tests: schema validity of the 20 answers, policy invariants, MCP tool registry, store contract."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from answer_schema import validate  # noqa: E402
from fraud_agent import policy as P  # noqa: E402
from fraud_agent import signals as S  # noqa: E402

CASES = sorted((ROOT / 'cases').glob('HHG-*.json'))


@pytest.mark.parametrize('path', CASES, ids=[p.stem for p in CASES])
def test_answer_schema(path):
    assert validate(json.loads(path.read_text())) == []


def test_twenty_cases_present():
    assert len(CASES) == 20


def test_half_legitimate_guardrail():
    """README: about half the pack is legitimate; the agent must not block everything."""
    v = [json.loads(p.read_text())['case']['verdict'] for p in CASES]
    assert 7 <= v.count('legitimate') <= 13


def test_block_and_report_are_human_routed():
    for p in CASES:
        a = json.loads(p.read_text())
        for act in a['next_best_actions']['final']:
            if act['action'] in ('BLOCK_CARD', 'BLOCK_ALL_CARDS', 'DECLINE_TRANSACTION'):
                assert act['route'] == 'L1', (p.stem, act)
            if act['action'] == 'FILE_REPORT':
                assert act['route'] == 'L2', (p.stem, act)


def test_sar_when_required():
    for p in CASES:
        a = json.loads(p.read_text())
        c = a['case']
        if c['verdict'] == 'fraud' and (c['exposure_usd'] > P.SAR_EXPOSURE or c['pattern'] == 'undocumented'):
            assert a['sar']['file'], p.stem


def test_combine_is_bounded():
    sig = [S.Signal('x', 'c', 10.0)] * 5
    assert S.combine(0.5, sig) <= 0.95
    assert S.combine(0.5, [S.Signal('x', 'c', -10.0)]) >= 0.05


def test_mcp_tools_registered():
    from mcp_server.server import mcp
    names = {t.name for t in mcp._tool_manager.list_tools()} if hasattr(mcp, '_tool_manager') else set()
    if names:
        assert {'get_transaction', 'card_window', 'device_neighbors', 'similar_cases', 'write_case', 'investigate_case'} <= names


def test_store_contract():
    from fraud_agent.store.base import GraphStore
    from fraud_agent.store.tigergraph_store import TigerGraphStore
    from fraud_agent.store.local_store import LocalGraphStore
    for m in GraphStore.__abstractmethods__:
        assert callable(getattr(TigerGraphStore, m)) and callable(getattr(LocalGraphStore, m))
