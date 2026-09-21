"""Tool layer the agent calls. Every call is counted (reported as ``tool_calls`` in the answer file) and logged
so the UI can replay the investigation step by step. The same functions are exported through the MCP server.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .store import get_store, GraphStore

_STORE: GraphStore | None = None


def store() -> GraphStore:
    global _STORE
    if _STORE is None:
        _STORE = get_store()
    return _STORE


@dataclass
class ToolLog:
    calls: list[dict] = field(default_factory=list)

    def record(self, name: str, args: dict, result: Any, ms: float):
        self.calls.append({'step': len(self.calls) + 1, 'tool': name, 'args': args, 'ms': round(ms, 1),
                           'result_preview': _preview(result)})

    @property
    def count(self):
        return len(self.calls)


def _preview(x, n=240):
    s = str(x)
    return s if len(s) <= n else s[:n] + '…'


class Toolbox:
    """Bound tool set for one investigation. Mirrors the MCP tool names 1:1."""

    def __init__(self, log: ToolLog | None = None, as_of: str | None = None):
        self.log = log or ToolLog()
        self.s = store()
        self.as_of = as_of  # investigation clock: nothing after this timestamp is visible

    def _call(self, name: str, fn: Callable, **kw):
        t = time.time()
        res = fn(**kw)
        self.log.record(name, kw, res, (time.time() - t) * 1000)
        return res

    # graph tools -------------------------------------------------------------------------------
    def get_transaction(self, txn_id): return self._call('get_transaction', self.s.get_transaction, txn_id=int(txn_id))
    def card_window(self, card_id, center_txn_id, hours_before=48, hours_after=48):
        return self._call('card_window', self.s.card_window, card_id=card_id, center_txn_id=int(center_txn_id),
                          hours_before=hours_before, hours_after=hours_after, as_of=self.as_of)
    def card_profile(self, card_id, before_txn_id): return self._call('card_profile', self.s.card_profile, card_id=card_id, before_txn_id=int(before_txn_id))
    def device_neighbors(self, device_id, start, end): return self._call('device_neighbors', self.s.device_neighbors, device_id=device_id, start=start, end=end)
    def region_history(self, card_id, region, before_txn_id): return self._call('region_history', self.s.region_history, card_id=card_id, region=region, before_txn_id=int(before_txn_id), as_of=self.as_of)
    def recurring_match(self, card_id, txn_id): return self._call('recurring_match', self.s.recurring_match, card_id=card_id, txn_id=int(txn_id), as_of=self.as_of)
    def small_auth_run(self, card_id, txn_id): return self._call('small_auth_run', self.s.small_auth_run, card_id=card_id, txn_id=int(txn_id), as_of=self.as_of)
    def customer_fraud_rate(self, customer_id): return self._call('customer_fraud_rate', self.s.customer_fraud_rate, customer_id=customer_id, as_of=self.as_of)
    def card_txns_on_device(self, card_id, device_id):
        return self._call('card_txns_on_device', self.s.card_txns_on_device, card_id=card_id, device_id=device_id, as_of=self.as_of)
    def pattern_peers(self, kind, start, end, exclude_customer=None):
        return self._call('pattern_peers', self.s.pattern_peers, kind=kind, start=start, end=end, exclude_customer=exclude_customer)
    # memory / GraphRAG ---------------------------------------------------------------------------
    def prior_cases_for_customer(self, customer_id): return self._call('prior_cases_for_customer', self.s.prior_cases_for_customer, customer_id=customer_id)
    def similar_cases(self, query_text, pattern_hint=None, customer_id=None, k=5):
        return self._call('similar_cases', self.s.similar_cases, query_text=query_text, pattern_hint=pattern_hint, customer_id=customer_id, k=k)
    def retrieve_policy(self, query_text, k=4): return self._call('retrieve_policy', self.s.retrieve_policy, query_text=query_text, k=k)
    def write_case(self, case): return self._call('write_case', self.s.write_case, case=case)

    # controlled evidence-gathering actions (simulated; policy §5) --------------------------------
    def request_evidence(self, kind: str, prompt: str, assumed_response: str):
        return self._call(f'evidence_request:{kind}', lambda **kw: {'type': kind, 'prompt': prompt, 'assumed_response': assumed_response},
                          kind=kind, prompt=prompt, assumed_response=assumed_response)

    # mock downstream systems (policy: may be stubbed) ---------------------------------------------
    def execute_action(self, action: str, route: str, payload: dict):
        executed = route == 'auto'
        return self._call('execute_action', lambda **kw: {'action': action, 'route': route,
                                                          'status': 'executed' if executed else 'pending_human_approval',
                                                          'payload': payload}, action=action, route=route, payload=payload)
