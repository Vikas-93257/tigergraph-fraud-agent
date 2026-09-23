"""Query contract shared by both graph backends.

Every method maps 1:1 to a GSQL query in ``graph/queries/investigation.gsql``. Names are used verbatim as
``evidence.ref`` in the answer files (``query:card_window(...)``), so keep them stable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class GraphStore(ABC):
    """Abstract graph store. All returns are plain dicts / lists so they serialize to JSON for MCP."""

    @abstractmethod
    def get_transaction(self, txn_id: int) -> dict | None: ...

    @abstractmethod
    def card_window(self, card_id: str, center_txn_id: int, hours_before: float, hours_after: float, as_of: str | None = None) -> list[dict]:
        """Transactions on the card in [t-hours_before, t+hours_after] following NEXT edges, capped at as_of."""

    @abstractmethod
    def card_profile(self, card_id: str, before_txn_id: int) -> dict:
        """Behavioural baseline for the card strictly *before* the flagged transaction."""

    @abstractmethod
    def device_neighbors(self, device_id: str, start: str, end: str) -> dict:
        """Other cards that used the same DeviceProfile in [start, end], with proxy flags and prior-case links."""

    @abstractmethod
    def region_history(self, card_id: str, region, before_txn_id: int, as_of: str | None = None) -> dict: ...

    @abstractmethod
    def recurring_match(self, card_id: str, txn_id: int, amount_tol: float = 0.6, as_of: str | None = None) -> dict: ...

    @abstractmethod
    def small_auth_run(self, card_id: str, txn_id: int, hours: float = 1.0, max_amt: float = 5.0, as_of: str | None = None) -> dict: ...

    @abstractmethod
    def customer_fraud_rate(self, customer_id: str, as_of: str | None = None) -> dict: ...

    @abstractmethod
    def card_txns_on_device(self, card_id: str, device_id: str, as_of: str | None = None) -> list[dict]: ...

    @abstractmethod
    def pattern_peers(self, kind: str, start: str, end: str, exclude_customer: str | None = None) -> dict: ...

    @abstractmethod
    def similar_cases(self, query_text: str, pattern_hint: str | None, customer_id: str | None, k: int = 5) -> list[dict]:
        """Vector / keyword search over ClosedCase + AgentCase narratives."""

    @abstractmethod
    def prior_cases_for_customer(self, customer_id: str) -> list[dict]: ...

    @abstractmethod
    def write_case(self, case: dict) -> str:
        """Upsert an AgentCase vertex + edges. Returns graph case id."""

    @abstractmethod
    def retrieve_policy(self, query_text: str, k: int = 4) -> list[dict]: ...

    @abstractmethod
    def alerts_in_period(self, start: str, end: str, min_score: float = 0.85, limit: int = 50) -> list[dict]: ...

    def describe(self) -> dict[str, Any]:
        return {"backend": type(self).__name__}
