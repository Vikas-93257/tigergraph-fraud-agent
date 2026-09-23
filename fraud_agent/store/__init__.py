"""Graph access layer.

Two interchangeable backends expose the *same* named queries:

* ``TigerGraphStore`` – calls installed GSQL queries (graph/queries/*.gsql) over pyTigerGraph / REST++.
* ``LocalGraphStore`` – in-process pandas implementation of the identical queries so the agent, UI and the 20-case
  benchmark run without a live cluster. It is also the reference implementation the GSQL was written against.

Pick with ``GRAPH_BACKEND=tigergraph|local``.
"""
from .base import GraphStore  # noqa: F401


def get_store():
    from .. import config
    if config.GRAPH_BACKEND == "tigergraph":
        from .tigergraph_store import TigerGraphStore
        return TigerGraphStore()
    from .local_store import LocalGraphStore
    return LocalGraphStore()
