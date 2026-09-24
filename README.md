# Graph-native Fraud Investigation Agent — TigerGraph Agentic Fraud Investigation (HHGOA)

An autonomous investigator that turns a bank's fraud alert into a **decision with a paper trail**: it walks the
transaction graph in TigerGraph, retrieves the bank's policy and prior cases (GraphRAG), asks for evidence when the
rules say so, chooses next-best-actions with the right approval route, files a SAR when policy requires one, writes the
case back into the graph as memory — and explains every claim with the query that produced it.

```
trigger ──► gather (GSQL) ──► memory (GraphRAG) ──► assess ──► initial NBAs ─┐
                                                       ▲                      │ need evidence?
                                                       └── request_evidence ◄─┘
                                                             (simulated customer / step-up)
                       final NBAs ──► explain (summary, SAR) ──► write AgentCase to graph
```

| Judging axis | Where to look |
|---|---|
| Accuracy | `cases/HHG-0xx.json` — 20 answers, 10 fraud / 10 legitimate, validated by `answer_schema.py` |
| Next-best-actions | `fraud_agent/policy.py`, `agent.py::n_initial_decision / n_final_decision`; every action has a route + policy reason |
| Agentic design | LangGraph state machine with conditional evidence loop, budgeted stopping, tool log per case (`traces/`) |
| Innovation | supporting/contradicting/missing evidence with policy-gated evidence requests, initial-vs-final counterfactual plans, signature-level peer search across customers (`pattern_peers`), case memory later cases retrieve via TigerVector |
| Explainability | every evidence item = `{claim, source, ref: "query:<gsql_name>(...)", entity_ids}` + log-odds weight in the UI |
| Demo | `ui/` investigator console, `docs/video_script.md` |

## What's in the repo

```
fraud_agent/           the agent
  agent.py             LangGraph workflow (trigger → gather → memory → assess → decide → evidence → decide → explain → memorize)
  signals.py           evidence extractors (each returns a claim + log-odds weight + query ref) and the calibrated combiner
  policy.py            allowed actions, routing (auto / L1 / L2), thresholds ($500 escalate, $1,000 SAR, 0.30 case, 0.70 verify)
  documents.py         policy text, typology catalogue, analyst memory notes, FinCEN guidance → PolicyChunk vertices for GraphRAG
  narrate.py           summary + SAR narrative (LLM if OPENAI_API_KEY is set, deterministic template otherwise)
  tools.py             Toolbox: every graph call is logged (tool name, args, latency) → tool_calls / traces
  store/base.py        the query contract (13 named queries)
  store/tigergraph_store.py   pyTigerGraph → installed GSQL queries (+ TigerVector when available)
  store/local_store.py        identical queries on pandas/TF-IDF so the benchmark runs without a cluster
graph/
  schema.gsql          Customer, Card, Transaction, DeviceProfile, EmailDomain, BillingRegion, ClosedCase, AgentCase, PolicyChunk
  load.gsql / load.py  export CSV → loading job (python graph/load.py all)
  queries/investigation.gsql   the GSQL behind every tool (card_window, device_neighbors, recurring_match, pattern_peers_*, write_case…)
mcp_server/
  server.py            MCP server (stdio / streamable-http) exposing the 15 graph tools + investigate_case / investigate_transaction
  tigergraph_mcp_client.py   the same investigation driven through the official TigerGraph MCP server
ui/                    FastAPI + zero-dependency investigator console (graph neighbourhood, evidence weights, NBAs, SAR, tool log)
monitoring/            optional track: sweep a period, investigate the top alerts, write a period summary
cases/                 the 20 answer files        traces/  per-case state (window, profile, signals, tool log)
answer_schema.py       README-format validator    tests/   pytest (schema, policy invariants, contracts)
docs/blog.md           technical write-up         docs/video_script.md   3–5 min demo plan   docs/social_post.md
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATA_DIR=/path/to/HHGOA_IEEE          # transactions.csv, identity.csv, closed_cases_history.csv, case_pack.csv

python run_cases.py                          # all 20 → cases/*.json  (≈25 s, local backend)
python run_cases.py HHG-006                  # one case, prints the answer + tool log
python answer_schema.py && pytest -q         # validate
uvicorn ui.app:app --host 0.0.0.0 --port 8000   # UI → http://localhost:8000
python -m mcp_server.server                  # MCP over stdio (add to Claude Desktop / Cursor)
python monitoring/monitor.py --start 2016-12-01 --end 2016-12-02 --limit 8
```

### TigerGraph (Savanna or Community Edition)

```bash
cp .env.example .env            # TG_HOST, TG_USERNAME, TG_PASSWORD (or TG_SECRET), GRAPH_BACKEND=tigergraph
python graph/load.py export     # ~100 MB of CSV in graph/export (chunked; runs in 2 GB RAM)
python graph/load.py schema     # schema.gsql + load.gsql + queries → INSTALL QUERY ALL
python graph/load.py load       # loading job
python graph/load.py embed      # TigerVector embeddings (optional; TF-IDF fallback otherwise)
GRAPH_BACKEND=tigergraph python run_cases.py
```

### MCP setup (Claude Desktop / Cursor)

```json
{
  "mcpServers": {
    "fraud-investigator": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/tigergraph-fraud-agent",
      "env": {"GRAPH_BACKEND": "tigergraph", "TG_HOST": "https://<workspace>.i.tgcloud.io", "TG_SECRET": "<secret>"}
    }
  }
}
```

Tools exposed: the 15 graph tools (`get_transaction`, `card_window`, `device_neighbors`, `pattern_peers`,
`similar_cases`, `retrieve_policy`, `write_case`, …) plus `investigate_case(case_id)` and `investigate_transaction(txn_id)`.
`mcp_server/tigergraph_mcp_client.py` runs the same query names through the official
[tigergraph-mcp](https://github.com/tigergraph/tigergraph-mcp) server instead.

The agent code does not change between backends: `GraphStore` is the contract, `evidence.ref` strings are the GSQL
query names, and `TigerGraphStore` reshapes REST++ results into the same dicts `LocalGraphStore` returns. Vector
retrieval (`similar_cases`, `retrieve_policy`) uses TigerVector indexes on `ClosedCase/AgentCase/PolicyChunk.embedding`
when they exist and falls back to TF-IDF over the same vertices otherwise.

## Architecture

```
                 ┌──────────────────────────────────────────────────────────────┐
                 │  Analyst UI (FastAPI + static)   ·   MCP clients (Claude, Cursor) │
                 └───────────────┬──────────────────────────────┬───────────────┘
                                 │ REST                         │ MCP (stdio / http)
                    ┌────────────▼────────────┐    ┌────────────▼────────────┐
                    │  run_cases / monitoring │    │   mcp_server/server.py  │
                    └────────────┬────────────┘    └────────────┬────────────┘
                                 └──────────────┬───────────────┘
                                   ┌────────────▼────────────┐
                                   │   LangGraph agent       │  trigger → gather → memory → assess
                                   │   fraud_agent/agent.py  │  → initial NBA → evidence loop → final NBA
                                   └────────────┬────────────┘  → explain → memorize
                     signals.py ◄── evidence ────┤──── policy.py (final authority, routes auto/L1/L2)
                                   ┌────────────▼────────────┐
                                   │ GraphStore contract     │  15 named tools = GSQL query names
                                   │ store/base.py           │
                                   └──────┬───────────┬──────┘
                          TigerGraphStore │           │ LocalGraphStore (pandas, tests)
                                   ┌──────▼───────────▼──────┐
                                   │ TigerGraph Savanna 4.2.5│  FraudGraph: 9 vertex / 29 edge types
                                   │ 13 installed GSQL       │  TigerVector: ClosedCase · AgentCase · PolicyChunk
                                   │ tigergraph-mcp (official)│  AgentCase written back = case memory
                                   └─────────────────────────┘
```

## Evidence intelligence and control (what makes it an agent, not a scorer)

* **Supporting / contradicting / missing evidence.** Every signal is an evidence row with a signed log-odds weight:
  positive rows support the fraud hypothesis, negative rows contradict it (shown green in the UI — e.g. a known device,
  a home-region in-person purchase, a 56-month recurring charge). Missing evidence becomes an explicit
  `evidence_request` rather than a guess.
* **Evidence acquisition is policy-gated.** The agent may only ask for evidence through controlled actions
  (`VERIFY_WITH_CUSTOMER`, `STEP_UP_AUTH`) and only when policy R1/R2 says the current evidence is insufficient
  (single weak signal, or probability inside the uncertainty band). It records what it asked, after which step, and what
  it assumed — so the initial plan and the final plan are both auditable.
* **Counterfactual plan.** `next_best_actions.initial` is the plan *before* the evidence arrives, `final` is after, and
  `what_changed` states the delta (e.g. HHG-015: step-up → block + report once the customer denies the purchases).
* **Policy has the last word.** `policy.py` decides which actions are allowed, which route they take (auto / L1 / L2)
  and when a SAR is mandatory. No block or report ever executes without a human route; an LLM, if configured, only writes
  prose and cannot change an action.
* **Graph XAI.** Each claim carries `ref: query:<gsql_name>(args)` and `entity_ids`, so an analyst can click from the
  recommendation to the evidence to the exact graph path (device → 20 cards → 4 closed cases). No SHAP needed.
* **Case memory.** Verdicts are written back as `AgentCase` vertices with SIMILAR_TO / CONNECTED_TO / USED_DEVICE edges
  and a TigerVector embedding; later investigations retrieve them (HHG-005 cites HHG-003).

## How a case is investigated

1. **Trigger** — `case_pack` row; the trigger type sets the prior (risk_score 0.22, customer_report 0.55, analyst 0.40).
   The model score is *one input* (README: above 0.7 most flagged are still legitimate), weighted +0.2 log-odds at most.
2. **Gather (GSQL)** — `get_transaction`, `card_window` (NEXT/PREV walk ±48 h, capped at `opened_at`), `card_profile`
   (baseline strictly before the alert), `device_neighbors` (hub test: cards / proxy share / New share / closed cases on the device),
   `region_history`, `recurring_match`, `small_auth_run`, `customer_fraud_rate`, and — only when a coordinated signature
   is present — `pattern_peers` across *all* customers.
3. **Memory (GraphRAG)** — `prior_cases_for_customer`, `similar_cases` over 5,565 closed-case narratives + the agent's own
   cases, `retrieve_policy` over the policy/typology chunks. Retrieved passages are cited as `source: document`.
4. **Assess** — every signal is an evidence row with a log-odds weight; weights sum onto the prior and are clamped to
   [0.05, 0.95]. Pattern classification follows the typology catalogue; unknown but coordinated → `undocumented` (R9).
5. **Initial NBAs** — policy-driven (R1 verify before block on a weak signal, 3a case ≠ report, R7 recurring, R8 exposure,
   R9 undocumented, R10 never block-all). `BLOCK_CARD` routes to L1, `FILE_REPORT` to L2, everything else auto.
6. **Evidence loop** — if the plan contains VERIFY / STEP_UP, the agent issues an `evidence_request` and continues with
   the *assumed* answer. Simulation rule (documented, deterministic): strong recurring history → customer recognises the
   charge; otherwise p ≥ 0.50 → denial (R2, +2.4), p < 0.50 → confirmation (R3, −2.2). Stopping: two independent signals
   and p outside [0.15, 0.85], or budget (≤ 18 tool calls).
7. **Final NBAs + explain** — `what_changed` states the delta; SAR narrative follows FinCEN's five elements; exposure =
   sum of the episode; `connected_card_ids` / `connected_device_profiles` from hub queries.
8. **Memorize** — `write_case` creates an `AgentCase` vertex with INVOLVES / ON_CARD / CONNECTED_TO / USED_DEVICE /
   SIMILAR_TO edges. Later cases retrieve it (HHG-005 already cites HHG-003).

## Results on the 20-case pack

| case | trigger | verdict | p | pattern | exposure | SAR | final actions |
|---|---|---|---|---|---|---|---|
| HHG-001 | risk_score | legitimate | 0.05 | none | 0 |  | ALLOW, CLOSE_NO_FRAUD |
| HHG-002 | risk_score | fraud | 0.93 | card_not_present_fraud | 292.36 |  | CREATE_CASE, BLOCK_CARD |
| HHG-003 | customer_report | legitimate | 0.05 | none (recurring $49 W ×56) | 0 |  | CREATE_CASE, WARN_CUSTOMER, CLOSE_NO_FRAUD |
| HHG-004 | customer_report | fraud | 0.95 | card_not_present_new_device | 128.33 |  | CREATE_CASE, BLOCK_CARD |
| HHG-005 | risk_score | legitimate | 0.05 | none (step-up passed) | 0 |  | CREATE_CASE, CLOSE_NO_FRAUD |
| HHG-006 | customer_report | fraud | 0.95 | undocumented (4 × ~$480 in 30 min, 6 peer cards) | 1,906.07 | ✔ | CREATE_CASE, BLOCK_CARD, MONITOR_CONNECTED_CARDS, FILE_REPORT, ESCALATE |
| HHG-007 | risk_score | legitimate | 0.12 | none | 0 |  | ALLOW, CLOSE_NO_FRAUD |
| HHG-008 | customer_report | fraud | 0.95 | card_not_present_fraud (3 × $55.68) | 166.97 |  | CREATE_CASE, BLOCK_CARD |
| HHG-009 | customer_report | legitimate | 0.05 | none (recurring $30.02 S) | 0 |  | CREATE_CASE, CLOSE_NO_FRAUD |
| HHG-010 | risk_score | fraud | 0.94 | card_not_present_new_device | 1,000.03 | ✔ | CREATE_CASE, BLOCK_CARD, FILE_REPORT |
| HHG-011 | customer_report | legitimate | 0.07 | none (recurring ~$131.5) | 0 |  | CREATE_CASE, WARN_CUSTOMER, CLOSE_NO_FRAUD |
| HHG-012 | risk_score | legitimate | 0.05 | none | 0 |  | ALLOW, CLOSE_NO_FRAUD |
| HHG-013 | risk_score | fraud | 0.92 | card_not_present_new_device | 35.66 |  | CREATE_CASE, BLOCK_CARD |
| HHG-014 | analyst_request | fraud | 0.90 | undocumented (device hub, 20 cards, 4 closed cases) | 187.33 | ✔ | CREATE_CASE, BLOCK_CARD, MONITOR_CONNECTED_CARDS, FILE_REPORT, ESCALATE |
| HHG-015 | risk_score | fraud | 0.95 | card_not_present_new_device (+ in-person spike, 3 txns) | 1,699.98 | ✔ | CREATE_CASE, BLOCK_CARD, FILE_REPORT |
| HHG-016 | customer_report | fraud | 0.92 | card_not_present_new_device | 59.67 |  | CREATE_CASE, BLOCK_CARD |
| HHG-017 | risk_score | legitimate | 0.14 | none | 0 |  | ALLOW, CLOSE_NO_FRAUD |
| HHG-018 | customer_report | legitimate | 0.05 | none (recurring $39.08 W ×145) | 0 |  | CREATE_CASE, WARN_CUSTOMER, CLOSE_NO_FRAUD |
| HHG-019 | risk_score | legitimate | 0.07 | none | 0 |  | ALLOW, CLOSE_NO_FRAUD |
| HHG-020 | risk_score | fraud | 0.95 | card_not_present_new_device | 125.08 |  | CREATE_CASE, BLOCK_CARD |

10 fraud / 10 legitimate, 4 SARs, 14 tool calls per case on average, 25 s for the pack on the local backend, 0 LLM tokens
(template narrator). Regenerate with `python run_cases.py`.

## Design notes worth knowing

* **Time discipline.** Every query is capped at the trigger's `opened_at`; the agent never sees the future.
* **The score is a hint.** Six of the ten risk_score cases close as legitimate; the graph baseline decides, not the model.
* **Disputes are evidence, not verdicts.** Three customer_report cases (003/011/018) close as legitimate because the disputed
  amount recurs monthly on the customer's own card (R7) — the agent recommends WARN_CUSTOMER instead of a block.
* **Undocumented ≠ unknown.** R9 is triggered only when the signature repeats *across customers* in the same window
  (`pattern_peers` / `device_neighbors`) — which is also what earns MONITOR_CONNECTED_CARDS and the SAR.
* **Human in the loop where it costs money.** No block or report ever executes automatically; the routes are in the JSON.
* **Cheap.** No LLM is required for correctness; the LLM (if configured) only writes prose.

## Not done / honest limits

* Evidence requests are simulated with a deterministic rule (README allows it); a real deployment plugs the customer
  channel into `Toolbox.request_evidence`.
* Card ids for customers not present in `case_pack` / closed cases are assumed `<customer>-K1`.
* The submitted `cases/*.json` were produced with `GRAPH_BACKEND=tigergraph` against a TigerGraph Savanna 4.2.5
  cluster (590,743 Transaction vertices, 5,566 ClosedCase, 24 PolicyChunk, 12 installed GSQL queries, TigerVector
  indexes populated; 20 `AgentCase` vertices written back). `GRAPH_BACKEND=local` reproduces the same verdicts,
  probabilities, exposures, actions and SARs from the CSVs alone; only `similar_prior_cases` ordering differs
  (TigerVector cosine vs. TF-IDF).
* Savanna 4.2.5 gotchas we hit: `proxy` is a reserved attribute name (renamed `proxy_type`); vector attributes must be
  added with `ALTER VERTEX ... ADD VECTOR ATTRIBUTE` in a schema-change job; `vectorSearch` needs `LIST<FLOAT>` params
  and `MapAccum<VERTEX, FLOAT>` distance maps; the REST endpoint for tokens is `/gsql/v1/tokens`.
