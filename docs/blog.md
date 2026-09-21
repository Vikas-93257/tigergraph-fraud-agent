# An investigator, not a classifier: building a graph-native fraud agent on TigerGraph

*Technical write-up for the TigerGraph Agentic Fraud Investigation hackathon (HHGOA dataset).*

## The problem is not "is this fraud?"

The hackathon hands you 590,742 card transactions from 13,553 customers, 5,565 closed cases with analyst notes, a
fraud policy with ten rules, and twenty alerts. The scoreboard says accuracy is 25 % of the grade. Next-best-actions are
another 25 %. Agentic design, innovation, explainability and the demo make up the rest.

That split is the whole brief. A fraud model already exists in the data (`risk_score`), and the README warns that above
0.7 most flagged transactions are still legitimate. What a bank pays an analyst for is what happens *after* the score:
pull the card's history, check whether the device is new, see whether anyone else got hit from the same place, decide
whether to call the customer, and write it up so the next analyst — or the regulator — can follow the reasoning.

So we built an investigator. It has tools (GSQL queries), a memory (the closed cases and its own past cases), a rule book
(the policy, retrieved not hard-coded), a budget, and the obligation to cite everything.

## The graph

```
Customer ─OWNS─► Card ─MADE─► Transaction ─FROM_DEVICE─► DeviceProfile
                                   │ ─BILLED_IN─► BillingRegion
                                   │ ─PURCHASER_EMAIL / RECIPIENT_EMAIL─► EmailDomain
                                   └─NEXT─► Transaction          (per-card timeline)
ClosedCase ─INVOLVES─► Transaction, ─ON_CARD─► Card, ─CONNECTED_TO─► Card
AgentCase  ─AC_INVOLVES / AC_ON_CARD / AC_CONNECTED_TO / AC_USED_DEVICE / AC_SIMILAR_TO
PolicyChunk (text + embedding)
```

Two modelling decisions did most of the work.

**DeviceProfile is a vertex.** The IEEE identity table gives you `DeviceInfo | OS | browser | screen`. Treating that
tuple as a node turns "has anyone else used this device?" into a two-hop query (`device_neighbors`). That is how case
HHG-014 finds a Samsung SM-G935F behind an anonymous proxy sitting on **20 cards in 30 days** and on four closed cases
from August–September — a ring, not a stolen number.

**NEXT edges per card.** `card_window` walks NEXT/PREV from the flagged transaction until it leaves ±48 h. It is cheap,
it never touches the rest of the card's history, and it is how HHG-006 becomes four purchases of $478.95, $456.96,
$488.04 and $482.12 in thirty minutes rather than one $482 alert.

The rest of the schema is there to be explainable: `ClosedCase` narratives and `PolicyChunk` text carry embeddings so
retrieval is a graph query too, and `AgentCase` is what the agent writes back.

## The agent

A LangGraph state machine with eight nodes:

```
trigger → gather → memory → assess → initial_decision ─┬─ need evidence? → request_evidence → assess → final_decision
                                                        └───────────────────────────────────────────► final_decision
final_decision → explain → memorize
```

*Gather* is a fixed plan of graph queries — deliberately. Every case gets `get_transaction`, `card_window`,
`card_profile`, `device_neighbors`, `region_history`, `recurring_match`, `small_auth_run`, `customer_fraud_rate`. Two
extra queries run only when the evidence calls for them: `pattern_peers` (sub-$500 burst, anonymous-proxy-new-device)
across the whole customer base, and `card_txns_on_device` when the device is New and unseen. Every call is capped at the
trigger's `opened_at`; the agent cannot peek at the future.

*Memory* is GraphRAG: prior cases on the customer's cards, the five nearest closed cases to a query that is built from
the graph facts ("online, product C, device never seen, four purchases just under $500 within forty minutes"), and the
policy chunks that apply. The retrieved texts appear in the answer as `source: document` evidence.

*Assess* is where we resisted the temptation to hand the whole thing to an LLM. Each extractor returns a **Signal** —
a claim in plain English, a `source`, a `ref` that is literally the GSQL query name and arguments, the entity ids it
rests on, and a log-odds weight. Weights sum onto a trigger-specific prior (risk_score 0.22, customer_report 0.55,
analyst_request 0.40) and are clamped to [0.05, 0.95]. The model score gets +0.2 at most. A recurring-charge match is
−1.6. A device shared by five or more cards, mostly New and behind a proxy, is +2.3. The point is that the number is
auditable: in the UI every evidence line shows its weight, and you can add them up yourself.

*Decide* is policy, retrieved and applied: R1 verify before you block on a single weak signal; 3a a case is not a report;
R7 a disputed charge that matches the customer's own recurring pattern gets a reminder, not a block; R8 escalate when
uncertain and exposed above $500; R9 undocumented-but-coordinated patterns get a report and a human; R10 never block all
cards. `BLOCK_CARD` always routes to L1, `FILE_REPORT` to L2.

*Request evidence* is the loop. If the plan includes VERIFY or STEP_UP the agent logs an `evidence_request`, continues
with a documented assumed reply, re-assesses and re-decides. The simulation rule is deterministic and printed in the
README: a strong recurring history means the customer recognises the charge; otherwise p ≥ 0.50 → denial, p < 0.50 →
confirmation. `what_changed` in the answer says what the reply did to the plan.

*Stop* when two independent signals agree and p is outside [0.15, 0.85], or at 18 tool calls. Average was 14.

*Memorize* writes an `AgentCase` vertex and edges. Case HHG-005, investigated later in the pack, already retrieves
HHG-003 as a similar prior case — the memory is live, not a log file.

## What the twenty cases taught us

**The score is a hint.** Ten of the twenty alerts are model-scored. Six of them close as legitimate: a $77 in-person
purchase in a billing region the card has used ten times (HHG-001), a $112 purchase in the customer's home region
of 2,221 prior visits (HHG-007), a $99.92 amount that recurs on the card (HHG-019).

**Disputes are evidence, not verdicts.** Eight alerts are customer reports. Three of them — $49, $131, $39 — match a
monthly charge on the customer's own card 56, 4 and 145 times respectively. The policy's R7 exists for exactly this. The
agent opens the case (3a), verifies, and closes with WARN_CUSTOMER so the next cycle is not disputed again.

**Undocumented means coordinated.** The typology catalogue has five patterns. Two of the fraud cases match none of them
and both are cross-customer: the device ring in HHG-014 and the sub-threshold burst in HHG-006 (six other cards show the
same 3–4-purchases-just-under-$500-in-an-hour signature in the same month). Only that cross-customer evidence earns the
`undocumented` label, the SAR and `MONITOR_CONNECTED_CARDS`. A single odd transaction does not.

**Exposure is an episode, not a transaction.** HHG-015 arrives as a $599.94 online alert. The window shows three in-person
purchases of $441, $550 and $550 the day before in a billing region the card had never used; together with the alert the
episode is $1,699.98 and crosses the $1,000 SAR threshold. HHG-010 is a single $1,000.03 purchase from a New device and
crosses it alone.

**Half legitimate, no LLM required.** Final tally: 10 fraud, 10 legitimate, 4 SARs, 0 tokens. The template narrator writes
the summaries and SAR narratives from the evidence list following FinCEN's five elements (who, what, when, where, why).
Set `OPENAI_API_KEY` and the same evidence is handed to an LLM for prose — nothing else changes.

## MCP and the two backends

Every tool is a named GSQL query. That gave us three things for free:

1. **An MCP server** (`mcp_server/server.py`) that exposes the fifteen graph tools plus `investigate_case` to any MCP client;
   the LangGraph agent, Claude Desktop and the UI all call the same functions.
2. **Interchangeability with the official TigerGraph MCP** — `tigergraph_mcp_client.py` runs the identical query names
   through `tigergraph-mcp`'s generic query execution tool.
3. **A local backend** with the same contract (pandas + TF-IDF standing in for TigerVector) so the twenty-case benchmark,
   the tests and the UI run in 25 seconds on a laptop, and the GSQL has a reference implementation to be checked against.

## What we'd do next

Real evidence channels instead of the simulation rule; TigerVector embeddings from a sentence-transformer instead of the
hashed fallback; a scheduled monitoring sweep (the `monitoring/` track already investigates the top alerts of a period
and writes a summary — on 1 Dec 2016 it reviewed eight, auto-closed five, and flagged one $1,650 account takeover for a
SAR); and feedback edges from L1/L2 decisions back onto `AgentCase` so the weights can be recalibrated from outcomes.

*Code, GSQL, answers and UI: see the repository.* #TigerGraph #GraphRAG #AgenticAI #FraudDetection
