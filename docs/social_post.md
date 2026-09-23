# Social post (LinkedIn / X) — post from your own account and tag **@TigerGraphDB and @247pmstudio** (both required by the form)

**LinkedIn**

We built an autonomous fraud *investigator* — not another classifier — for the @TigerGraph Agentic Fraud Investigation hackathon.

Given an alert it walks the transaction graph in TigerGraph (GSQL), pulls prior cases + the bank's policy with GraphRAG, asks the customer for evidence when the rules say so, picks next-best-actions with the right approval route, files a SAR when policy requires, and writes the case back to the graph as memory. Every claim in the answer carries the GSQL query that produced it.

Results on the 20-case pack: 10 fraud / 10 legitimate, 4 SARs, 14 GSQL calls per case, 26 s for the pack on TigerGraph Savanna (590k transactions, TigerVector GraphRAG), 0 LLM tokens needed for correctness. Highlights: a shared-device ring across 20 cards found in two hops; a 4×$480-in-30-minutes threshold-evasion burst matched on 6 other cards; three "disputed" charges that were the customers' own subscriptions (R7 → warn, don't block).

LangGraph · TigerGraph GSQL · TigerGraph MCP · GraphRAG · FastAPI UI
Repo + blog + video: <link>

cc @TigerGraphDB @247pmstudio

#TigerGraph #GraphRAG #AgenticAI #FraudDetection #MCP

**X / Twitter (≤280)**

Built a graph-native fraud investigator for the @TigerGraphDB agentic fraud hackathon: GSQL tools + GraphRAG memory + MCP, LangGraph loop that asks for evidence, routes actions to humans and files SARs. 20 cases, 10/10 split, every claim cites its query. @247pmstudio <link> #TigerGraph #GraphRAG
