# Demo video script (3–5 min, target 4:00)

**Recorder:** OBS / Loom / Xbox Game Bar (`Win+Alt+R`). Mic on. Browser zoom 110–125 %. One take is fine; trim later
(Clipchamp is built into Windows). Upload to YouTube as **Unlisted** and paste the link in the submission form.

## Set-up before recording (open these 4 tabs)

| Tab | URL / screen | Pre-state |
|---|---|---|
| 1 | GitHub repo `https://github.com/Vikas-93257/tigergraph-fraud-agent` | README top visible |
| 2 | TigerGraph Savanna → **Query Editor** → graph `FraudGraph` | left panel: Queries list expanded; `demo_device_ring` run once already (default params) |
| 3 | Agent UI (`uvicorn ui.app:app --port 8000`, `GRAPH_BACKEND=tigergraph`) | case **HHG-014** open |
| 4 | Savanna Query Editor (second tab) → `prior_cases_for_customer`, param `cust = C13487` | run once already |

Make sure the Savanna workspace is **Active** (green dot) before starting — it auto-stops when idle.

## Script

| Time | Screen | Say |
|---|---|---|
| 0:00–0:20 | **Tab 1** README header | "Hi, I'm ___. This is a graph-native fraud *investigator* built on TigerGraph for the HHGoa hackathon. Given an alert it doesn't just score — it walks the graph, reads the bank's policy, asks for evidence, picks next-best-actions, files SARs when required, and remembers the case." |
| 0:20–0:50 | **Tab 2** Savanna — show the schema diagram on the right and the queries list on the left | "Everything runs on TigerGraph Savanna. 590 thousand transactions, 13 thousand cards, 9,700 device profiles, 5,500 closed cases and the bank's policy text — all vertices. These thirteen installed GSQL queries are the agent's only tools. Similar cases and policy are retrieved with TigerVector search inside the graph." |
| 0:50–1:15 | **Tab 2** run `demo_device_ring` → **JSON** tab (`distinct_cards: 20`, `proxies` all `IP_PROXY:ANONYMOUS`) → **Graph** tab | "One device profile. Two hops: device → transactions → cards. Twenty different cards, every one behind an anonymous proxy, and four already-confirmed fraud cases. That's a ring, found in a single query." |
| 1:15–2:05 | **Tab 3** UI → HHG-014 → click **↻ Re-run live**; scroll evidence table, actions, hover the graph | "Same case inside the agent. An analyst asks about one 75-dollar purchase. Live run against TigerGraph — about a second. Verdict fraud, 0.90, pattern *undocumented*, 19 connected cards. Every evidence line cites the GSQL query that produced it. Actions are routed: create case is automatic, block card goes to L1, SAR and escalation go to L2." |
| 2:05–2:35 | **Tab 3** → HHG-006 → SAR box | "A customer disputes 482 dollars. The 48-hour card window turns it into four purchases just under 500 in thirty minutes — threshold evasion — and `pattern_peers` finds six other cards with the same signature. Exposure 1,906 dollars; the SAR narrative is written in FinCEN's five elements." |
| 2:35–3:00 | **Tab 3** → HHG-003 → evidence table | "Another dispute, 49 dollars. `recurring_match`: the same amount 56 times over six months on the customer's own card — a subscription. Policy R7 says verify and warn, don't block. Half of the pack is legitimate and the agent gets those right too." |
| 3:00–3:25 | **Tab 3** → HHG-015 → initial vs final actions + "what changed" | "A 600-dollar online alert. The window shows three in-person purchases the day before in a region the card had never used — the episode is 1,700 dollars, over the SAR threshold. The agent requests customer confirmation; the denial flips the plan from step-up to block-and-report. Initial plan, final plan, what changed." |
| 3:25–3:45 | **Tab 4** `prior_cases_for_customer("C13487")` result → `AC` list shows `CASE-2016-014` | "And it remembers. Every verdict is written back as an `AgentCase` vertex with edges to cards, devices and similar cases, so the next investigation retrieves it through GraphRAG." |
| 3:45–4:00 | **Tab 1** README results table | "Twenty cases in 26 seconds on the cluster. Ten fraud, ten legitimate, four SARs, every claim cited, zero LLM tokens needed for correctness. The same tools are exposed over MCP. Thanks TigerGraph and 247pm studio." |

## Facts you can quote (all verified on the Savanna cluster)

* Graph: 590,743 Transaction · 13,554 Customer · 13,575 Card · 9,706 DeviceProfile · 5,566 ClosedCase · 24 PolicyChunk · 20 AgentCase; 577k NEXT edges.
* Ring device: `SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for android | 1920x1080` → 52 cards all-time, 20 cards / 26 txns in the 30-day window, closed cases CC-2649, CC-2971, CC-2985, CC-3035.
* HHG-006: 4 × ~$480 in 30 min, $1,906.07, 6 peer cards, SAR, escalated.
* HHG-003: $49 × 56 over 6 months → legitimate, R7 warn-not-block.
* HHG-015: $441 + $550 + $550 in-person + $599.94 online = $1,699.98, SAR.
* Pack: 10 fraud / 10 legitimate, 4 SARs, ~14 GSQL calls per case, 26 s total, 0 LLM tokens.

## If something breaks while recording

* UI header says "store offline" → the Savanna workspace has paused; resume it in tgcloud.io (2–5 min), restart uvicorn.
* GraphStudio DATETIME pickers don't accept 2016 → use `demo_device_ring` (string dates, defaults already set).
