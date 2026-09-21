# Demo video plan (3–5 minutes)

Record with the UI running (`uvicorn ui.app:app --port 8000`) and a terminal. Target 4:00.

| t | screen | say |
|---|---|---|
| 0:00 | title slide / README header | "Twenty fraud alerts, one policy, one graph. This is an agent that investigates — it doesn't just score." |
| 0:20 | `graph/schema.gsql` scrolled slowly, then diagram | "Customer, Card, Transaction, DeviceProfile as a vertex, NEXT edges per card, closed cases and policy text with embeddings. AgentCase is what the agent writes back." |
| 0:45 | terminal: `python run_cases.py` | "All twenty cases in about 25 seconds, fourteen GSQL queries each. Ten fraud, ten legitimate, four SARs." (show the table) |
| 1:10 | UI → HHG-014 | "An analyst asks about one $75 purchase. `device_neighbors` shows the same Samsung profile behind an anonymous proxy on 20 cards this month and on four closed cases. Pattern: undocumented → block (L1), monitor connected cards, SAR (L2), escalate. Every evidence line shows its query and its weight." Hover the graph: the hub device with the fan of cards. |
| 1:55 | UI → HHG-006 | "A customer disputes $482. The 48-hour window turns it into four purchases just under $500 in thirty minutes — and `pattern_peers` finds six other cards with the same signature. Exposure $1,906, SAR narrative in FinCEN's five elements." |
| 2:30 | UI → HHG-003 | "Another dispute — $49. `recurring_match`: the same amount, 56 times over six months on the customer's own card. Policy R7: verify, warn, close — don't block. Watch the initial plan turn into the final plan after the evidence request." |
| 3:00 | UI → HHG-015 | "A $600 online alert. The window shows three in-person purchases the day before in a region the card had never used. Episode $1,700 — over the SAR threshold. The agent asks the customer; the denial (R2) flips the plan to block and report." |
| 3:25 | terminal: MCP client or Claude Desktop calling `investigate_case HHG-010` | "The same tools are an MCP server — any client can drive an investigation step by step, or through the official TigerGraph MCP." |
| 3:45 | UI → HHG-005 evidence list | "And it remembers: this case already retrieves HHG-003 from the agent's own memory in the graph." |
| 3:55 | closing slide | "Graph-native, policy-driven, every claim cited. Thanks TigerGraph." |
