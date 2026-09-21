"""Policy, typology and regulatory text chunks loaded into the vector index (GraphRAG document side).

Source: the HHGOA dataset README (Fraud Policy v1.0, Known Fraud Patterns) and public FinCEN SAR narrative guidance.
Each chunk is small so retrieval returns the rule that applies, not the whole policy.
"""

POLICY_CHUNKS = [
    {"id": "policy:R1", "title": "Fraud Policy R1 – verify before you block on a weak signal",
     "text": "R1. Verify before you block on a weak signal. If the case rests on a single signal (including a risk score alone) "
             "and the assessed fraud probability is below 0.70, recommend VERIFY_WITH_CUSTOMER or STEP_UP_AUTH before any block. "
             "Blocking a legitimate customer on one signal is a policy breach."},
    {"id": "policy:R2", "title": "Fraud Policy R2 – customer denies the transaction",
     "text": "R2. Customer denies the transaction. Recommend BLOCK_CARD and CREATE_CASE. Add FILE_REPORT if exposure exceeds $1,000 "
             "or the case connects to a shared device profile or another card's fraud."},
    {"id": "policy:R3", "title": "Fraud Policy R3 – customer confirms the transaction",
     "text": "R3. Customer confirms the transaction. Recommend CLOSE_NO_FRAUD. Note the confirmation in the case file."},
    {"id": "policy:R4", "title": "Fraud Policy R4 – no reply within 24 hours",
     "text": "R4. No reply within 24 hours. Recommend MONITOR_CARD and DECLINE_TRANSACTION for pending authorizations. "
             "Escalate if exposure exceeds $500."},
    {"id": "policy:R5", "title": "Fraud Policy R5 – card testing",
     "text": "R5. Card testing. Three or more small online authorizations on one card within an hour, followed by a larger purchase: "
             "recommend DECLINE_TRANSACTION and STEP_UP_AUTH. If a purchase over $100 has already cleared, recommend BLOCK_CARD."},
    {"id": "policy:R6", "title": "Fraud Policy R6 – shared origin",
     "text": "R6. Shared origin. When several cards show fraud from the same device profile, the same billing region, or the same "
             "recipient email in one window, name the shared element, recommend CREATE_CASE and FILE_REPORT, and MONITOR_CONNECTED_CARDS "
             "for every card that shares it."},
    {"id": "policy:R7", "title": "Fraud Policy R7 – disputed but legitimate recurring charge",
     "text": "R7. Disputed but legitimate. When the customer disputes a charge that matches their own recurring pattern (same merchant, "
             "same amount, monthly), recommend CREATE_CASE, VERIFY_WITH_CUSTOMER, and WARN_CUSTOMER. Do not block."},
    {"id": "policy:R8", "title": "Fraud Policy R8 – escalate when uncertain and exposed",
     "text": "R8. Escalate when uncertain and exposed. If the verdict is uncertain and exposure exceeds $500, or the evidence conflicts, "
             "recommend ESCALATE_TO_ANALYST."},
    {"id": "policy:R9", "title": "Fraud Policy R9 – undocumented patterns",
     "text": "R9. Undocumented patterns. When activity fits none of the known patterns but the evidence shows coordinated or repeated abuse "
             "across customers, recommend CREATE_CASE, FILE_REPORT, and ESCALATE_TO_ANALYST, and describe the pattern in your own words. "
             "Do not force it into a known category."},
    {"id": "policy:R10", "title": "Fraud Policy R10 – never block all cards",
     "text": "R10. Never BLOCK_ALL_CARDS unless at least two of the customer's cards show confirmed fraud or the customer's credentials "
             "are confirmed compromised."},
    {"id": "policy:3a", "title": "Fraud Policy 3a – a case is not a report",
     "text": "A case (CREATE_CASE) is the bank's internal record. Open one whenever fraud probability reaches 0.30, whenever evidence is "
             "requested, or whenever a customer disputes a charge. A suspicious activity report (FILE_REPORT) is a regulatory filing. File "
             "one when fraud is confirmed or strongly suspected AND at least one holds: exposure exceeds $1,000; the activity connects to a "
             "shared device profile, a shared region cluster, or another customer's fraud; the pattern is coordinated or undocumented (R9). "
             "A report always has a case behind it. Most cases never need a report."},
    {"id": "policy:routing", "title": "Fraud Policy 2 – approval routing",
     "text": "auto: ALLOW_TRANSACTION, MONITOR_CARD, MONITOR_CONNECTED_CARDS, WARN_CUSTOMER, VERIFY_WITH_CUSTOMER, STEP_UP_AUTH, "
             "GENERATE_REPORT, CREATE_CASE, ESCALATE_TO_ANALYST, CLOSE_NO_FRAUD. L1 (team lead): DECLINE_TRANSACTION; BLOCK_CARD when "
             "exposure <= $2,500. L2 (fraud manager): BLOCK_CARD when exposure > $2,500; BLOCK_ALL_CARDS always; FILE_REPORT always. "
             "The agent recommends; only auto actions may be executed by the agent."},
    {"id": "policy:stopping", "title": "Fraud Policy 6 – stopping",
     "text": "Stop investigating when fraud probability is at or above 0.85 or at or below 0.15 supported by at least two independent "
             "pieces of evidence; or a verification response settles the question; or further steps are unlikely to change the decision."},
    {"id": "policy:exposure", "title": "Fraud Policy 4 – exposure",
     "text": "Exposure is the sum of the absolute amounts of every transaction identified as part of the fraud episode, including the flagged one, in USD."},
    {"id": "typology:card_testing", "title": "Known pattern 1 – card testing",
     "text": "Card testing. A stolen card number is checked before use: three or more tiny online authorizations, often under $5, then a "
             "larger purchase. Confirmed by the sequence itself. Policy R5."},
    {"id": "typology:cnp", "title": "Known pattern 2 – card-not-present fraud",
     "text": "Card-not-present fraud. The number is used online without the card. Amounts and products that don't fit the cardholder's "
             "history, often in a burst of two to four within 48 hours. On its own, one unusual online purchase is ambiguous: verify. Policy R1 to R4."},
    {"id": "typology:cnp_new_device", "title": "Known pattern 3 – card-not-present from a new device",
     "text": "Card-not-present fraud from a new device. Same as card-not-present, with the identity record marking the device as New for "
             "this account, sometimes behind a proxy. Stronger than pattern 2, still not proof: people buy new phones."},
    {"id": "typology:out_of_region", "title": "Known pattern 4 – out-of-region use",
     "text": "Out-of-region use. Card-present purchases in a billing region the cardholder has no history in, while their normal activity "
             "continues at home. Several days of purchases in one new region is a trip, not a clone. Policy R2, R3."},
    {"id": "typology:ato", "title": "Known pattern 5 – account takeover",
     "text": "Account takeover. Mixed-channel activity inconsistent with the cardholder, often with device and match-flag anomalies, "
             "pointing to stolen credentials rather than a stolen number."},
    {"id": "typology:undocumented_hub_device", "title": "Analyst memory – undocumented shared-device ring (Aug–Sep closed cases)",
     "text": "Closed cases CC-2649, CC-2971, CC-2985, CC-3035: purchases from a Samsung SM-G935F on Chrome for Android behind an anonymous "
             "proxy, a device never seen on the account, reported by several cardholders in the same month. Pattern not matched to a "
             "documented typology. Reports filed; connected cards monitored."},
    {"id": "typology:undocumented_threshold", "title": "Analyst memory – undocumented sub-threshold burst (Sep closed cases)",
     "text": "Closed cases CC-3748, CC-3841, CC-3907, CC-4086, CC-4124: four online purchases within forty minutes, each just under $500, "
             "none made by the cardholder. Amounts appear chosen to stay under a $500 authorization threshold. Not a documented typology."},
    {"id": "fincen:narrative", "title": "FinCEN SAR narrative guidance – the five essential elements",
     "text": "A SAR narrative must answer who conducted the activity, what instruments or mechanisms were used, when it took place, where "
             "it took place, why the filer thinks the activity is suspicious, and how it was conducted. Write chronologically, name all "
             "subjects and account identifiers, state total dollar amounts and dates, and describe the action taken by the institution. "
             "The narrative must stand on its own without the supporting file."},
    {"id": "fincen:ato", "title": "FinCEN Advisory FIN-2011-A016 – account takeover red flags",
     "text": "Account takeover indicators include access from unfamiliar devices or IP addresses, use of anonymizing proxies, changes to "
             "contact details followed by transactions, and transactions inconsistent with the customer's profile across multiple channels."},
]
