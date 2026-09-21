"""Fraud Policy v1.0 encoded as data + pure functions.

The agent *recommends*; ``route_for`` decides who must approve. Rule numbers are cited in every reason string
so the answer files satisfy policy §7 (explain and cite the rule).
"""
from __future__ import annotations

ACTIONS = {
    'ALLOW_TRANSACTION', 'DECLINE_TRANSACTION', 'MONITOR_CARD', 'MONITOR_CONNECTED_CARDS', 'WARN_CUSTOMER',
    'VERIFY_WITH_CUSTOMER', 'STEP_UP_AUTH', 'BLOCK_CARD', 'BLOCK_ALL_CARDS', 'GENERATE_REPORT', 'CREATE_CASE',
    'FILE_REPORT', 'ESCALATE_TO_ANALYST', 'CLOSE_NO_FRAUD',
}
AUTO = {'ALLOW_TRANSACTION', 'MONITOR_CARD', 'MONITOR_CONNECTED_CARDS', 'WARN_CUSTOMER', 'VERIFY_WITH_CUSTOMER',
        'STEP_UP_AUTH', 'GENERATE_REPORT', 'CREATE_CASE', 'ESCALATE_TO_ANALYST', 'CLOSE_NO_FRAUD'}

# ordering used when we emit several actions: "order them by what happens first"
ORDER = ['DECLINE_TRANSACTION', 'STEP_UP_AUTH', 'VERIFY_WITH_CUSTOMER', 'CREATE_CASE', 'BLOCK_CARD', 'BLOCK_ALL_CARDS',
         'MONITOR_CARD', 'MONITOR_CONNECTED_CARDS', 'WARN_CUSTOMER', 'FILE_REPORT', 'ESCALATE_TO_ANALYST',
         'GENERATE_REPORT', 'ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']

PATTERNS = ['card_testing', 'card_not_present_fraud', 'card_not_present_new_device', 'out_of_region_use',
            'account_takeover', 'undocumented', 'none']

SAR_EXPOSURE = 1000.0
ESCALATE_EXPOSURE = 500.0
CASE_THRESHOLD = 0.30
VERIFY_THRESHOLD = 0.70
STOP_HIGH, STOP_LOW = 0.85, 0.15


def route_for(action: str, exposure_usd: float) -> str:
    """Policy §2 approval routing."""
    if action in AUTO:
        return 'auto'
    if action == 'DECLINE_TRANSACTION':
        return 'L1'
    if action == 'BLOCK_CARD':
        return 'L1' if exposure_usd <= 2500 else 'L2'
    if action in ('BLOCK_ALL_CARDS', 'FILE_REPORT'):
        return 'L2'
    raise ValueError(action)


def action(name: str, reason: str, exposure_usd: float) -> dict:
    assert name in ACTIONS, name
    return {'action': name, 'route': route_for(name, exposure_usd), 'reason': reason}


def sort_actions(actions: list[dict]) -> list[dict]:
    seen, out = set(), []
    for a in sorted(actions, key=lambda a: ORDER.index(a['action'])):
        if a['action'] not in seen:
            seen.add(a['action'])
            out.append(a)
    return out


def sar_required(verdict: str, probability: float, exposure: float, shared_origin: bool, other_customer_fraud: bool,
                 undocumented: bool) -> tuple[bool, str]:
    """Policy §3a. Returns (file?, reason)."""
    strong = verdict == 'fraud' and probability >= 0.70
    if not strong:
        return False, ("3a: fraud is not confirmed or strongly suspected (probability %.2f, verdict %s); no report" % (probability, verdict))
    reasons = []
    if exposure > SAR_EXPOSURE:
        reasons.append(f"exposure ${exposure:,.2f} exceeds $1,000")
    if shared_origin:
        reasons.append("activity connects to a shared device profile / region cluster (R6)")
    if other_customer_fraud:
        reasons.append("activity connects to another customer's confirmed fraud")
    if undocumented:
        reasons.append("pattern is coordinated or undocumented (R9)")
    if reasons:
        return True, "3a: " + "; ".join(reasons)
    return False, (f"3a: fraud confirmed but exposure ${exposure:,.2f} is under $1,000, no shared origin and a documented pattern; "
                   "case only, no report")
