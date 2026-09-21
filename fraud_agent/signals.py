"""Turn raw graph query results into named, weighted evidence signals.

Each signal carries a log-odds weight; the investigator sums them onto a trigger-specific prior to get a calibrated
fraud probability. Weights were set from the closed-case history (July–October) – see docs/calibration.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Signal:
    key: str
    claim: str
    weight: float                       # log-odds contribution (+ = fraud)
    source: str = 'graph'               # graph | document | customer | external
    ref: str = ''
    entity_ids: list[str] = field(default_factory=list)
    independent: bool = True            # counts toward the "two independent pieces of evidence" stopping rule

    def as_evidence(self) -> dict:
        return {'claim': self.claim, 'source': self.source, 'ref': self.ref, 'entity_ids': self.entity_ids}


PRIORS = {'risk_score': 0.22, 'customer_report': 0.55, 'analyst_request': 0.40}


def logit(p): return math.log(p / (1 - p))
def sigmoid(x): return 1 / (1 + math.exp(-x))


def combine(prior: float, signals: list[Signal]) -> float:
    x = logit(prior) + sum(s.weight for s in signals)
    return round(min(0.95, max(0.05, sigmoid(x))), 2)


def _money(x): return f"${x:,.2f}"


# ----------------------------------------------------------------------------------------------- signal extractors

def risk_score_signal(txn: dict) -> Signal:
    rs = txn.get('risk_score') or 0.0
    # README: above 0.7 most flagged are legitimate -> weak weight, never decisive
    w = round((rs - 0.5) * 0.8, 2)
    return Signal('risk_score', f"Bank model risk score {rs:.2f} on the flagged transaction (an input, not a verdict)",
                  w, ref=f"query:get_transaction(txn_id={txn['txn_id']})", entity_ids=[txn['txn_id']], independent=False)


def amount_signals(txn: dict, prof: dict) -> list[Signal]:
    out = []
    fl = prof['flagged']
    amt = txn['amount']
    ref = f"query:card_profile(card_id={prof['card_id']})"
    if prof['n_txns'] < 5:
        return out
    if prof['amount_max'] is not None and amt > prof['amount_max']:
        out.append(Signal('amount_above_max', f"{_money(amt)} exceeds every prior amount on this card (previous max {_money(prof['amount_max'])}, median {_money(prof['amount_median'])})", 1.2, ref=ref, entity_ids=[txn['txn_id']]))
    elif prof['amount_p95'] is not None and amt > prof['amount_p95']:
        out.append(Signal('amount_above_p95', f"{_money(amt)} is above the card's 95th percentile ({_money(prof['amount_p95'])}; median {_money(prof['amount_median'])})", 0.7, ref=ref, entity_ids=[txn['txn_id']]))
    elif prof['amount_median'] is not None and amt <= prof['amount_median'] * 1.5:
        out.append(Signal('amount_typical', f"{_money(amt)} is within the card's normal range (median {_money(prof['amount_median'])}, p95 {_money(prof['amount_p95'])})", -0.5, ref=ref, entity_ids=[txn['txn_id']]))
    if fl['product_seen_before'] == 0 and prof['n_txns'] >= 20:
        out.append(Signal('product_new', f"Product code {txn['product']} never used on this card in {prof['n_txns']} prior transactions", 0.7, ref=ref, entity_ids=[txn['txn_id']]))
    return out


def channel_device_signals(txn: dict, prof: dict, devn: dict) -> list[Signal]:
    out = []
    if txn['channel'] != 'online':
        return out
    ref = f"query:card_profile(card_id={prof['card_id']})"
    dev = txn.get('device_profile')
    status = txn.get('device_status')
    seen = prof['flagged'].get('device_seen_before')
    generic = devn.get('generic_fingerprint', False) if devn else True
    share_online = prof['flagged'].get('channel_share_online') or 0
    new_share = prof.get('new_device_share')
    if dev and seen == 0 and status == 'New' and new_share is not None and new_share >= 0.3:
        out.append(Signal('device_new_common', f"Device is marked New, but {int(new_share*100)}% of this account's online history comes from New devices — a new device is normal here", 0.1, ref=ref, entity_ids=[txn['txn_id']], independent=False))
    elif dev and seen == 0 and status == 'New' and not generic:
        out.append(Signal('device_new_unseen', f"Device profile '{dev}' is marked New and has never been seen on this account ({prof['n_known_devices']} known devices)", 1.0, ref=ref, entity_ids=[txn['txn_id']]))
    elif dev and seen == 0 and status == 'New' and generic:
        out.append(Signal('device_new_generic', f"Device is marked New but the fingerprint '{dev}' is too coarse to identify a device; weak signal", 0.3, ref=ref, entity_ids=[txn['txn_id']], independent=False))
    elif dev and (seen or 0) > 0:
        out.append(Signal('device_known', f"Device profile '{dev}' was already used {seen} time(s) on this account before the alert", -0.7, ref=ref, entity_ids=[txn['txn_id']]))
    elif not dev:
        out.append(Signal('device_unknown', "No device fingerprint captured for this online transaction", 0.0, ref=ref, entity_ids=[txn['txn_id']], independent=False))
    proxy = txn.get('proxy')
    if proxy and ('ANONYMOUS' in proxy or 'HIDDEN' in proxy):
        out.append(Signal('proxy_anon', f"Connection came through an anonymising proxy ({proxy}); card has {prof.get('proxy_history', 0)} prior proxied transactions", 0.9, ref=ref, entity_ids=[txn['txn_id']]))
    elif proxy:
        out.append(Signal('proxy_transparent', f"Connection flagged {proxy}", 0.3, ref=ref, entity_ids=[txn['txn_id']], independent=False))
    if share_online < 0.1 and prof['n_txns'] >= 30:
        out.append(Signal('channel_shift', f"Cardholder is almost exclusively in-person ({prof['n_online']} of {prof['n_txns']} prior transactions online); an online purchase is itself unusual", 0.6, ref=ref, entity_ids=[txn['txn_id']]))
    if txn.get('p_email') == 'anonymous.com' and txn.get('r_email') == 'anonymous.com':
        out.append(Signal('anon_email', "Purchaser and recipient email domains are both anonymous.com", 0.3, ref=ref, entity_ids=[txn['txn_id']], independent=False))
    return out


def region_signals(txn: dict, prof: dict, reg: dict) -> list[Signal]:
    out = []
    if txn['channel'] != 'in_person' or not reg or reg.get('region') is None:
        return out
    ref = f"query:region_history(card_id={prof['card_id']}, region={int(reg['region'])})"
    n = reg['n_before']
    if n == 0:
        w = 1.3
        claim = f"Billing region {int(reg['region'])} has no history on this card (home region {int(reg['home_region']) if reg.get('home_region') is not None else '?'})"
        if reg['in_region_next_7d'] >= 3 and reg['home_activity_next_7d'] == 0:
            w = 0.2
            claim += f"; {reg['in_region_next_7d']} further purchases in the same region over the next week with no home activity — consistent with travel, not a clone"
        elif reg['home_activity_next_7d'] > 0 and reg['in_region_next_7d'] >= 1:
            claim += f"; home-region activity continues in parallel ({reg['home_activity_next_7d']} home purchases in the next 7 days)"
            w = 1.6
        out.append(Signal('region_new', claim, w, ref=ref, entity_ids=[txn['txn_id']]))
    elif n >= 3:
        out.append(Signal('region_known', f"Billing region {int(reg['region'])} already appears {n} times on this card across {reg['months_active_in_region']} month(s)", -0.9, ref=ref, entity_ids=[txn['txn_id']]))
    else:
        out.append(Signal('region_rare', f"Billing region {int(reg['region'])} seen only {n} time(s) before on this card", 0.3, ref=ref, entity_ids=[txn['txn_id']]))
    return out


def recurring_signal(txn: dict, prof: dict, rec: dict) -> Signal | None:
    if not rec or not rec['is_recurring']:
        return None
    strong = rec['n_months'] >= 4 and rec['n_matches'] >= 5
    return Signal('recurring' if strong else 'recurring_weak',
                  f"Same product and amount ({_money(rec['amount'])} {rec['product']}) recurs {rec['n_matches']} times across {rec['n_months']} months on this card — "
                  + ("matches the cardholder's own recurring pattern" if strong else "a possible recurring charge, but the history is thin"),
                  -1.6 if strong else -0.6, ref=f"query:recurring_match(card_id={prof['card_id']}, txn_id={txn['txn_id']})", entity_ids=rec['sample_txn_ids'][-4:])


def testing_signal(txn: dict, prof: dict, sar: dict) -> Signal | None:
    if not sar or not sar['testing_sequence']:
        return None
    ids = sar['small_txn_ids'] + [t['txn_id'] for t in sar['larger_after']]
    return Signal('card_testing', f"{sar['n_small_auths']} online authorizations of $5 or less within an hour followed by a larger purchase — a card-testing sequence",
                  2.4, ref=f"query:small_auth_run(card_id={prof['card_id']}, txn_id={txn['txn_id']})", entity_ids=ids)


def burst_signals(txn: dict, prof: dict, window: list[dict]) -> tuple[list[Signal], list[str]]:
    """Look for a 48h burst of unusual online activity and the sub-$500 threshold-evasion signature."""
    out, episode = [], []
    p95 = prof.get('amount_p95') or 0
    onl = [t for t in window if t['channel'] == 'online']
    new_common = (prof.get('new_device_share') or 0) >= 0.3
    unusual = [t for t in onl if (t['amount'] > p95 and prof['n_txns'] >= 5) or
               (t.get('device_status') == 'New' and prof['flagged'].get('device_seen_before') == 0 and not new_common)]
    ref = f"query:card_window(card_id={prof['card_id']}, hours=48)"
    # threshold evasion: >=3 online txns in 60 min, each 400 <= amt < 500
    big = sorted([t for t in onl if 400 <= t['amount'] < 500], key=lambda t: t['ts'])
    for i in range(len(big)):
        grp = [t for t in big if 0 <= _mins(big[i]['ts'], t['ts']) <= 60]
        if len(grp) >= 3:
            ids = [t['txn_id'] for t in grp]
            out.append(Signal('threshold_burst', f"{len(grp)} online purchases of {', '.join(_money(t['amount']) for t in grp)} within {int(_mins(grp[0]['ts'], grp[-1]['ts']))} minutes, each just under $500 — amounts consistent with staying under a $500 authorization threshold", 2.6, ref=ref, entity_ids=ids))
            episode = ids
            break
    if not episode and len(unusual) >= 2:
        ids = [t['txn_id'] for t in unusual]
        out.append(Signal('burst', f"{len(unusual)} unusual online purchases within 48 hours ({', '.join(_money(t['amount']) for t in unusual[:5])})", 0.8, ref=ref, entity_ids=ids))
        episode = ids
    # spending spike across channels: several top-5% amounts in the window when the card's tempo makes that unlikely
    if prof['n_txns'] >= 20 and p95:
        spikes = [t for t in window if t['amount'] > p95]
        expected = 0.05 * len(window)
        if len(spikes) >= 3 and len(spikes) >= 4 * expected and sum(t['amount'] for t in spikes) > 2 * p95:
            ids = [t['txn_id'] for t in spikes]
            chans = sorted({t['channel'].replace('_', '-') for t in spikes})
            out.append(Signal('spend_spike', f"{len(spikes)} purchases above the card's 95th percentile ({_money(p95)}) within 48 hours ({', '.join(_money(t['amount']) for t in spikes[:5])}; {' and '.join(chans)}) — the card normally shows one such amount every {int(1/0.05)} purchases",
                              1.0, ref=ref, entity_ids=ids))
            episode = list(dict.fromkeys(episode + ids))
    return out, episode


def duplicate_burst_signal(txn: dict, prof: dict, rec: dict) -> Signal | None:
    ids = (rec or {}).get('same_amount_burst_ids') or []
    if len(ids) < 3:
        return None
    return Signal('duplicate_burst', f"{len(ids)} charges of the same amount ({_money(rec['amount'])}) within one hour on this card — repeated submissions rather than a single purchase",
                  0.8, ref=f"query:recurring_match(card_id={prof['card_id']}, txn_id={txn['txn_id']})", entity_ids=ids)


def hub_device_signal(txn: dict, devn: dict, prof: dict) -> Signal | None:
    if not devn or not devn.get('device_profile') or devn.get('generic_fingerprint'):
        return None
    n = devn['n_cards']
    fraud_cases = [c for c in devn['closed_cases_on_device'] if c['outcome'] == 'confirmed_fraud']
    ref = f"query:device_neighbors(device_id={devn['device_profile']})"
    proxy_share = devn.get('proxy_share_in_window') or 0
    new_share = devn.get('new_share_in_window') or 0
    if n >= 5 and proxy_share >= 0.5 and new_share >= 0.8:
        return Signal('hub_device', f"Device profile '{devn['device_profile']}' was used by {n} different cards in the window, {int(proxy_share*100)}% behind a proxy and {int(new_share*100)}% marked New for the account; {len(fraud_cases)} closed fraud case(s) already sit on this device",
                      2.3, ref=ref, entity_ids=[c['card_id'] for c in devn['cards'] if c['customer_id'] != prof['customer_id']][:30] + [c['case_id'] for c in fraud_cases])
    if fraud_cases and devn['n_cards_all_time'] <= 60:
        return Signal('device_prior_fraud', f"Device profile '{devn['device_profile']}' appears on {len(fraud_cases)} confirmed fraud case(s) ({', '.join(c['case_id'] for c in fraud_cases[:3])}) and only {devn['n_cards_all_time']} cards overall",
                      0.9, ref=ref, entity_ids=[c['case_id'] for c in fraud_cases[:5]])
    return None


def history_signal(prior_cases: list[dict], customer_id: str) -> Signal | None:
    if not prior_cases:
        return None
    fraud = [c for c in prior_cases if c['outcome'] == 'confirmed_fraud']
    cleared = [c for c in prior_cases if c['outcome'] == 'cleared']
    ref = f"query:prior_cases_for_customer(customer_id={customer_id})"
    pats = {}
    for c in fraud:
        pats[c['pattern']] = pats.get(c['pattern'], 0) + 1
    if len(fraud) >= 3:
        top = max(pats, key=pats.get)
        return Signal('history_repeat_victim', f"Customer has {len(fraud)} confirmed fraud cases since July (mostly {top}) and {len(cleared)} cleared alert(s) — a repeat target",
                      0.35, ref=ref, entity_ids=[c['case_id'] for c in fraud[:5]], independent=False)
    if cleared and not fraud:
        return Signal('history_cleared', f"Customer's only prior alert(s) were cleared as false alarms ({', '.join(c['case_id'] for c in cleared[:3])})",
                      -0.3, ref=ref, entity_ids=[c['case_id'] for c in cleared[:3]], independent=False)
    return None


def _mins(a: str, b: str) -> float:
    from datetime import datetime
    fa = datetime.strptime(a, '%Y-%m-%d %H:%M:%S'); fb = datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    return (fb - fa).total_seconds() / 60
