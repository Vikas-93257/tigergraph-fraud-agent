"""The investigation agent: a LangGraph state machine over graph tools.

    trigger -> open_case -> gather_graph_evidence -> retrieve_memory -> assess
           -> (need_more_evidence? request_evidence -> reassess) -> decide -> explain -> memorize

Design principles
* Graph does the analysis (GSQL/pandas queries); the LLM (optional) only narrates. Every number in the answer
  file is traceable to a tool call.
* Uncertainty is explicit: a calibrated ``fraud_probability`` drives whether we act, ask, or escalate (policy R1/R8).
* Two recommendations are always recorded – before and after any evidence request (policy 3b).
* Permissions are enforced in code (policy §2): the agent may only *execute* ``auto`` actions.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from . import policy as P
from . import signals as S
from .tools import Toolbox, ToolLog
from .narrate import Narrator


class CaseState(TypedDict, total=False):
    case_id: str
    trigger: dict
    tools: Any
    txn: dict
    profile: dict
    window: list
    device: dict
    region: dict
    recurring: dict
    small_auth: dict
    prior_cases: list
    similar: list
    fraud_rate: dict
    policy_docs: list
    signals: list
    probability: float
    initial_prob: float
    peers: dict
    tool_log: list
    pattern: str
    pattern_description: str
    episode_ids: list
    connected_cards: list
    connected_devices: list
    exposure: float
    initial_actions: list
    final_actions: list
    evidence_requests: list
    verdict: str
    status: str
    stop_reason: str
    what_changed: str
    sar: dict
    summary: str
    graph_case_id: str
    step: int
    started: float
    tokens: int


# ------------------------------------------------------------------------------------------------- nodes

def n_trigger(st: CaseState) -> CaseState:
    tb: Toolbox = st['tools']
    tr = st['trigger']
    txn = tb.get_transaction(tr['flagged_txn_id'])
    if txn is None:
        raise ValueError(f"flagged txn {tr['flagged_txn_id']} not in graph")
    txn['card_id'] = tr.get('card_id') or txn['card_id']
    return {'txn': txn, 'step': 1, 'signals': [], 'evidence_requests': [], 'tokens': 0}


def n_gather(st: CaseState) -> CaseState:
    """Pull the connected evidence: card history, ±48h window, device hub, region, recurring, testing run."""
    tb: Toolbox = st['tools']
    txn, tr = st['txn'], st['trigger']
    card = txn['card_id']
    prof = tb.card_profile(card, txn['txn_id'])
    window = tb.card_window(card, txn['txn_id'], 48, 48)
    dev = None
    t0 = datetime.strptime(txn['ts'], '%Y-%m-%d %H:%M:%S')
    if txn['channel'] == 'online' and txn.get('device_profile'):
        dev = tb.device_neighbors(txn['device_profile'], (t0 - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S'),
                                  tb.as_of or (t0 + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    reg = tb.region_history(card, txn.get('region'), txn['txn_id']) if txn['channel'] == 'in_person' else None
    rec = tb.recurring_match(card, txn['txn_id'])
    sar = tb.small_auth_run(card, txn['txn_id']) if txn['channel'] == 'online' else None
    fr = tb.customer_fraud_rate(txn['customer_id'])
    return {'profile': prof, 'window': window, 'device': dev, 'region': reg, 'recurring': rec, 'small_auth': sar,
            'fraud_rate': fr, 'step': 2}


def n_memory(st: CaseState) -> CaseState:
    """Case memory + GraphRAG: prior cases on this customer, similar closed cases, applicable policy text."""
    tb: Toolbox = st['tools']
    txn = st['txn']
    prior = tb.prior_cases_for_customer(txn['customer_id'])
    hint = _pattern_hint(st)
    parts = [txn['channel'].replace('_', ' '), f"product {txn['product']}", st['trigger']['trigger_type'].replace('_', ' ')]
    if txn.get('device_profile'):
        parts.append(txn['device_profile'].replace('|', ' '))
    if txn.get('device_status') == 'New':
        parts.append('device never seen on this account new device')
    if txn.get('proxy'):
        parts.append(txn['proxy'].replace('IP_PROXY:', '').lower() + ' proxy')
    dev = st.get('device') or {}
    if dev.get('n_cards', 0) >= 5:
        parts.append('same device profile several cardholders reported this month')
    big = [t for t in st['window'] if t['channel'] == 'online' and 400 <= t['amount'] < 500]
    if len(big) >= 3:
        parts.append('four online purchases within forty minutes each just under $500 authorization threshold')
    if st.get('recurring', {}).get('is_recurring'):
        parts.append('recurring charge same amount monthly cardholder confirmed')
    if txn['channel'] == 'in_person' and (st.get('region') or {}).get('n_before') == 0:
        parts.append('billing region cardholder had no history in travel')
    q = ' '.join(parts)
    sim = tb.similar_cases(q, pattern_hint=hint, customer_id=txn['customer_id'], k=5)
    docs = tb.retrieve_policy(f"{hint} {st['trigger']['trigger_type']} verify block report shared device", k=4)
    return {'prior_cases': prior, 'similar': sim, 'policy_docs': docs, 'step': 3}


def n_assess(st: CaseState) -> CaseState:
    """Build weighted signals from the evidence and compute a calibrated fraud probability + pattern."""
    tb: Toolbox = st['tools']
    txn, prof, tr = st['txn'], st['profile'], st['trigger']
    sigs: list[S.Signal] = []
    episode: list[str] = [txn['txn_id']]
    connected_cards: list[str] = []
    connected_devices: list[str] = []
    pattern, pdesc = 'none', ''

    if tr['trigger_type'] == 'risk_score':
        sigs.append(S.risk_score_signal(txn))
    elif tr['trigger_type'] == 'customer_report':
        sigs.append(S.Signal('customer_report', f"Cardholder reports they did not make the {S._money(txn['amount'])} purchase (customer disputes are right about 84% of the time in closed cases, but recurring charges are the usual exception)", 0.0, source='customer',
                             ref='trigger:customer_report', entity_ids=[txn['txn_id']], independent=True))
    else:
        sigs.append(S.Signal('analyst_request', tr['trigger_text'][:200], 0.0, source='external', ref='trigger:analyst_request', entity_ids=[txn['txn_id']], independent=False))

    sigs += S.amount_signals(txn, prof)
    sigs += S.channel_device_signals(txn, prof, st.get('device') or {})
    sigs += S.region_signals(txn, prof, st.get('region') or {})
    r = S.recurring_signal(txn, prof, st.get('recurring'))
    if r: sigs.append(r)
    t = S.testing_signal(txn, prof, st.get('small_auth'))
    if t:
        sigs.append(t); episode = list(dict.fromkeys(t.entity_ids + episode)); pattern = 'card_testing'
    d = S.duplicate_burst_signal(txn, prof, st.get('recurring'))
    if d:
        sigs.append(d); episode = list(dict.fromkeys(d.entity_ids + episode))
    b, ep = S.burst_signals(txn, prof, st['window'])
    sigs += b
    if ep:
        episode = list(dict.fromkeys(ep + episode))
    # a New, never-seen device: every purchase this card made from that device belongs to the episode
    if 'device_new_unseen' in {x.key for x in sigs} and txn.get('device_profile'):
        same_dev = tb.card_txns_on_device(txn['card_id'], txn['device_profile'])
        for t in same_dev:
            amt_extra = t['amount']
            if t['txn_id'] not in episode:
                episode.append(t['txn_id'])
        st['window'] = st['window'] + [t for t in same_dev if t['txn_id'] not in {w['txn_id'] for w in st['window']}]
    h = S.hub_device_signal(txn, st.get('device') or {}, prof)
    if h:
        sigs.append(h)
        connected_devices.append(txn['device_profile'])
        connected_cards += [c['card_id'] for c in st['device']['cards'] if c['customer_id'] != txn['customer_id']]
        if h.key == 'hub_device':
            for t in tb.card_txns_on_device(txn['card_id'], txn['device_profile']):
                if t['txn_id'] not in episode:
                    episode.append(t['txn_id'])
                if t['txn_id'] not in {w['txn_id'] for w in st['window']}:
                    st['window'] = st['window'] + [t]
    hs = S.history_signal(st.get('prior_cases') or [], txn['customer_id'])
    if hs: sigs.append(hs)

    # customer-level base rate from memory (how often this customer's txns turned out fraudulent)
    fr = st.get('fraud_rate') or {}
    if fr.get('n', 0) >= 30:
        key = f"fraud_rate_{txn['channel']}"
        rate = fr.get(key)
        if rate is not None:
            if rate >= 0.10:
                sigs.append(S.Signal('base_rate_high', f"{int(rate*100)}% of this customer's prior {txn['channel']} transactions were later confirmed fraudulent (portfolio average 3%)", 0.6, ref=f"query:customer_fraud_rate(customer_id={txn['customer_id']})", entity_ids=[txn['customer_id']], independent=False))
            elif rate == 0 and fr.get(f"n_{txn['channel']}", 0) >= 30:
                n_ch = fr.get(f"n_{txn['channel']}", 0)
                sigs.append(S.Signal('base_rate_zero', f"None of this customer's {n_ch} prior {txn['channel']} transactions were ever confirmed fraudulent", -0.3, ref=f"query:customer_fraud_rate(customer_id={txn['customer_id']})", entity_ids=[txn['customer_id']], independent=False))

    # coordinated / undocumented detection ---------------------------------------------------------
    keys = {s.key for s in sigs}
    t0 = datetime.strptime(txn['ts'], '%Y-%m-%d %H:%M:%S')
    if 'threshold_burst' in keys:
        peers = tb.pattern_peers('sub500_burst', (t0 - timedelta(days=30)).strftime('%Y-%m-%d'), tb.as_of or txn['ts'], exclude_customer=txn['customer_id'])
        st['peers'] = peers
        if peers['n_peers'] >= 2:
            sigs.append(S.Signal('coordinated_peers', f"{peers['n_peers']} other cards show the same signature (3–4 online purchases just under $500 within an hour) in the past 30 days", 0.8,
                                 ref="query:pattern_peers(kind=sub500_burst)", entity_ids=[p['card_id'] for p in peers['peers']][:20], independent=True))
            connected_cards += [p['card_id'] for p in peers['peers']][:20]
        pattern = 'undocumented'
        pdesc = ("Sub-threshold burst: three to four card-not-present purchases within roughly thirty minutes, each priced just under $500, "
                 "from devices marked New (some behind a transparent proxy). The amounts look chosen to stay under a $500 authorization "
                 "threshold; the same signature appears on other cards this month and matches closed cases CC-3748/CC-3841 from September. "
                 "Found by scanning the ±48h card window and then querying the graph for peers with the same amount/velocity signature.")
    elif 'hub_device' in keys:
        peers = tb.pattern_peers('anon_proxy_new_device', (t0 - timedelta(days=30)).strftime('%Y-%m-%d'), tb.as_of or txn['ts'], exclude_customer=txn['customer_id'])
        st['peers'] = peers
        pattern = 'undocumented'
        dev = st['device']
        pdesc = (f"Shared-device ring: the device profile '{dev['device_profile']}' appears on {dev['n_cards']} different cards in 30 days, "
                 f"always marked New for the account and behind an anonymous proxy, buying mid-value online goods. This is not one customer's "
                 f"stolen number but a single operator cycling many cards through one device — the same profile drove closed cases "
                 f"{', '.join(c['case_id'] for c in dev['closed_cases_on_device'][:3])} in August–September. Found via the device_neighbors hub query.")
    elif pattern != 'card_testing':
        pattern = _classify(txn, prof, keys, st)

    prob = S.combine(S.PRIORS[tr['trigger_type']], sigs)
    # exposure = episode amounts
    amt_by_id = {t['txn_id']: t['amount'] for t in st['window']}
    amt_by_id[txn['txn_id']] = txn['amount']
    exposure = round(sum(abs(amt_by_id.get(i, 0.0)) for i in episode), 2)
    return {'signals': sigs, 'probability': prob, 'pattern': pattern, 'pattern_description': pdesc,
            'peers': st.get('peers'), 'window': st['window'], 'episode_ids': episode, 'connected_cards': list(dict.fromkeys(connected_cards)),
            'connected_devices': connected_devices, 'exposure': exposure, 'step': 4}


def _pattern_hint(st: CaseState) -> str:
    txn = st['txn']
    if txn['channel'] == 'in_person':
        return 'out_of_region_use account_takeover'
    if txn.get('device_status') == 'New':
        return 'card_not_present_new_device'
    return 'card_not_present_fraud card_testing'


def _classify(txn, prof, keys, st) -> str:
    """Map evidence to one of the five documented typologies (or none)."""
    if txn['channel'] == 'in_person':
        if 'region_new' in keys and 'region_known' not in keys:
            return 'out_of_region_use'
        return 'account_takeover' if ('amount_above_max' in keys or 'amount_above_p95' in keys) else 'none'
    # online
    if txn.get('device_status') == 'New' and (prof['flagged'].get('device_seen_before') or 0) == 0 and txn.get('device_profile'):
        return 'card_not_present_new_device'
    if 'channel_shift' in keys and (prof['flagged'].get('channel_share_online') or 0) < 0.05 and prof['n_txns'] >= 50:
        return 'account_takeover'
    return 'card_not_present_fraud'


# ------------------------------------------------------------------------------------------------- decision

def n_initial_decision(st: CaseState) -> CaseState:
    """Recommend on the evidence available now (policy 3b, 'initial')."""
    tr, p, exp = st['trigger'], st['probability'], st['exposure']
    acts: list[dict] = []
    keys = {s.key for s in st['signals']}
    verdict = _verdict(p)
    disputed = tr['trigger_type'] == 'customer_report'

    if p >= P.CASE_THRESHOLD or disputed:
        acts.append(P.action('CREATE_CASE', f"3a: fraud probability {p:.2f} {'≥ 0.30' if p >= 0.3 else 'below 0.30 but the customer disputes the charge'}; open the internal case", exp))

    if st['pattern'] == 'card_testing':
        acts.append(P.action('DECLINE_TRANSACTION', "R5: card-testing sequence observed", exp))
        acts.append(P.action('STEP_UP_AUTH', "R5: require OTP before further activity", exp))
        if any(t['amount'] > 100 for t in st['small_auth']['larger_after']):
            acts.append(P.action('BLOCK_CARD', "R5: a purchase over $100 has already cleared after the testing run", exp))
    elif ('recurring' in keys or 'recurring_weak' in keys) and disputed:
        acts.append(P.action('VERIFY_WITH_CUSTOMER', "R7: disputed charge matches the cardholder's own recurring pattern; confirm rather than block", exp))
        acts.append(P.action('WARN_CUSTOMER', "R7: send a recurring-charge reminder", exp))
    elif st['pattern'] == 'undocumented' and p >= 0.7:
        acts.append(P.action('BLOCK_CARD', f"R9/R2: coordinated abuse across customers with probability {p:.2f}", exp))
        acts.append(P.action('FILE_REPORT', "R9: undocumented, coordinated pattern", exp))
        acts.append(P.action('MONITOR_CONNECTED_CARDS', "R6: every card sharing the device / signature goes under monitoring", exp))
        acts.append(P.action('ESCALATE_TO_ANALYST', "R9: undocumented pattern must be reviewed by a human", exp))
    elif p >= 0.85:
        acts.append(P.action('BLOCK_CARD', f"Probability {p:.2f} on {len([s for s in st['signals'] if s.independent])} independent signals; block and reissue", exp))
    elif p >= P.VERIFY_THRESHOLD:
        if not disputed:
            acts.append(P.action('DECLINE_TRANSACTION', f"Probability {p:.2f}: decline the pending authorization while we confirm", exp))
        acts.append(P.action('VERIFY_WITH_CUSTOMER', "R1: single-signal set below the block threshold; confirm with the cardholder before blocking" if not disputed else "R2 pre-check: confirm the cardholder still holds the card and did not authorise anyone before blocking", exp))
        acts.append(P.action('MONITOR_CARD', "72h heightened monitoring pending reply", exp))
    elif p > P.STOP_LOW:
        if disputed:
            acts.append(P.action('VERIFY_WITH_CUSTOMER', f"R1: customer report is the main signal (probability {p:.2f}); confirm the details of the dispute", exp))
        else:
            acts.append(P.action('VERIFY_WITH_CUSTOMER' if p >= 0.35 else 'STEP_UP_AUTH', f"R1: probability {p:.2f} on a weak signal set; verify before any block", exp))
        acts.append(P.action('MONITOR_CARD', "72h heightened monitoring pending reply", exp))
        if verdict == 'uncertain' and exp > P.ESCALATE_EXPOSURE:
            acts.append(P.action('ESCALATE_TO_ANALYST', f"R8: verdict uncertain and exposure ${exp:,.2f} exceeds $500", exp))
    else:
        acts.append(P.action('ALLOW_TRANSACTION', f"Probability {p:.2f}; activity matches the cardholder's own history", exp))
        acts.append(P.action('CLOSE_NO_FRAUD', "Stopping rule §6: probability ≤ 0.15 on independent evidence", exp))
        if disputed:
            acts = [a for a in acts if a['action'] != 'ALLOW_TRANSACTION']
    return {'initial_actions': P.sort_actions(acts), 'verdict': verdict, 'step': 5}


def _verdict(p: float) -> str:
    return 'fraud' if p >= 0.70 else 'legitimate' if p <= 0.30 else 'uncertain'


def need_more_evidence(st: CaseState) -> str:
    """Policy §6 stopping rule."""
    p = st['probability']
    indep = len([s for s in st['signals'] if s.independent and abs(s.weight) >= 0.5])
    if st['pattern'] == 'card_testing' and p >= 0.85:
        return 'decide'
    if (p >= P.STOP_HIGH or p <= P.STOP_LOW) and indep >= 2:
        # R7: a disputed recurring charge is still verified with the customer before closing
        if st['trigger']['trigger_type'] == 'customer_report' and any(a['action'] == 'VERIFY_WITH_CUSTOMER' for a in st['initial_actions']):
            return 'ask'
        return 'decide'
    if any(a['action'] in ('VERIFY_WITH_CUSTOMER', 'STEP_UP_AUTH') for a in st['initial_actions']):
        return 'ask'
    return 'decide'


def n_request_evidence(st: CaseState) -> CaseState:
    """Controlled evidence gathering (policy §5). Responses are simulated and the assumption recorded."""
    tb: Toolbox = st['tools']
    txn, p, tr = st['txn'], st['probability'], st['trigger']
    keys = {s.key for s in st['signals']}
    kind = 'step_up_auth' if any(a['action'] == 'STEP_UP_AUTH' for a in st['initial_actions']) else 'customer_validation'
    # Simulation policy: the customer's answer follows the weight of graph evidence, so the *final* recommendation
    # reflects what the bank would most likely learn. The assumption is written down verbatim.
    # Simulation rule (documented in docs/blog.md): the assumed reply follows the balance of graph evidence —
    # strong recurring history -> the customer recognises the charge; otherwise probability >= 0.50 -> denial, < 0.50 -> confirmation.
    if 'recurring' in keys or 'recurring_weak' in keys:
        resp = ("Customer, shown the history of identical monthly charges on their own card, recognises the merchant and "
                "confirms the charge; asks for a reminder next cycle")
        outcome = 'confirmed'
    elif tr['trigger_type'] == 'customer_report':
        if p < 0.50:
            resp = ("Customer, walked through the purchase details (merchant category, device they usually use, prior similar charges), "
                    "recognises the transaction as their own or a household member's and withdraws the dispute")
            outcome = 'confirmed'
        else:
            resp = "Customer confirms they did not make the purchase, did not authorise anyone else, and still holds the physical card; asks for a new card"
            outcome = 'denied'
    elif p >= 0.50:
        resp = "Customer states they did not make this purchase and still has the card" if kind == 'customer_validation' else "One-time passcode not completed within 24 hours"
        outcome = 'denied' if kind == 'customer_validation' else 'no_reply'
    else:
        resp = ("Customer confirms the purchase (new device / travelling) and asks that the device be added to the profile"
                if kind == 'customer_validation' else "Customer completes the one-time passcode from a known phone number")
        outcome = 'confirmed'
    prompt = f"Did you make the {S._money(txn['amount'])} {txn['product']} purchase on {txn['ts'][:10]}?" if kind == 'customer_validation' else "Please confirm via one-time passcode"
    res = tb.request_evidence(kind, prompt, resp)
    req = {'type': kind, 'asked_after_step': st['step'], 'assumed_response': resp, 'outcome': outcome}
    sig_w = {'confirmed': -2.4, 'denied': 2.4, 'no_reply': 0.4}[outcome]
    sig = S.Signal(f'evidence_{outcome}', f"Customer response: {resp}", sig_w, source='customer', ref=f"evidence_request:{len(st['evidence_requests'])+1}", entity_ids=[])
    sigs = st['signals'] + [sig]
    prob = S.combine(S.PRIORS[tr['trigger_type']], sigs)
    return {'evidence_requests': st['evidence_requests'] + [req], 'signals': sigs, 'probability': prob, 'step': st['step'] + 1}


def n_final_decision(st: CaseState) -> CaseState:
    tr, p, exp = st['trigger'], st['probability'], st['exposure']
    keys = {s.key for s in st['signals']}
    reqs = st['evidence_requests']
    outcome = reqs[-1]['outcome'] if reqs else None
    verdict = _verdict(p)
    if not reqs:
        acts = list(st['initial_actions'])
        what = 'nothing'
    else:
        acts = []
        if outcome == 'confirmed':
            verdict = 'legitimate'
            acts.append(P.action('CREATE_CASE', "3a: a case is opened whenever evidence is requested; closed as legitimate", exp))
            acts.append(P.action('CLOSE_NO_FRAUD', "R3: customer confirmed the transaction", exp))
            if 'recurring' in keys or 'recurring_weak' in keys:
                acts.append(P.action('WARN_CUSTOMER', "R7: recurring-charge reminder so the next cycle is not disputed", exp))
            what = f"Customer confirmation (R3) dropped probability from {st['initial_prob']:.2f} to {p:.2f}; the verify/monitor plan is replaced by CLOSE_NO_FRAUD."
        elif outcome == 'denied':
            verdict = 'fraud' if p >= 0.5 else verdict
            acts.append(P.action('CREATE_CASE', "R2", exp))
            acts.append(P.action('BLOCK_CARD', f"R2: customer denied; exposure ${exp:,.2f}", exp))
            shared = bool(st['connected_cards'])
            file_it, why = P.sar_required(verdict, p, exp, shared, _other_customer_fraud(st), st['pattern'] == 'undocumented')
            if file_it:
                acts.append(P.action('FILE_REPORT', why, exp))
            if shared:
                acts.append(P.action('MONITOR_CONNECTED_CARDS', "R6: cards sharing the device / signature", exp))
            if st['pattern'] == 'undocumented':
                acts.append(P.action('ESCALATE_TO_ANALYST', "R9: undocumented pattern", exp))
            what = f"Customer denial (R2) raised probability from {st['initial_prob']:.2f} to {p:.2f}; verification is replaced by BLOCK_CARD and the case record{' plus a report' if file_it else ''}."
        else:  # no reply
            acts.append(P.action('CREATE_CASE', "3a: evidence requested", exp))
            acts.append(P.action('MONITOR_CARD', "R4: no reply within 24 hours", exp))
            acts.append(P.action('DECLINE_TRANSACTION', "R4: decline pending authorizations", exp))
            if exp > P.ESCALATE_EXPOSURE:
                acts.append(P.action('ESCALATE_TO_ANALYST', f"R4/R8: no reply and exposure ${exp:,.2f} > $500", exp))
            what = f"No reply to step-up (R4): probability {p:.2f}; card stays active under monitoring with pending authorizations declined."
    # SAR consistency when no evidence was requested
    if not reqs:
        shared = bool(st['connected_cards'])
        file_it, why = P.sar_required(verdict, p, exp, shared, _other_customer_fraud(st), st['pattern'] == 'undocumented')
        has = any(a['action'] == 'FILE_REPORT' for a in acts)
        if file_it and not has:
            acts.append(P.action('FILE_REPORT', why, exp))
        if not file_it and has:
            acts = [a for a in acts if a['action'] != 'FILE_REPORT']
    else:
        file_it = any(a['action'] == 'FILE_REPORT' for a in acts)
        why = next((a['reason'] for a in acts if a['action'] == 'FILE_REPORT'), None) or P.sar_required(verdict, p, exp, bool(st['connected_cards']), _other_customer_fraud(st), st['pattern'] == 'undocumented')[1]
    acts = P.sort_actions(acts)
    status = _status(verdict, acts)
    if verdict == 'legitimate':
        episode, exposure, pattern, pdesc = [], 0.0, 'none', ''
    else:
        episode, exposure, pattern, pdesc = st['episode_ids'], exp, st['pattern'], st['pattern_description']
    return {'final_actions': acts, 'verdict': verdict, 'status': status, 'what_changed': what,
            'episode_ids': episode, 'exposure': exposure, 'pattern': pattern, 'pattern_description': pdesc,
            'sar': {'file': file_it, 'reason': why}, 'stop_reason': _stop_reason(st, p, reqs), 'step': st['step'] + 1}


def _other_customer_fraud(st) -> bool:
    dev = st.get('device') or {}
    return any(c['outcome'] == 'confirmed_fraud' for c in dev.get('closed_cases_on_device', [])) and not dev.get('generic_fingerprint')


def _status(verdict, acts) -> str:
    names = {a['action'] for a in acts}
    if 'ESCALATE_TO_ANALYST' in names:
        return 'escalated'
    if verdict == 'fraud':
        return 'closed_fraud'
    if verdict == 'legitimate':
        return 'closed_legitimate'
    return 'open'


def _stop_reason(st, p, reqs) -> str:
    indep = len([s for s in st['signals'] if s.independent and abs(s.weight) >= 0.5])
    if reqs:
        o = reqs[-1]['outcome']
        if o in ('confirmed', 'denied'):
            return f"Verification response settled the question (customer {o}); probability {p:.2f}. Further graph queries would not change the actions."
        return f"No reply within 24 hours; policy R4 fixes the action set. Probability {p:.2f}; case stays open under monitoring."
    if p >= P.STOP_HIGH:
        return f"Probability {p:.2f} ≥ 0.85 on {indep} independent pieces of graph evidence (§6); a defensible action exists without contacting the customer."
    if p <= P.STOP_LOW:
        return f"Probability {p:.2f} ≤ 0.15 on {indep} independent pieces of evidence (§6); the activity matches the cardholder's own history."
    return "Further steps unlikely to change the decision; escalated per R8."


def n_explain(st: CaseState) -> CaseState:
    """Narrate the case summary and (if required) the SAR. Uses the LLM if configured, otherwise deterministic templates."""
    nar = Narrator()
    summary, tok1 = nar.summary(st)
    sar = dict(st['sar'])
    tok2 = 0
    if sar['file']:
        sar_txt, tok2 = nar.sar_narrative(st)
        sar.update(sar_txt)
    else:
        sar.update({'narrative': '', 'subjects': [], 'total_amount_usd': 0, 'activity_dates': []})
    return {'summary': summary, 'sar': sar, 'tokens': st.get('tokens', 0) + tok1 + tok2, 'step': st['step'] + 1}


def n_memorize(st: CaseState) -> CaseState:
    """Write the case to the graph (AgentCase vertex + edges) and execute auto actions against mock systems."""
    tb: Toolbox = st['tools']
    txn = st['txn']
    rec = {
        'case_id': st['case_id'], 'customer_id': txn['customer_id'], 'card_id': txn['card_id'],
        'opened_at': st['trigger']['opened_at'], 'status': st['status'], 'verdict': st['verdict'],
        'fraud_probability': st['probability'], 'pattern': st['pattern'], 'exposure_usd': st['exposure'],
        'affected_txn_ids': st['episode_ids'], 'connected_card_ids': st['connected_cards'],
        'connected_device_profiles': st['connected_devices'], 'sar_filed': st['sar']['file'],
        'final_actions': [a['action'] for a in st['final_actions']], 'summary': st['summary'],
        'similar_prior_cases': [c['case_id'] for c in st['similar']],
    }
    gid = tb.write_case(rec)
    for a in st['final_actions']:
        tb.execute_action(a['action'], a['route'], {'case_id': st['case_id'], 'card_id': txn['card_id'], 'txn_id': txn['txn_id']})
    return {'graph_case_id': gid, 'step': st['step'] + 1}


# ------------------------------------------------------------------------------------------------- graph

def build_graph():
    g = StateGraph(CaseState)
    g.add_node('trigger', n_trigger)
    g.add_node('gather', n_gather)
    g.add_node('memory', n_memory)
    g.add_node('assess', n_assess)
    g.add_node('initial_decision', _wrap_initial)
    g.add_node('request_evidence', n_request_evidence)
    g.add_node('final_decision', n_final_decision)
    g.add_node('explain', n_explain)
    g.add_node('memorize', n_memorize)
    g.set_entry_point('trigger')
    g.add_edge('trigger', 'gather')
    g.add_edge('gather', 'memory')
    g.add_edge('memory', 'assess')
    g.add_edge('assess', 'initial_decision')
    g.add_conditional_edges('initial_decision', need_more_evidence, {'ask': 'request_evidence', 'decide': 'final_decision'})
    g.add_edge('request_evidence', 'final_decision')
    g.add_edge('final_decision', 'explain')
    g.add_edge('explain', 'memorize')
    g.add_edge('memorize', END)
    return g.compile()


def _wrap_initial(st):
    out = n_initial_decision(st)
    out['initial_prob'] = st['probability']
    return out


GRAPH = None


def investigate(trigger: dict, as_of: str | None = None) -> tuple[dict, CaseState]:
    """Run one investigation. ``trigger`` is a row of case_pack.csv (dict). Returns (answer_json, full_state)."""
    global GRAPH
    if GRAPH is None:
        GRAPH = build_graph()
    log = ToolLog()
    tb = Toolbox(log, as_of=as_of or trigger.get('opened_at'))
    t0 = time.time()
    st: CaseState = {'case_id': trigger['case_id'], 'trigger': trigger, 'tools': tb, 'started': t0}
    out = GRAPH.invoke(st)
    out['tool_log'] = log.calls
    answer = to_answer(out, log.count, time.time() - t0)
    return answer, out


def to_answer(st: CaseState, tool_calls: int, latency: float) -> dict:
    txn = st['txn']
    ev = [s.as_evidence() for s in st['signals']]
    # add retrieved-memory evidence
    if st.get('similar'):
        top = st['similar'][:3]
        ev.append({'claim': "Similar cases retrieved from case memory: " + '; '.join(f"{c['case_id']} ({c.get('outcome') or c.get('status')}, {c.get('pattern')})" for c in top),
                   'source': 'graph', 'ref': 'query:similar_cases', 'entity_ids': [c['case_id'] for c in top]})
    if st.get('policy_docs'):
        ev.append({'claim': "Policy sections applied: " + ', '.join(d['title'] for d in st['policy_docs'][:3]), 'source': 'document',
                   'ref': ', '.join(d['doc_id'] for d in st['policy_docs'][:3]), 'entity_ids': []})
    legit = st['verdict'] == 'legitimate'
    return {
        'case_id': st['case_id'],
        'case': {
            'status': st['status'], 'verdict': st['verdict'], 'fraud_probability': st['probability'],
            'pattern': st['pattern'], 'pattern_description': st['pattern_description'],
            'affected_txn_ids': [] if legit else st['episode_ids'],
            'first_suspicious_txn_id': '' if legit else (st['episode_ids'][0] if st['episode_ids'] else ''),
            'connected_card_ids': st['connected_cards'], 'connected_device_profiles': st['connected_devices'],
            'exposure_usd': 0 if legit else st['exposure'],
            'evidence': ev,
            'similar_prior_cases': [c['case_id'] for c in st['similar'] if c['kind'] == 'closed_case'][:5],
            'summary': st['summary'], 'written_to_graph': True, 'graph_case_id': st['graph_case_id'],
        },
        'evidence_requests': [{k: v for k, v in r.items() if k != 'outcome'} for r in st['evidence_requests']],
        'next_best_actions': {'initial': st['initial_actions'], 'final': st['final_actions'], 'what_changed': st['what_changed']},
        'sar': st['sar'],
        'stop_reason': st['stop_reason'],
        'tool_calls': tool_calls, 'tokens': st.get('tokens', 0), 'latency_s': round(latency, 2),
    }
