"""Narration layer: case summary + SAR narrative.

If ``OPENAI_API_KEY`` is set the LLM writes the prose from a structured evidence pack (GraphRAG: graph facts +
retrieved policy text are passed as context, never raw rows). Without a key we fall back to deterministic templates
so the benchmark is reproducible; ``tokens`` is then reported as 0.
"""
from __future__ import annotations

import json
from datetime import datetime

from . import config


def _money(x): return f"${x:,.2f}"


class Narrator:
    def __init__(self):
        self.client = None
        if config.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            except Exception:
                self.client = None

    # ------------------------------------------------------------------------------------ LLM path
    def _llm(self, system: str, user: str, max_tokens=500) -> tuple[str, int]:
        r = self.client.chat.completions.create(model=config.OPENAI_MODEL, temperature=0.2, max_tokens=max_tokens,
                                                messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
        return r.choices[0].message.content.strip(), int(r.usage.total_tokens)

    def _pack(self, st) -> dict:
        txn = st['txn']
        return {
            'case_id': st['case_id'], 'trigger': st['trigger']['trigger_text'], 'customer_id': txn['customer_id'], 'card_id': txn['card_id'],
            'flagged': {k: txn[k] for k in ('txn_id', 'ts', 'amount', 'product', 'channel', 'region', 'device_profile', 'device_status', 'proxy')},
            'verdict': st['verdict'], 'probability': st['probability'], 'pattern': st['pattern'], 'exposure_usd': st['exposure'],
            'affected_txn_ids': st['episode_ids'], 'connected_cards': st['connected_cards'][:15],
            'evidence': [s.as_evidence() for s in st['signals']],
            'evidence_requests': st['evidence_requests'], 'final_actions': st['final_actions'],
            'policy_context': [d['text'] for d in st.get('policy_docs', [])[:3]],
            'similar_cases': [{'id': c['case_id'], 'outcome': c.get('outcome'), 'pattern': c.get('pattern')} for c in st.get('similar', [])[:3]],
        }

    # ------------------------------------------------------------------------------------ summary
    def summary(self, st) -> tuple[str, int]:
        if self.client:
            try:
                txt, tok = self._llm("You are a bank fraud analyst. Write a 2–6 sentence case summary another analyst could read. "
                                     "Use only the facts provided; cite policy rule numbers where relevant; no headings.",
                                     json.dumps(self._pack(st), default=str), 350)
                return txt, tok
            except Exception:
                pass
        return self._template_summary(st), 0

    def _template_summary(self, st) -> str:
        txn, p = st['txn'], st['probability']
        strong = [s for s in st['signals'] if abs(s.weight) >= 0.6]
        strong.sort(key=lambda s: -abs(s.weight))
        lead = {'fraud': 'Assessed as fraud', 'legitimate': 'Assessed as legitimate', 'uncertain': 'Verdict uncertain'}[st['verdict']]
        pat = st['pattern'] if st['pattern'] not in ('none',) else 'no fraud pattern'
        s1 = (f"{lead} (probability {p:.2f}, {pat.replace('_', ' ')}). Flagged {txn['channel'].replace('_', '-')} {txn['product']} transaction "
              f"{txn['txn_id']} of {_money(txn['amount'])} on card {txn['card_id']} at {txn['ts']}.")
        s2 = ' '.join(f"{s.claim}." for s in strong[:3])
        req = st['evidence_requests']
        s3 = ''
        if req:
            s3 = f" Evidence requested ({req[-1]['type'].replace('_', ' ')}); assumed response: {req[-1]['assumed_response']}."
        acts = ', '.join(a['action'] for a in st['final_actions'])
        s4 = f" Exposure {_money(st['exposure'])} across {len(st['episode_ids'])} transaction(s)." if st['verdict'] != 'legitimate' else ''
        s5 = f" Final actions: {acts}."
        return (s1 + ' ' + s2 + s3 + s4 + s5).strip()

    # ------------------------------------------------------------------------------------ SAR
    def sar_narrative(self, st) -> tuple[dict, int]:
        txn = st['txn']
        amt_by_id = {t['txn_id']: t for t in st['window']}
        amt_by_id[txn['txn_id']] = txn
        ep = [amt_by_id[i] for i in st['episode_ids'] if i in amt_by_id]
        dates = sorted({e['ts'][:10] for e in ep}) or [txn['ts'][:10]]
        subjects = [txn['customer_id'], txn['card_id']] + st['connected_cards'][:25] + st['connected_devices']
        base = {'subjects': subjects, 'total_amount_usd': st['exposure'], 'activity_dates': [dates[0], dates[-1]]}
        if self.client:
            try:
                txt, tok = self._llm("You write FinCEN-compliant SAR narratives. Six to twelve sentences, chronological, standing on its own: "
                                     "who (customer, cards, devices), what, when (dates), where (regions/channels), how, why suspicious, "
                                     "total amount, action taken. Use only the facts given.", json.dumps(self._pack(st), default=str), 600)
                base['narrative'] = txt
                return base, tok
            except Exception:
                pass
        base['narrative'] = self._template_sar(st, ep, dates)
        return base, 0

    def _template_sar(self, st, ep, dates) -> str:
        txn = st['txn']
        where = f"billing region {int(txn['region'])}" if txn.get('region') is not None else "an online channel with no billing region recorded"
        dev = txn.get('device_profile')
        lines = []
        lines.append(f"Between {dates[0]} and {dates[-1]}, card {txn['card_id']} held by customer {txn['customer_id']} was used for "
                     f"{len(ep)} {txn['channel'].replace('_', '-')} transaction(s) totalling {_money(st['exposure'])} "
                     f"({', '.join(_money(e['amount']) for e in ep[:6])}) under product code {txn['product']} in {where}.")
        if dev:
            lines.append(f"The transactions were submitted from device profile '{dev}'" +
                         (f", recorded as New for this account" if txn.get('device_status') == 'New' else '') +
                         (f" and connecting through {txn['proxy']}" if txn.get('proxy') else '') + ".")
        for s in sorted(st['signals'], key=lambda s: -s.weight)[:4]:
            if s.weight > 0.5 and s.source == 'graph':
                lines.append(s.claim + '.')
        if st['connected_cards']:
            lines.append(f"The same device profile or transaction signature appears on {len(st['connected_cards'])} other card(s) in the same period, "
                         f"including {', '.join(st['connected_cards'][:5])}, indicating a common actor across multiple cardholders.")
        dev_cases = [c for c in (st.get('device') or {}).get('closed_cases_on_device', []) if c['outcome'] == 'confirmed_fraud']
        if dev_cases:
            lines.append(f"This device profile was previously recorded on closed fraud case(s) {', '.join(c['case_id'] for c in dev_cases[:3])}.")
        req = st['evidence_requests']
        if req and req[-1]['outcome'] == 'denied':
            lines.append("The cardholder, contacted the same day, stated that they did not make these purchases and remained in possession of the card.")
        if st['pattern'] == 'undocumented':
            lines.append("The activity does not match a documented typology: " + st['pattern_description'].split('.')[0] + '.')
        else:
            lines.append(f"The activity is consistent with {st['pattern'].replace('_', ' ')} and is inconsistent with the cardholder's established history.")
        acts = [a['action'] for a in st['final_actions']]
        taken = []
        if 'BLOCK_CARD' in acts: taken.append('the card has been blocked and scheduled for reissue')
        if 'MONITOR_CONNECTED_CARDS' in acts: taken.append('connected cards have been placed under monitoring')
        if 'ESCALATE_TO_ANALYST' in acts: taken.append('the case has been escalated to a fraud analyst')
        lines.append(f"Total suspected unauthorised amount: {_money(st['exposure'])}. " + ('; '.join(taken).capitalize() + '.' if taken else ''))
        return ' '.join(lines)
