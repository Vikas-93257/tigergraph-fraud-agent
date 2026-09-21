"""Validate cases/*.json against the answer format in the dataset README.

    python answer_schema.py            # validates all files in cases/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STATUSES = {'open', 'closed_fraud', 'closed_legitimate', 'escalated'}
VERDICTS = {'fraud', 'legitimate', 'uncertain'}
PATTERNS = {'card_testing', 'card_not_present_fraud', 'card_not_present_new_device', 'out_of_region_use',
            'account_takeover', 'undocumented', 'none'}
SOURCES = {'graph', 'document', 'customer', 'external'}
EVIDENCE_TYPES = {'customer_validation', 'step_up_auth', 'analyst_info'}
ACTIONS = {'ALLOW_TRANSACTION', 'DECLINE_TRANSACTION', 'MONITOR_CARD', 'MONITOR_CONNECTED_CARDS', 'WARN_CUSTOMER',
           'VERIFY_WITH_CUSTOMER', 'STEP_UP_AUTH', 'BLOCK_CARD', 'BLOCK_ALL_CARDS', 'GENERATE_REPORT', 'CREATE_CASE',
           'FILE_REPORT', 'ESCALATE_TO_ANALYST', 'CLOSE_NO_FRAUD'}
ROUTES = {'auto', 'L1', 'L2'}


def validate(a: dict) -> list[str]:
    err = []
    req = lambda d, k, t=None: err.append(f'missing {k}') if k not in d else (err.append(f'{k} wrong type') if t and not isinstance(d[k], t) else None)
    for k, t in [('case_id', str), ('case', dict), ('evidence_requests', list), ('next_best_actions', dict), ('sar', dict),
                 ('stop_reason', str), ('tool_calls', int), ('tokens', int), ('latency_s', (int, float))]:
        req(a, k, t)
    c = a.get('case', {})
    for k, t in [('status', str), ('verdict', str), ('fraud_probability', (int, float)), ('pattern', str), ('pattern_description', str),
                 ('affected_txn_ids', list), ('connected_card_ids', list), ('connected_device_profiles', list),
                 ('exposure_usd', (int, float)), ('evidence', list), ('similar_prior_cases', list), ('summary', str),
                 ('written_to_graph', bool), ('graph_case_id', str)]:
        req(c, k, t)
    if 'first_suspicious_txn_id' not in c:
        err.append('missing first_suspicious_txn_id')
    if c.get('status') not in STATUSES: err.append(f"bad status {c.get('status')}")
    if c.get('verdict') not in VERDICTS: err.append(f"bad verdict {c.get('verdict')}")
    if c.get('pattern') not in PATTERNS: err.append(f"bad pattern {c.get('pattern')}")
    if not (0 <= float(c.get('fraud_probability', -1)) <= 1): err.append('probability out of range')
    if c.get('verdict') == 'fraud' and not c.get('affected_txn_ids'): err.append('fraud verdict without affected_txn_ids')
    if c.get('verdict') == 'legitimate' and c.get('affected_txn_ids'): err.append('legitimate verdict with affected_txn_ids')
    if c.get('verdict') == 'fraud' and c.get('pattern') == 'none': err.append('fraud verdict with pattern none')
    if c.get('verdict') == 'legitimate' and c.get('pattern') not in ('none',): err.append('legitimate verdict with a fraud pattern')
    if c.get('affected_txn_ids') and c.get('first_suspicious_txn_id') not in c['affected_txn_ids']:
        err.append('first_suspicious_txn_id not in affected_txn_ids')
    if not all(isinstance(x, str) for x in c.get('affected_txn_ids', [])): err.append('affected_txn_ids must be strings')
    for i, e in enumerate(c.get('evidence', [])):
        for k in ('claim', 'source', 'ref', 'entity_ids'):
            if k not in e: err.append(f'evidence[{i}] missing {k}')
        if e.get('source') not in SOURCES: err.append(f'evidence[{i}] bad source {e.get("source")}')
    for i, r in enumerate(a.get('evidence_requests', [])):
        if r.get('type') not in EVIDENCE_TYPES: err.append(f'evidence_requests[{i}] bad type')
        for k in ('asked_after_step', 'assumed_response'):
            if k not in r: err.append(f'evidence_requests[{i}] missing {k}')
    n = a.get('next_best_actions', {})
    for k in ('initial', 'final', 'what_changed'):
        if k not in n: err.append(f'next_best_actions missing {k}')
    for lst in ('initial', 'final'):
        for i, x in enumerate(n.get(lst, [])):
            if x.get('action') not in ACTIONS: err.append(f'{lst}[{i}] bad action {x.get("action")}')
            if x.get('route') not in ROUTES: err.append(f'{lst}[{i}] bad route {x.get("route")}')
            if not x.get('reason'): err.append(f'{lst}[{i}] missing reason')
    if not n.get('final'): err.append('final actions empty')
    s = a.get('sar', {})
    for k in ('file', 'reason', 'narrative', 'subjects', 'total_amount_usd', 'activity_dates'):
        if k not in s: err.append(f'sar missing {k}')
    if s.get('file'):
        if not s.get('narrative'): err.append('SAR filed without narrative')
        if 'FILE_REPORT' not in {x['action'] for x in n.get('final', [])}: err.append('sar.file true but FILE_REPORT not in final actions')
    else:
        if 'FILE_REPORT' in {x['action'] for x in n.get('final', [])}: err.append('FILE_REPORT in final actions but sar.file false')
    # policy consistency: exposure > $1,000 or coordinated/undocumented pattern needs a report
    if float(c.get('exposure_usd', 0)) > 1000 and c.get('verdict') == 'fraud' and not s.get('file'):
        err.append('exposure > $1,000 on a fraud verdict without SAR')
    if c.get('verdict') == 'fraud' and c.get('exposure_usd', 0) <= 0: err.append('fraud verdict with zero exposure')
    return err


def main(paths):
    bad = 0
    for p in paths:
        e = validate(json.loads(Path(p).read_text()))
        print(f"{Path(p).name}: {'OK' if not e else 'FAIL'}" + (''.join(f"\n   - {x}" for x in e)))
        bad += bool(e)
    print(f"\n{len(paths) - bad}/{len(paths)} valid")
    return bad


if __name__ == '__main__':
    files = sys.argv[1:] or sorted(str(p) for p in (Path(__file__).parent / 'cases').glob('HHG-*.json'))
    sys.exit(1 if main(files) else 0)
