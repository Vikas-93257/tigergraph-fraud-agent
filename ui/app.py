"""Investigator UI: FastAPI + a single static page.

    uvicorn ui.app:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /api/cases                   -> list of case_pack triggers + answer summaries
  GET  /api/cases/{id}              -> answer JSON + trace (signals, tool log, window)
  POST /api/cases/{id}/investigate  -> re-run the agent live for this case
  GET  /api/graph/{id}              -> node/edge neighbourhood for the graph view
  GET  /api/store                   -> backend description
  POST /api/adhoc/{txn_id}          -> investigate any transaction id (monitoring track)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraud_agent import config  # noqa: E402
from fraud_agent.tools import store  # noqa: E402
from run_cases import load_case_pack  # noqa: E402

app = FastAPI(title="Fraud Investigation Agent")
STATIC = ROOT / 'ui' / 'static'
TRACES = ROOT / 'traces'


def _answer(cid: str) -> dict | None:
    p = config.CASES_DIR / f"{cid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _trace(cid: str) -> dict | None:
    p = TRACES / f"{cid}.json"
    return json.loads(p.read_text()) if p.exists() else None


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/api/store')
def api_store():
    return store().describe()


@app.get('/api/cases')
def api_cases():
    out = []
    for trig in load_case_pack():
        a = _answer(trig['case_id'])
        row = {'case_id': trig['case_id'], 'trigger_type': trig['trigger_type'], 'opened_at': trig['opened_at'],
               'customer_id': trig['customer_id'], 'card_id': trig['card_id'], 'flagged_txn_id': trig['flagged_txn_id'],
               'trigger_text': trig['trigger_text']}
        if a:
            row |= {'verdict': a['case']['verdict'], 'fraud_probability': a['case']['fraud_probability'], 'pattern': a['case']['pattern'],
                    'exposure_usd': a['case']['exposure_usd'], 'status': a['case']['status'], 'sar': a['sar']['file'],
                    'final_actions': [x['action'] for x in a['next_best_actions']['final']]}
        out.append(row)
    return out


@app.get('/api/cases/{cid}')
def api_case(cid: str):
    a = _answer(cid)
    if not a:
        raise HTTPException(404, 'case not investigated yet')
    return {'answer': a, 'trace': _trace(cid)}


def _run(trig: dict) -> dict:
    from fraud_agent.agent import investigate
    answer, st = investigate(trig)
    config.CASES_DIR.mkdir(exist_ok=True); TRACES.mkdir(exist_ok=True)
    (config.CASES_DIR / f"{trig['case_id']}.json").write_text(json.dumps(answer, indent=2, default=str))
    trace = {'case_id': trig['case_id'], 'trigger': trig, 'txn': st.get('txn'), 'window': st.get('window'), 'profile': st.get('profile'),
             'device': st.get('device'), 'peers': st.get('peers'),
             'signals': [x.as_evidence() | {'weight': x.weight, 'key': x.key} for x in st.get('signals', [])],
             'initial_prob': st.get('initial_prob'), 'tool_log': st['tool_log']}
    (TRACES / f"{trig['case_id']}.json").write_text(json.dumps(trace, indent=1, default=str))
    return {'answer': answer, 'trace': trace}


@app.post('/api/cases/{cid}/investigate')
def api_investigate(cid: str):
    rows = {r['case_id']: r for r in load_case_pack()}
    if cid not in rows:
        raise HTTPException(404, 'unknown case')
    return _run(rows[cid])


@app.post('/api/adhoc/{txn_id}')
def api_adhoc(txn_id: int, trigger_type: str = 'analyst_request', note: str = ''):
    t = store().get_transaction(txn_id)
    if not t:
        raise HTTPException(404, 'unknown transaction')
    trig = {'case_id': f'ADHOC-{txn_id}', 'trigger_type': trigger_type, 'customer_id': t['customer_id'], 'card_id': t['card_id'],
            'flagged_txn_id': int(txn_id), 'opened_at': t['ts'], 'trigger_text': note or f'Ad-hoc review of {txn_id}',
            'risk_score': t.get('risk_score')}
    return _run(trig)


@app.get('/api/graph/{cid}')
def api_graph(cid: str):
    """Small neighbourhood for the visual: Customer -> Card -> flagged/episode txns -> Device -> other cards; closed cases."""
    a = _answer(cid); tr = _trace(cid)
    if not a or not tr:
        raise HTTPException(404, 'case not investigated yet')
    txn = tr['txn']; c = a['case']
    nodes, edges = {}, []

    def add(nid, kind, label, **extra):
        if nid not in nodes:
            nodes[nid] = {'id': nid, 'kind': kind, 'label': label, **extra}

    add(txn['customer_id'], 'customer', txn['customer_id'])
    add(txn['card_id'], 'card', txn['card_id'])
    edges.append((txn['customer_id'], txn['card_id'], 'OWNS'))
    affected = set(c['affected_txn_ids'])
    for t in (tr.get('window') or [])[:60]:
        tid = t['txn_id']
        add(tid, 'txn', f"${t['amount']:.0f}", amount=t['amount'], ts=t['ts'], channel=t['channel'],
            flagged=(tid == txn['txn_id']), affected=(tid in affected), product=t['product'])
        edges.append((txn['card_id'], tid, 'MADE'))
        if t.get('device_profile'):
            d = t['device_profile']
            add(d, 'device', d.split('|')[0].strip()[:22], full=d, status=t.get('device_status'))
            edges.append((tid, d, 'FROM_DEVICE'))
    if txn['txn_id'] not in nodes:
        add(txn['txn_id'], 'txn', f"${txn['amount']:.0f}", amount=txn['amount'], ts=txn['ts'], channel=txn['channel'], flagged=True, affected=True, product=txn['product'])
        edges.append((txn['card_id'], txn['txn_id'], 'MADE'))
    dev = tr.get('device') or {}
    if dev.get('device_profile') and dev.get('n_cards', 0) > 1:
        d = dev['device_profile']
        add(d, 'device', d.split('|')[0].strip()[:22], full=d)
        for k in dev['cards'][:25]:
            if k['card_id'] == txn['card_id']:
                continue
            add(k['card_id'], 'card', k['card_id'], other=True, n=k['n_txns'])
            edges.append((k['card_id'], d, 'USED'))
        for cc in dev.get('closed_cases_on_device', []):
            add(cc['case_id'], 'case', cc['case_id'], outcome=cc['outcome'], pattern=cc['pattern'])
            edges.append((cc['case_id'], d, 'INVOLVES_DEVICE'))
    for pr in (tr.get('peers') or {}).get('peers', [])[:12]:
        add(pr['card_id'], 'card', pr['card_id'], other=True, peer=True, n=pr.get('n'))
        add('SIG', 'signature', 'same signature')
        edges.append((pr['card_id'], 'SIG', 'MATCHES'))
        edges.append((txn['card_id'], 'SIG', 'MATCHES'))
    for sc in c['similar_prior_cases'][:3]:
        add(sc, 'case', sc, similar=True)
        edges.append((c['graph_case_id'], sc, 'SIMILAR_TO'))
    add(c['graph_case_id'], 'agentcase', c['graph_case_id'], verdict=c['verdict'])
    edges.append((c['graph_case_id'], txn['card_id'], 'ON_CARD'))
    for tid in list(affected)[:20]:
        if tid in nodes:
            edges.append((c['graph_case_id'], tid, 'INVOLVES'))
    return {'nodes': list(nodes.values()), 'edges': [{'source': s, 'target': t, 'type': k} for s, t, k in edges]}


app.mount('/static', StaticFiles(directory=STATIC), name='static')
