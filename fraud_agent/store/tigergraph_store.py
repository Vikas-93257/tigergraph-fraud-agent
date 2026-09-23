"""TigerGraph backend: every tool call is an installed GSQL query (graph/queries/investigation.gsql).

Responses are reshaped into the same dicts that ``LocalGraphStore`` returns, so the agent, MCP server and UI are
backend-agnostic. Verified against TigerGraph 4.2.5 on Savanna.

Text retrieval (similar_cases / retrieve_policy) uses TigerVector when the embeddings have been populated
(`python graph/load.py embed`), otherwise the same TF-IDF retriever as the local store fed from vertices pulled
out of the graph.

Environment: TG_HOST, TG_GRAPH, TG_SECRET (or TG_USERNAME/TG_PASSWORD).
"""
from __future__ import annotations

import json
import statistics
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from .. import config
from .base import GraphStore

try:
    import pyTigerGraph as tg
except ImportError:  # pragma: no cover
    tg = None

FAR_FUTURE = '2100-01-01 00:00:00'
PORTFOLIO_FRAUD_RATE = 0.0337
warnings.filterwarnings('ignore', message='Deprecated parameter format')


def _dt(s) -> str | None:
    if not s:
        return None
    return str(s).replace('T', ' ')[:19]


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace(' ', 'T')[:19])


def _num(x):
    return None if x in (None, '') else x


class TigerGraphStore(GraphStore):
    def __init__(self):
        if tg is None:
            raise RuntimeError("pip install pyTigerGraph")
        self.conn = tg.TigerGraphConnection(host=config.TG_HOST, graphname=config.TG_GRAPH,
                                            username=config.TG_USERNAME or 'tigergraph', password=config.TG_PASSWORD,
                                            gsqlSecret=config.TG_SECRET or None)
        if config.TG_SECRET:
            self.conn.getToken(config.TG_SECRET)
        else:
            self.conn.getToken(self.conn.createSecret())
        self._card_cache: dict[str, list[dict]] = {}
        self._text_index = None
        self.memory_file = Path(config.MEMORY_FILE)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.has_vector = self._probe_vector()

    # ------------------------------------------------------------------ helpers
    VERTEX_PARAMS = {'t', 'center', 'card', 'dev', 'before', 'cust'}

    def _q(self, name: str, **params):
        params = {k: ((v,) if k in self.VERTEX_PARAMS else v) for k, v in params.items()}
        return self.conn.runInstalledQuery(name, params=params, timeout=120000)

    def _probe_vector(self) -> bool:
        """Vector retrieval is only useful once embeddings are populated; probe with a policy chunk."""
        try:
            from .embeddings import embed
            r = self._q('retrieve_policy', query_vec=embed('card testing small authorizations'), k=1)
            d = list(r[1]['distance'].values())
            return bool(d) and d[0] < 0.999
        except Exception:
            return False

    @staticmethod
    def _attrs(v: dict) -> dict:
        d = dict(v.get('attributes', {}))
        d['v_id'] = v.get('v_id')
        return d

    def _txn(self, v: dict) -> dict:
        a = self._attrs(v)
        region = a.get('region') or None
        return {
            'txn_id': str(a['v_id']), 'ts': _dt(a.get('ts')), 'amount': round(float(a.get('amount') or 0), 2),
            'product': a.get('product'), 'channel': a.get('channel'),
            'region': float(region) if region else None, 'country': float(a['country']) if a.get('country') else None,
            'dist1': a.get('dist1'), 'p_email': a.get('p_email') or None, 'r_email': a.get('r_email') or None,
            'risk_score': a.get('risk_score'), 'device_profile': a.get('@dev') or None,
            'device_status': a.get('device_status') or None, 'proxy': a.get('proxy_type') or None,
            'device_type': a.get('device_type') or None,
            'match_flags': {'M4': a.get('m4') or None, 'M5': a.get('m5') or None, 'M6': a.get('m6') or None},
            'counts': {'C1': a.get('c1'), 'C13': a.get('c13'), 'C14': a.get('c14')}, 'days_since_prev': a.get('d1'),
            '_fraud': bool(a.get('@fraud')),
        }

    def _card_txns(self, card_id: str, as_of: str | None = None) -> list[dict]:
        key = (card_id, as_of or FAR_FUTURE)
        if key not in self._card_cache:
            r = self._q('card_txns', card=card_id, as_of=as_of or FAR_FUTURE)
            self._card_cache[key] = [self._txn(v) for v in r[0]['T']]
        return self._card_cache[key]

    # ------------------------------------------------------------------ transactions
    def get_transaction(self, txn_id: int) -> dict | None:
        r = self._q('get_transaction', t=str(txn_id))
        if not r or not r[0].get('txn'):
            return None
        d = self._txn(r[0]['txn'][0])
        d.pop('_fraud', None)
        card = r[1].get('card') or []
        if card:
            c = self._attrs(card[0])
            d['customer_id'] = c.get('customer_id'); d['card_id'] = c['v_id']
            d['card_network'] = c.get('network') or None; d['card_type'] = c.get('card_type') or None
        dev = r[2].get('device') or []
        if dev:
            d['device_profile'] = dev[0]['v_id']
        cc = r[3].get('closed_case') or []
        if cc:
            a = self._attrs(cc[0])
            d['closed_case'] = {'case_id': a['v_id'], 'outcome': a.get('outcome'), 'pattern': a.get('pattern')}
        return d

    def card_window(self, card_id, center_txn_id, hours_before=48, hours_after=48, as_of=None) -> list[dict]:
        r = self._q('card_window', center=str(center_txn_id), hours_before=int(hours_before), hours_after=int(hours_after),
                    as_of=_dt(as_of) or FAR_FUTURE)
        rows = [self._txn(v) for v in r[0]['W']]
        for x in rows:
            x.pop('_fraud', None)
        return rows

    def card_profile(self, card_id: str, before_txn_id: int) -> dict:
        f = self.get_transaction(before_txn_id)
        hist = [t for t in self._card_txns(card_id) if t['ts'] < f['ts']]
        online = [t for t in hist if t['channel'] == 'online']
        amts = [t['amount'] for t in hist]
        prods = Counter(t['product'] for t in hist)
        regions = Counter(t['region'] for t in hist if t['region'] is not None)
        devs = Counter(t['device_profile'] for t in online if t['device_profile'])
        pemail = Counter(t['p_email'] for t in hist if t['p_email'])
        srt = sorted(amts)
        p95 = _quantile(srt, 0.95) if srt else None
        prof = {
            'card_id': card_id, 'customer_id': f.get('customer_id') or card_id[:6], 'n_txns': len(hist),
            'first_seen': hist[0]['ts'] if hist else None, 'n_online': len(online), 'n_in_person': len(hist) - len(online),
            'products': dict(prods), 'regions_top': {str(k): v for k, v in regions.most_common(8)}, 'n_regions': len(regions),
            'amount_median': statistics.median(amts) if amts else None, 'amount_p95': p95, 'amount_max': max(amts) if amts else None,
            'known_devices': dict(devs.most_common(10)),
            'new_device_share': (sum(1 for t in online if t['device_status'] == 'New') / len(online)) if len(online) >= 5 else None,
            'n_known_devices': len(devs), 'p_email_top': dict(pemail.most_common(4)),
            'proxy_history': sum(1 for t in hist if t['proxy']),
        }
        mean = statistics.mean(amts) if amts else 0
        sd = statistics.stdev(amts) if len(amts) > 3 else 0
        prof['flagged'] = {
            'amount_percentile': (sum(1 for a in amts if a < f['amount']) / len(amts)) if amts else None,
            'amount_z': ((f['amount'] - mean) / sd) if sd else None,
            'product_seen_before': prods.get(f['product'], 0),
            'region_seen_before': regions.get(f['region'], 0) if f['region'] is not None else None,
            'device_seen_before': devs.get(f['device_profile'], 0) if f.get('device_profile') else None,
            'channel_share_online': (len(online) / len(hist)) if hist else None,
        }
        return prof

    # ------------------------------------------------------------------ hubs
    def device_neighbors(self, device_id: str, start: str, end: str) -> dict:
        if not device_id:
            return {'device_profile': None, 'cards': [], 'n_cards': 0, 'note': 'no device profile on this transaction'}
        r = self._q('device_neighbors', dev=device_id, start_ts=_dt(start), end_ts=_dt(end))
        r0 = r[0]
        cards = []
        for cid, n in sorted(r0['n_txns'].items()):
            cards.append({'card_id': cid, 'customer_id': r0['customer_of'].get(cid, cid[:6]), 'n_txns': n,
                          'total_usd': round(r0['total_usd'].get(cid, 0), 2),
                          'new_device_share': (r0['n_new'].get(cid, 0) / n) if n else None,
                          'proxy': sorted(r0['proxies'].get(cid, [])), 'first': _dt(r0['first_ts'].get(cid)), 'last': _dt(r0['last_ts'].get(cid)),
                          'txn_ids': sorted(r0['txn_ids'].get(cid, []))[:20]})
        cases = sorted([{'case_id': c['v_id'], 'outcome': c['attributes'].get('outcome'), 'pattern': c['attributes'].get('pattern')}
                        for c in r[1]['closed_cases_on_device']], key=lambda c: c['case_id'])
        n_tx = sum(r0['n_txns'].values()); n_new = sum(r0['n_new'].values()); n_px = sum(r0['n_proxy'].values())
        return {'device_profile': device_id, 'window': [start, end], 'n_cards': len(cards), 'cards': cards,
                'n_cards_all_time': r0['n_cards_all_time'], 'n_txns_all_time': r0['n_txns_all_time'],
                'closed_cases_on_device': cases,
                'proxy_share_in_window': (n_px / n_tx) if n_tx else None, 'new_share_in_window': (n_new / n_tx) if n_tx else None,
                'generic_fingerprint': device_id.count('?') >= 2}

    def region_history(self, card_id, region, before_txn_id, as_of=None) -> dict:
        if region is None:
            return {'card_id': card_id, 'region': None, 'n_before': 0, 'home_region': None, 'first_seen_in_region': None,
                    'months_active_in_region': 0, 'in_region_next_7d': 0, 'home_activity_next_7d': 0, 'in_person_share_in_region': None}
        reg = str(int(float(region)))
        r = self._q('region_history', card=card_id, region=reg, before=str(before_txn_id), as_of=_dt(as_of) or FAR_FUTURE)[0]
        n = r['n_before']
        return {'card_id': card_id, 'region': float(reg), 'n_before': n,
                'home_region': float(r['home_region']) if r.get('home_region') else None,
                'first_seen_in_region': _dt(r.get('first_seen_in_region')) if n else None,
                'months_active_in_region': r['months_active_in_region'], 'in_region_next_7d': r['in_region_next_7d'],
                'home_activity_next_7d': r['home_activity_next_7d'],
                'in_person_share_in_region': (r['in_person_in_region'] / n) if n else None}

    def recurring_match(self, card_id, txn_id, amount_tol=0.6, as_of=None) -> dict:
        s = self._card_txns(card_id, as_of)
        f = next((t for t in s if t['txn_id'] == str(txn_id)), None) or self.get_transaction(txn_id)
        m = [t for t in s if t['product'] == f['product'] and abs(t['amount'] - f['amount']) <= amount_tol]
        months = sorted({t['ts'][:7] for t in m})
        near = [t for t in m if abs((_ts(t['ts']) - _ts(f['ts'])).total_seconds()) <= 3600]
        n, n_months = len(m), len(months)
        share = n / max(len(s), 1)
        is_rec = n >= 3 and (share >= 0.02 or n_months >= 4) and len(near) <= 1
        return {'amount': f['amount'], 'product': f['product'], 'n_matches': n, 'months_with_match': months, 'n_months': n_months,
                'n_same_region': sum(1 for t in m if t['region'] == f['region']) if f['region'] is not None else 0,
                'share_of_card_txns': round(share, 4), 'n_same_amount_within_1h': len(near),
                'same_amount_burst_ids': [t['txn_id'] for t in near] if len(near) >= 3 else [],
                'is_recurring': bool(is_rec), 'sample_txn_ids': [t['txn_id'] for t in m[-8:]]}

    def small_auth_run(self, card_id, txn_id, hours=1.0, max_amt=5.0, as_of=None) -> dict:
        s = self._card_txns(card_id, as_of)
        f = next((t for t in s if t['txn_id'] == str(txn_id)), None) or self.get_transaction(txn_id)
        w = [t for t in s if t['channel'] == 'online' and abs((_ts(t['ts']) - _ts(f['ts'])).total_seconds()) <= hours * 3600]
        small = [t for t in w if t['amount'] <= max_amt]
        larger = [t for t in w if t['amount'] > 20 and small and t['ts'] >= min(x['ts'] for x in small)]
        return {'n_small_auths': len(small), 'small_txn_ids': [t['txn_id'] for t in small],
                'larger_after': [{k: v for k, v in t.items() if k != '_fraud'} for t in larger],
                'testing_sequence': bool(len(small) >= 3 and larger),
                'cleared_over_100': bool(larger and max(t['amount'] for t in larger) > 100)}

    def customer_fraud_rate(self, customer_id, as_of=None) -> dict:
        r = self._q('customer_fraud_rate', cust=customer_id, cutoff=_dt(as_of) or FAR_FUTURE)[0]
        return {'customer_id': customer_id, 'n': r['n'], 'fraud_rate': (r['n_fraud'] / r['n']) if r['n'] else None,
                'global_rate': PORTFOLIO_FRAUD_RATE,
                'fraud_rate_in_person': (r['n_in_person_fraud'] / r['n_in_person']) if r['n_in_person'] else None, 'n_in_person': r['n_in_person'],
                'fraud_rate_online': (r['n_online_fraud'] / r['n_online']) if r['n_online'] else None, 'n_online': r['n_online']}

    def card_txns_on_device(self, card_id, device_id, as_of=None) -> list[dict]:
        return [{k: v for k, v in t.items() if k != '_fraud'} for t in self._card_txns(card_id, as_of) if t['device_profile'] == device_id]

    def pattern_peers(self, kind, start, end, exclude_customer=None) -> dict:
        peers = []
        if kind == 'sub500_burst':
            r = self._q('pattern_peers_sub500', start_ts=_dt(start), end_ts=_dt(end))[0]
            for cid, tss in r['timestamps'].items():
                rows = sorted(zip([_ts(x) for x in tss], r['txn_ids'][cid], r['amounts'][cid]))
                for i in range(len(rows)):
                    grp = [x for x in rows if rows[i][0] <= x[0] <= rows[i][0] + timedelta(minutes=60)]
                    if len(grp) >= 3:
                        peers.append({'card_id': cid, 'customer_id': r['customer_of'].get(cid, cid[:6]), 'n': len(grp),
                                      'total_usd': round(sum(x[2] for x in grp), 2), 'first': grp[0][0].strftime('%Y-%m-%d %H:%M:%S'),
                                      'txn_ids': [x[1] for x in grp]})
                        break
        elif kind == 'anon_proxy_new_device':
            r = self._q('pattern_peers_anon_proxy', start_ts=_dt(start), end_ts=_dt(end))[0]
            for cid, n in r['n_by_card'].items():
                peers.append({'card_id': cid, 'customer_id': r['customer_of'].get(cid, cid[:6]), 'n': n,
                              'total_usd': round(r['usd_by_card'].get(cid, 0), 2), 'first': _dt(r['first_by_card'].get(cid)),
                              'devices': sorted(r['devices_by_card'].get(cid, []))[:3]})
        peers.sort(key=lambda p: p['card_id'])
        if exclude_customer:
            peers = [p for p in peers if p['customer_id'] != exclude_customer]
        return {'kind': kind, 'window': [start, end], 'n_peers': len(peers), 'peers': peers[:40]}

    # ------------------------------------------------------------------ memory / GraphRAG
    def prior_cases_for_customer(self, customer_id: str) -> list[dict]:
        r = self._q('prior_cases_for_customer', cust=customer_id)
        out = []
        for v in r[0]['CC']:
            a = self._attrs(v)
            out.append({'case_id': a['v_id'], 'opened_at': _dt(a.get('opened_at')), 'outcome': a.get('outcome'), 'pattern': a.get('pattern'),
                        'exposure_usd': a.get('exposure_usd'), 'report_filed': 'Yes' if a.get('report_filed') else 'No',
                        'notes': (a.get('analyst_notes') or '')[:300]})
        for v in r[1]['AC']:
            a = self._attrs(v)
            out.append({'case_id': a['v_id'], 'opened_at': _dt(a.get('opened_at')), 'outcome': a.get('status'), 'pattern': a.get('pattern'),
                        'exposure_usd': a.get('exposure_usd'), 'report_filed': a.get('sar_filed'), 'notes': (a.get('summary') or '')[:300]})
        return sorted(out, key=lambda c: c['opened_at'] or '')

    def _text_retriever(self):
        """TF-IDF over ClosedCase/AgentCase/PolicyChunk vertices pulled from the graph (fallback when vectors are not populated)."""
        if self._text_index is None:
            from .local_store import TextIndex
            docs = []
            for v in self.conn.getVertices('ClosedCase', limit=100000):
                a = v['attributes']
                docs.append({'id': v['v_id'], 'kind': 'closed_case', 'text': f"{a.get('pattern')} {a.get('outcome')} {a.get('analyst_notes')}",
                             'meta': {'customer_id': a.get('customer_id'), 'card_id': a.get('card_id'), 'outcome': a.get('outcome'),
                                      'pattern': a.get('pattern'), 'exposure_usd': a.get('exposure_usd'),
                                      'report_filed': 'Yes' if a.get('report_filed') else 'No', 'opened_at': _dt(a.get('opened_at'))}})
            for v in self.conn.getVertices('AgentCase', limit=100000):
                a = v['attributes']
                docs.append({'id': v['v_id'], 'kind': 'agent_case', 'text': f"{a.get('pattern')} {a.get('verdict')} {a.get('summary')}",
                             'meta': {'customer_id': a.get('customer_id'), 'card_id': a.get('card_id'), 'status': a.get('status'),
                                      'verdict': a.get('verdict'), 'pattern': a.get('pattern'), 'exposure_usd': a.get('exposure_usd'),
                                      'opened_at': _dt(a.get('opened_at')), 'fraud_probability': a.get('fraud_probability')}})
            for v in self.conn.getVertices('PolicyChunk', limit=1000):
                a = v['attributes']
                docs.append({'id': v['v_id'], 'kind': 'document', 'text': a.get('text'), 'meta': {'title': a.get('title')}})
            self._text_index = TextIndex(docs)
        return self._text_index

    def similar_cases(self, query_text, pattern_hint=None, customer_id=None, k=5) -> list[dict]:
        if self.has_vector:
            from .embeddings import embed
            r = self._q('similar_cases', query_vec=embed(f"{pattern_hint or ''} {query_text}"), k=k)
            dist = {**r[2]['closed_distance'], **r[2]['agent_distance']}
            out = []
            for key, kind in (('closed_cases', 'closed_case'), ('agent_cases', 'agent_case')):
                for v in r[0 if kind == 'closed_case' else 1][key]:
                    a = self._attrs(v)
                    score = 1 - dist.get(a['v_id'], 1.0)
                    if customer_id and a.get('customer_id') == customer_id:
                        score += 0.15
                    out.append({'case_id': a['v_id'], 'kind': kind, 'score': round(score, 3), 'customer_id': a.get('customer_id'),
                                'card_id': a.get('card_id'), 'outcome': a.get('outcome') or a.get('status'), 'status': a.get('status'),
                                'verdict': a.get('verdict'), 'pattern': a.get('pattern'), 'exposure_usd': a.get('exposure_usd'),
                                'opened_at': _dt(a.get('opened_at')), 'notes': (a.get('analyst_notes') or a.get('summary') or '')[:400]})
            out.sort(key=lambda x: -x['score'])
            return out[:k]
        return self._text_retriever().similar_cases(query_text, pattern_hint, customer_id, k)

    def retrieve_policy(self, query_text, k=4) -> list[dict]:
        if self.has_vector:
            from .embeddings import embed
            r = self._q('retrieve_policy', query_vec=embed(query_text), k=k)
            dist = r[1]['distance']
            out = [{'doc_id': v['v_id'], 'title': v['attributes'].get('title'), 'score': round(1 - dist.get(v['v_id'], 1.0), 3),
                    'text': v['attributes'].get('text')} for v in r[0]['R']]
            return sorted(out, key=lambda x: -x['score'])
        return self._text_retriever().retrieve_policy(query_text, k)

    def write_case(self, case: dict) -> str:
        gid = case.get('graph_case_id') or f"CASE-2016-{case['case_id'].split('-')[-1]}"
        case['graph_case_id'] = gid
        self._q('write_case', case_id=gid, customer_id=case['customer_id'], card_id=case['card_id'],
                opened_at=_dt(case['opened_at']), status=case['status'], verdict=case['verdict'],
                fraud_probability=float(case['fraud_probability']), pattern=case['pattern'],
                pattern_description=case.get('pattern_description', '') or '', exposure_usd=float(case['exposure_usd']),
                sar_filed=bool(case.get('sar_filed')), final_actions=','.join(case.get('final_actions', [])),
                summary=case.get('summary', '') or '', txn_ids=[str(x) for x in case.get('affected_txn_ids', [])],
                connected_cards=list(case.get('connected_card_ids', [])), device_ids=list(case.get('connected_device_profiles', [])),
                similar_closed=[c for c in case.get('similar_prior_cases', []) if str(c).startswith('CC-')])
        if self.has_vector:
            from .embeddings import embed
            self._q('set_embedding', vtype='AgentCase', id=gid, vec=embed(f"{case.get('pattern')} {case.get('verdict')} {case.get('summary', '')}"))
        with self.memory_file.open('a') as fh:  # local mirror for the UI
            fh.write(json.dumps(case, default=str) + '\n')
        self._text_index = None
        return gid

    def alerts_in_period(self, start, end, min_score=0.85, limit=50) -> list[dict]:
        r = self._q('alerts_in_period', start_ts=_dt(start), end_ts=_dt(end), min_score=min_score, k=limit)[0]['R']
        return [self.get_transaction(v['v_id']) for v in r]

    def describe(self):
        try:
            counts = self.conn.getVertexCount('*')
        except Exception:
            counts = {}
        return {'backend': 'TigerGraphStore', 'host': config.TG_HOST, 'graph': config.TG_GRAPH, 'vector_index': self.has_vector,
                'transactions': counts.get('Transaction'), 'closed_cases': counts.get('ClosedCase'), 'agent_cases': counts.get('AgentCase')}


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile matching pandas' default."""
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)
