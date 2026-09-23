"""In-process reference implementation of the graph queries (pandas + TF-IDF).

The data model mirrors graph/schema.gsql:
Customer -OWNS-> Card -MADE-> Transaction -FROM_DEVICE-> DeviceProfile, -BILLED_IN-> BillingRegion, -NEXT-> Transaction
ClosedCase / AgentCase -INVOLVES-> Transaction, -ON_CARD-> Card, -CONNECTED_TO-> Card
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..documents import POLICY_CHUNKS
from .base import GraphStore

TX_KEEP = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card4', 'card6', 'addr1', 'addr2', 'dist1',
           'P_emaildomain', 'R_emaildomain', 'C1', 'C13', 'C14', 'D1', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
           'customer_id', 'ts', 'channel', 'risk_score']
ID_KEEP = ['TransactionID', 'id_15', 'id_23', 'id_30', 'id_31', 'id_33', 'DeviceType', 'DeviceInfo']
PORTFOLIO_FRAUD_RATE = 0.0337


def _f(x):
    """numpy/pandas scalar -> plain python (None for NaN)."""
    if x is None:
        return None
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def device_profile_id(row) -> str | None:
    """'DeviceInfo | OS | browser | screen' — the DeviceProfile vertex key. None when nothing is known."""
    parts = [row.get('DeviceInfo'), row.get('id_30'), row.get('id_31'), row.get('id_33')]
    if all(p is None or (isinstance(p, float) and np.isnan(p)) for p in parts):
        return None
    return ' | '.join('?' if (p is None or (isinstance(p, float) and np.isnan(p))) else str(p) for p in parts)


class TextIndex:
    """TF-IDF retriever over case narratives + policy chunks (stands in for TigerVector; both backends use it)."""

    def __init__(self, docs: list[dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.docs = docs
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words='english')
        self.doc_mat = self.vec.fit_transform([d['text'] for d in docs])

    def _sims(self, text):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.vec.transform([text]), self.doc_mat).ravel()

    def similar_cases(self, query_text, pattern_hint=None, customer_id=None, k=5):
        sims = self._sims(f"{pattern_hint or ''} {query_text}")
        out = []
        for i in np.argsort(-sims):
            d = self.docs[i]
            if d['kind'] == 'document':
                continue
            score = float(sims[i])
            if customer_id and d['meta'].get('customer_id') == customer_id:
                score += 0.15
            out.append({'case_id': d['id'], 'kind': d['kind'], 'score': round(score, 3), **d['meta'], 'notes': d['text'][:400]})
            if len(out) >= k * 3:
                break
        out.sort(key=lambda x: -x['score'])
        return out[:k]

    def retrieve_policy(self, query_text, k=4):
        sims = self._sims(query_text)
        out = []
        for i in np.argsort(-sims):
            d = self.docs[i]
            if d['kind'] != 'document':
                continue
            out.append({'doc_id': d['id'], 'title': d['meta']['title'], 'score': round(float(sims[i]), 3), 'text': d['text']})
            if len(out) >= k:
                break
        return out


class LocalGraphStore(GraphStore):
    def __init__(self):
        t0 = time.time()
        self.tx = self._load_transactions()
        self.tx['dev'] = None
        on = self.tx.channel == 'online'
        self.tx.loc[on, 'dev'] = self.tx[on].apply(device_profile_id, axis=1)
        self.tx.sort_values('ts', inplace=True)
        self.tx.set_index('TransactionID', drop=False, inplace=True)
        self._by_cust = {c: g for c, g in self.tx.groupby('customer_id', sort=False)}

        self.closed = pd.read_csv(config.DATA_DIR / 'closed_cases_history.csv')
        self.closed['opened_at'] = pd.to_datetime(self.closed.opened_at)
        self.case_pack = pd.read_csv(config.DATA_DIR / 'case_pack.csv')

        self.card_of = {}
        for _, r in self.closed.iterrows():
            self.card_of[r.customer_id] = r.card_id
            if isinstance(r.connected_card_ids, str):
                for c in r.connected_card_ids.split('|'):
                    self.card_of.setdefault(c[:6], c)
        for _, r in self.case_pack.iterrows():
            self.card_of[r.customer_id] = r.card_id

        self.txn_case = {}
        for _, r in self.closed.iterrows():
            for t in str(r.txn_ids).split('|'):
                if t and t != 'nan':
                    self.txn_case[int(float(t))] = (r.case_id, r.outcome, r.pattern)
        self.dev_cases = {}
        for tid, (cid, outcome, pat) in self.txn_case.items():
            d = self.tx.dev.get(tid) if tid in self.tx.index else None
            if d:
                self.dev_cases.setdefault(d, set()).add((cid, outcome, pat))

        self.memory_file = Path(config.MEMORY_FILE)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.agent_cases = []
        if self.memory_file.exists():
            for line in self.memory_file.read_text().splitlines():
                if line.strip():
                    self.agent_cases.append(json.loads(line))
        self._build_text_index()
        self.load_seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ loading
    def _load_transactions(self) -> pd.DataFrame:
        slim = config.CACHE_DIR / 'tx_slim.parquet'
        if slim.exists():
            df = pd.read_parquet(slim)
            keep = [c for c in TX_KEEP + ID_KEEP[1:] if c in df.columns]
            df = df[keep]
        else:
            tx = pd.read_csv(config.DATA_DIR / 'transactions.csv', usecols=lambda c: c in TX_KEEP)
            idn = pd.read_csv(config.DATA_DIR / 'identity.csv', usecols=lambda c: c in ID_KEEP)
            df = tx.merge(idn, on='TransactionID', how='left')
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(slim)
        df['ts'] = pd.to_datetime(df.ts)
        return df

    def _build_text_index(self):
        self.docs = []
        for _, r in self.closed.iterrows():
            self.docs.append({'id': r.case_id, 'kind': 'closed_case', 'text': f"{r.pattern} {r.outcome} {r.analyst_notes}",
                              'meta': {'customer_id': r.customer_id, 'card_id': r.card_id, 'outcome': r.outcome, 'pattern': r.pattern,
                                       'exposure_usd': _f(r.exposure_usd), 'report_filed': r.report_filed, 'opened_at': _f(r.opened_at)}})
        for c in self.agent_cases:
            self.docs.append({'id': c['graph_case_id'], 'kind': 'agent_case', 'text': f"{c.get('pattern')} {c.get('verdict')} {c.get('summary')}",
                              'meta': {k: c.get(k) for k in ('customer_id', 'card_id', 'status', 'verdict', 'pattern', 'exposure_usd', 'opened_at', 'fraud_probability')}})
        for ch in POLICY_CHUNKS:
            self.docs.append({'id': ch['id'], 'kind': 'document', 'text': ch['text'], 'meta': {'title': ch['title']}})
        self.text_index = TextIndex(self.docs)

    # ------------------------------------------------------------------ helpers
    def _cust_of_card(self, card_id: str) -> str:
        return card_id[:6]

    def card_id_for(self, customer_id: str) -> str:
        return self.card_of.get(customer_id, f"{customer_id}-K1")

    def _row(self, r) -> dict:
        d = {
            'txn_id': str(int(r.TransactionID)), 'ts': _f(r.ts), 'amount': round(float(r.TransactionAmt), 2),
            'product': r.ProductCD, 'channel': r.channel, 'region': _f(r.addr1), 'country': _f(r.addr2),
            'dist1': _f(r.dist1), 'p_email': _f(r.P_emaildomain), 'r_email': _f(r.R_emaildomain),
            'risk_score': _f(r.risk_score), 'device_profile': _f(r.dev), 'device_status': _f(r.get('id_15')),
            'proxy': _f(r.get('id_23')), 'device_type': _f(r.get('DeviceType')),
            'match_flags': {k: _f(r.get(k)) for k in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']},
            'counts': {k: _f(r.get(k)) for k in ['C1', 'C13', 'C14']},
            'days_since_prev': _f(r.get('D1')),
        }
        cc = self.txn_case.get(int(r.TransactionID))
        if cc:
            d['closed_case'] = {'case_id': cc[0], 'outcome': cc[1], 'pattern': cc[2]}
        return d

    def _cap(self, s, as_of):
        return s[s.ts <= pd.Timestamp(as_of)] if as_of else s

    # ------------------------------------------------------------------ queries
    def get_transaction(self, txn_id: int) -> dict | None:
        txn_id = int(txn_id)
        if txn_id not in self.tx.index:
            return None
        r = self.tx.loc[txn_id]
        d = self._row(r)
        d['customer_id'] = r.customer_id
        d['card_id'] = self.card_id_for(r.customer_id)
        d['card_network'] = _f(r.card4)
        d['card_type'] = _f(r.card6)
        return d

    def card_window(self, card_id, center_txn_id, hours_before=48, hours_after=48, as_of=None) -> list[dict]:
        cust = self._cust_of_card(card_id)
        s = self._cap(self._by_cust.get(cust), as_of)
        f = self.tx.loc[int(center_txn_id)]
        lo, hi = f.ts - pd.Timedelta(hours=hours_before), f.ts + pd.Timedelta(hours=hours_after)
        w = s[(s.ts >= lo) & (s.ts <= hi)]
        return [self._row(r) for _, r in w.iterrows()]

    def card_profile(self, card_id: str, before_txn_id: int) -> dict:
        cust = self._cust_of_card(card_id)
        s = self._by_cust.get(cust)
        f = self.tx.loc[int(before_txn_id)]
        hist = s[s.ts < f.ts]
        online = hist[hist.channel == 'online']
        devs = online.dev.dropna()
        dev_counts = devs.value_counts()
        prods = hist.ProductCD.value_counts()
        regions = hist.addr1.value_counts()
        amt = hist.TransactionAmt
        prof = {
            'card_id': card_id, 'customer_id': cust,
            'n_txns': int(len(hist)), 'first_seen': _f(hist.ts.min()) if len(hist) else None,
            'n_online': int(len(online)), 'n_in_person': int((hist.channel == 'in_person').sum()),
            'products': {k: int(v) for k, v in prods.items()},
            'regions_top': {str(_f(k)): int(v) for k, v in regions.head(8).items()},
            'n_regions': int(regions.shape[0]),
            'amount_median': _f(amt.median()) if len(amt) else None,
            'amount_p95': _f(amt.quantile(.95)) if len(amt) else None,
            'amount_max': _f(amt.max()) if len(amt) else None,
            'known_devices': {k: int(v) for k, v in dev_counts.head(10).items()},
            'new_device_share': _f((online.id_15 == 'New').mean()) if len(online) >= 5 else None,
            'n_known_devices': int(dev_counts.shape[0]),
            'p_email_top': {str(k): int(v) for k, v in hist.P_emaildomain.value_counts().head(4).items()},
            'proxy_history': int(hist.id_23.notna().sum()) if 'id_23' in hist else 0,
        }
        prof['flagged'] = {
            'amount_percentile': _f((amt < f.TransactionAmt).mean()) if len(amt) else None,
            'amount_z': _f((f.TransactionAmt - amt.mean()) / amt.std()) if len(amt) > 3 and amt.std() > 0 else None,
            'product_seen_before': int(prods.get(f.ProductCD, 0)),
            'region_seen_before': int(regions.get(f.addr1, 0)) if not pd.isna(f.addr1) else None,
            'device_seen_before': int(dev_counts.get(f.dev, 0)) if f.dev else None,
            'channel_share_online': _f(len(online) / len(hist)) if len(hist) else None,
        }
        return prof

    def device_neighbors(self, device_id: str, start: str, end: str) -> dict:
        if not device_id:
            return {'device_profile': None, 'cards': [], 'n_cards': 0, 'note': 'no device profile on this transaction'}
        m = self.tx[(self.tx.dev == device_id) & (self.tx.ts >= pd.Timestamp(start)) & (self.tx.ts <= pd.Timestamp(end))]
        allm = self.tx[self.tx.dev == device_id]
        cards = []
        for cust, g in m.groupby('customer_id'):
            cards.append({'card_id': self.card_id_for(cust), 'customer_id': cust, 'n_txns': int(len(g)),
                          'total_usd': round(float(g.TransactionAmt.sum()), 2),
                          'new_device_share': _f((g.id_15 == 'New').mean()),
                          'proxy': sorted({str(p) for p in g.id_23.dropna().unique()}),
                          'first': _f(g.ts.min()), 'last': _f(g.ts.max()),
                          'txn_ids': [str(int(t)) for t in g.TransactionID.head(20)]})
        cases = sorted(self.dev_cases.get(device_id, set()))
        generic = device_id.count('?') >= 2
        return {
            'device_profile': device_id, 'window': [start, end], 'n_cards': len(cards), 'cards': cards,
            'n_cards_all_time': int(allm.customer_id.nunique()), 'n_txns_all_time': int(len(allm)),
            'closed_cases_on_device': [{'case_id': c, 'outcome': o, 'pattern': p} for c, o, p in cases],
            'proxy_share_in_window': _f(m.id_23.notna().mean()) if len(m) else None,
            'new_share_in_window': _f((m.id_15 == 'New').mean()) if len(m) else None,
            'generic_fingerprint': generic,
        }

    def region_history(self, card_id, region, before_txn_id, as_of=None) -> dict:
        cust = self._cust_of_card(card_id)
        s = self._cap(self._by_cust.get(cust), as_of)
        f = self.tx.loc[int(before_txn_id)]
        hist = s[s.ts < f.ts]
        home = hist.addr1.value_counts()
        home_region = _f(home.index[0]) if len(home) else None
        if region is None or (isinstance(region, float) and np.isnan(region)):
            return {'card_id': card_id, 'region': None, 'n_before': 0, 'home_region': home_region, 'first_seen_in_region': None,
                    'months_active_in_region': 0, 'in_region_next_7d': 0, 'home_activity_next_7d': 0, 'in_person_share_in_region': None}
        inr = hist[hist.addr1 == region]
        nxt = s[(s.ts >= f.ts) & (s.ts <= f.ts + pd.Timedelta(days=7)) & (s.TransactionID != f.TransactionID)]
        return {'card_id': card_id, 'region': _f(region), 'n_before': int(len(inr)), 'home_region': home_region,
                'first_seen_in_region': _f(inr.ts.min()) if len(inr) else None,
                'months_active_in_region': int(inr.ts.dt.to_period('M').nunique()) if len(inr) else 0,
                'in_region_next_7d': int((nxt.addr1 == region).sum()),
                'home_activity_next_7d': int((nxt.addr1 == home_region).sum()) if home_region is not None and home_region != region else 0,
                'in_person_share_in_region': _f((inr.channel == 'in_person').mean()) if len(inr) else None}

    def recurring_match(self, card_id, txn_id, amount_tol=0.6, as_of=None) -> dict:
        cust = self._cust_of_card(card_id)
        s = self._cap(self._by_cust.get(cust), as_of)
        f = self.tx.loc[int(txn_id)]
        m = s[(s.ProductCD == f.ProductCD) & ((s.TransactionAmt - f.TransactionAmt).abs() <= amount_tol)]
        months = sorted(m.ts.dt.strftime('%Y-%m').unique().tolist())
        near = m[(m.ts - f.ts).abs() <= pd.Timedelta(hours=1)]
        n, n_months = int(len(m)), len(months)
        share = n / max(len(s), 1)
        is_rec = n >= 3 and (share >= 0.02 or n_months >= 4) and len(near) <= 1
        return {'amount': round(float(f.TransactionAmt), 2), 'product': f.ProductCD, 'n_matches': n, 'months_with_match': months,
                'n_months': n_months, 'n_same_region': int((m.addr1 == f.addr1).sum()) if not pd.isna(f.addr1) else 0,
                'share_of_card_txns': round(share, 4), 'n_same_amount_within_1h': int(len(near)),
                'same_amount_burst_ids': [str(int(t)) for t in near.TransactionID] if len(near) >= 3 else [],
                'is_recurring': bool(is_rec), 'sample_txn_ids': [str(int(t)) for t in m.TransactionID.tail(8)]}

    def small_auth_run(self, card_id, txn_id, hours=1.0, max_amt=5.0, as_of=None) -> dict:
        cust = self._cust_of_card(card_id)
        s = self._cap(self._by_cust.get(cust), as_of)
        f = self.tx.loc[int(txn_id)]
        w = s[(s.channel == 'online') & ((s.ts - f.ts).abs() <= pd.Timedelta(hours=hours))]
        small = w[w.TransactionAmt <= max_amt]
        larger = w[(w.TransactionAmt > 20) & (w.ts >= small.ts.min())] if len(small) else w.iloc[0:0]
        return {'n_small_auths': int(len(small)), 'small_txn_ids': [str(int(t)) for t in small.TransactionID],
                'larger_after': [self._row(r) for _, r in larger.iterrows()],
                'testing_sequence': bool(len(small) >= 3 and len(larger) >= 1),
                'cleared_over_100': bool(len(larger) and larger.TransactionAmt.max() > 100)}

    def customer_fraud_rate(self, customer_id, as_of=None) -> dict:
        s = self._cap(self._by_cust.get(customer_id), as_of)
        fraud = s.TransactionID.map(lambda t: self.txn_case.get(int(t), (None, None))[1] == 'confirmed_fraud')
        out = {'customer_id': customer_id, 'n': int(len(s)), 'fraud_rate': _f(fraud.mean()) if len(s) else None, 'global_rate': PORTFOLIO_FRAUD_RATE}
        for ch in ('in_person', 'online'):
            g = fraud[s.channel == ch]
            out[f'fraud_rate_{ch}'] = _f(g.mean()) if len(g) else None
            out[f'n_{ch}'] = int(len(g))
        return out

    def card_txns_on_device(self, card_id, device_id, as_of=None) -> list[dict]:
        cust = self._cust_of_card(card_id)
        s = self._cap(self._by_cust.get(cust), as_of)
        return [self._row(r) for _, r in s[s.dev == device_id].iterrows()]

    def pattern_peers(self, kind, start, end, exclude_customer=None) -> dict:
        m = self.tx[(self.tx.ts >= pd.Timestamp(start)) & (self.tx.ts <= pd.Timestamp(end)) & (self.tx.channel == 'online')]
        peers = []
        if kind == 'sub500_burst':
            m = m[(m.TransactionAmt >= 400) & (m.TransactionAmt < 500)].sort_values(['customer_id', 'ts'])
            for cust, g in m.groupby('customer_id'):
                t = g.ts.values
                for i in range(len(t)):
                    sel = (t >= t[i]) & (t <= t[i] + np.timedelta64(60, 'm'))
                    if sel.sum() >= 3:
                        gg = g[sel]
                        peers.append({'card_id': self.card_id_for(cust), 'customer_id': cust, 'n': int(sel.sum()),
                                      'total_usd': round(float(gg.TransactionAmt.sum()), 2), 'first': _f(gg.ts.min()),
                                      'txn_ids': [str(int(x)) for x in gg.TransactionID]})
                        break
        elif kind == 'anon_proxy_new_device':
            m = m[(m.id_15 == 'New') & (m.id_23 == 'IP_PROXY:ANONYMOUS')]
            for cust, g in m.groupby('customer_id'):
                peers.append({'card_id': self.card_id_for(cust), 'customer_id': cust, 'n': int(len(g)),
                              'total_usd': round(float(g.TransactionAmt.sum()), 2), 'first': _f(g.ts.min()),
                              'devices': sorted(g.dev.dropna().unique().tolist())[:3]})
        if exclude_customer:
            peers = [p for p in peers if p['customer_id'] != exclude_customer]
        return {'kind': kind, 'window': [start, end], 'n_peers': len(peers), 'peers': peers[:40]}

    # ------------------------------------------------------------------ memory / GraphRAG
    def prior_cases_for_customer(self, customer_id: str) -> list[dict]:
        out = []
        for _, r in self.closed[self.closed.customer_id == customer_id].sort_values('opened_at').iterrows():
            out.append({'case_id': r.case_id, 'opened_at': _f(r.opened_at), 'outcome': r.outcome, 'pattern': r.pattern,
                        'exposure_usd': _f(r.exposure_usd), 'report_filed': r.report_filed, 'notes': str(r.analyst_notes)[:300]})
        for c in self.agent_cases:
            if c.get('customer_id') == customer_id:
                out.append({'case_id': c['graph_case_id'], 'opened_at': c.get('opened_at'), 'outcome': c.get('status'), 'pattern': c.get('pattern'),
                            'exposure_usd': c.get('exposure_usd'), 'report_filed': c.get('sar_filed'), 'notes': str(c.get('summary'))[:300]})
        return out

    def similar_cases(self, query_text, pattern_hint=None, customer_id=None, k=5) -> list[dict]:
        return self.text_index.similar_cases(query_text, pattern_hint, customer_id, k)

    def retrieve_policy(self, query_text, k=4) -> list[dict]:
        return self.text_index.retrieve_policy(query_text, k)

    def write_case(self, case: dict) -> str:
        gid = case.get('graph_case_id') or f"CASE-2016-{1000 + len(self.agent_cases) + 1:04d}"
        case['graph_case_id'] = gid
        self.agent_cases = [c for c in self.agent_cases if c.get('graph_case_id') != gid] + [case]
        with self.memory_file.open('w') as fh:
            for c in self.agent_cases:
                fh.write(json.dumps(c, default=str) + '\n')
        self._build_text_index()
        return gid

    def alerts_in_period(self, start, end, min_score=0.85, limit=50) -> list[dict]:
        m = self.tx[(self.tx.ts >= pd.Timestamp(start)) & (self.tx.ts <= pd.Timestamp(end)) & (self.tx.risk_score >= min_score)]
        m = m.sort_values('risk_score', ascending=False).head(limit)
        return [self.get_transaction(int(t)) for t in m.TransactionID]

    def describe(self):
        return {'backend': 'LocalGraphStore', 'transactions': int(len(self.tx)), 'customers': len(self._by_cust),
                'closed_cases': int(len(self.closed)), 'agent_cases': len(self.agent_cases), 'load_seconds': self.load_seconds}
