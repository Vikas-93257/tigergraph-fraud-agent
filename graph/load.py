"""Export the HHGOA CSVs into TigerGraph-friendly files and (optionally) load them.

    python graph/load.py export            # writes graph/export/*.csv (idempotent, ~2 min)
    python graph/load.py schema            # runs schema.gsql + load.gsql + queries via pyTigerGraph gsql()
    python graph/load.py load              # runs the loading job with the exported files
    python graph/load.py all

Requires TG_HOST / TG_USERNAME / TG_PASSWORD (or TG_SECRET) in .env for schema/load.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraud_agent import config  # noqa: E402
from fraud_agent.documents import POLICY_CHUNKS  # noqa: E402
from fraud_agent.store.local_store import LocalGraphStore, device_profile_id  # noqa: E402

OUT = ROOT / 'graph' / 'export'


def export():
    """Memory-light export (runs on a 2 GB box): streams from the slim parquet in column groups."""
    import gc
    import pyarrow.parquet as pq
    OUT.mkdir(exist_ok=True)
    st = LocalGraphStore()
    card_for = st.card_id_for
    closed = st.closed.copy()
    del st; gc.collect()

    pf = pq.ParquetFile(config.CACHE_DIR / 'tx_slim.parquet')
    tx_cols = ['TransactionID', 'ts', 'TransactionAmt', 'ProductCD', 'channel', 'addr1', 'addr2', 'dist1', 'P_emaildomain',
               'R_emaildomain', 'risk_score', 'id_15', 'id_23', 'DeviceType', 'M4', 'M5', 'M6', 'C1', 'C13', 'C14', 'D1',
               'customer_id', 'card4', 'card6', 'DeviceInfo', 'id_30', 'id_31', 'id_33']
    tx_cols = [c for c in tx_cols if c in pf.schema.names]
    first = True
    seen_dev, custs, cards = set(), set(), {}
    keyed = []  # (card_id, ts, txn_id) for NEXT edges
    for batch in pf.iter_batches(batch_size=100_000, columns=tx_cols):
        tx = batch.to_pandas()
        tx['ts'] = pd.to_datetime(tx.ts)
        tx['dev'] = None
        on = tx.channel == 'online'
        if on.any():
            tx.loc[on, 'dev'] = tx[on].apply(device_profile_id, axis=1)
        tx['card_id'] = tx.customer_id.map(card_for)
        custs.update(tx.customer_id.unique())
        for r in tx.drop_duplicates('card_id').itertuples():
            cards.setdefault(r.card_id, (r.customer_id, r.card4, r.card6))
        d = tx[tx.dev.notna() & ~tx.dev.isin(seen_dev)].drop_duplicates('dev')
        seen_dev.update(d.dev)
        pd.DataFrame({'device_id': d.dev, 'device_info': d.DeviceInfo.fillna('?'), 'os': d.id_30.fillna('?'),
                      'browser': d.id_31.fillna('?'), 'screen': d.id_33.fillna('?')}).to_csv(
            OUT / 'devices.csv', index=False, quoting=csv.QUOTE_ALL, mode='w' if first else 'a', header=first)
        t = pd.DataFrame({
            'txn_id': tx.TransactionID.astype(int), 'card_id': tx.card_id, 'ts': tx.ts.dt.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': tx.TransactionAmt.round(2), 'product': tx.ProductCD, 'channel': tx.channel,
            'region': tx.addr1.map(lambda x: '' if pd.isna(x) else str(int(x))), 'country': tx.addr2.map(lambda x: '' if pd.isna(x) else str(int(x))),
            'dist1': tx.dist1.fillna(0), 'p_email': tx.P_emaildomain.fillna(''), 'r_email': tx.R_emaildomain.fillna(''),
            'risk_score': tx.risk_score.fillna(0), 'device_status': tx.id_15.fillna(''), 'proxy': tx.id_23.fillna(''),
            'device_type': tx.DeviceType.fillna(''), 'm4': tx.M4.fillna(''), 'm5': tx.M5.fillna(''), 'm6': tx.M6.fillna(''),
            'c1': tx.C1.fillna(0), 'c13': tx.C13.fillna(0), 'c14': tx.C14.fillna(0), 'd1': tx.D1.fillna(0), 'device_id': tx.dev.fillna(''),
        })
        t.to_csv(OUT / 'transactions.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, mode='w' if first else 'a', header=first)
        keyed.append(tx[['card_id', 'ts', 'TransactionID']])
        first = False
        del tx, t; gc.collect()

    pd.DataFrame({'customer_id': sorted(custs)}).to_csv(OUT / 'customers.csv', index=False)
    pd.DataFrame([(k, *v) for k, v in cards.items()], columns=['card_id', 'customer_id', 'network', 'card_type']).to_csv(OUT / 'cards.csv', index=False)

    k = pd.concat(keyed).sort_values(['card_id', 'ts'])
    same = k.card_id.values[1:] == k.card_id.values[:-1]
    nxt = pd.DataFrame({'from_txn': k.TransactionID.astype(int).values[:-1][same], 'to_txn': k.TransactionID.astype(int).values[1:][same],
                        'gap_seconds': ((k.ts.values[1:] - k.ts.values[:-1]) / pd.Timedelta(seconds=1)).astype(int)[same]})
    nxt.to_csv(OUT / 'next.csv', index=False)
    del k, keyed, nxt; gc.collect()

    cc = closed
    cc['report_filed'] = cc.report_filed.map(lambda x: 'true' if str(x).lower() in ('yes', 'true', '1') else 'false')
    cc['opened_at'] = pd.to_datetime(cc.opened_at).dt.strftime('%Y-%m-%d %H:%M:%S')
    cc['closed_at'] = pd.to_datetime(cc.closed_at).dt.strftime('%Y-%m-%d %H:%M:%S')
    cc[['case_id', 'customer_id', 'card_id', 'opened_at', 'closed_at', 'outcome', 'pattern', 'first_fraud_txn_id', 'n_txns',
        'exposure_usd', 'actions_taken', 'report_filed', 'analyst_notes']].to_csv(OUT / 'closed_cases.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    rows = [(r.case_id, int(float(t))) for _, r in cc.iterrows() for t in str(r.txn_ids).split('|') if t and t != 'nan']
    pd.DataFrame(rows, columns=['case_id', 'txn_id']).to_csv(OUT / 'closed_case_txns.csv', index=False)
    rows = [(r.case_id, c) for _, r in cc.iterrows() if isinstance(r.connected_card_ids, str) for c in r.connected_card_ids.split('|') if c]
    pd.DataFrame(rows, columns=['case_id', 'card_id']).to_csv(OUT / 'closed_case_connected.csv', index=False)

    pd.DataFrame([{'chunk_id': c['id'], 'title': c['title'], 'text': c['text']} for c in POLICY_CHUNKS]).to_csv(
        OUT / 'policy_chunks.csv', index=False, quoting=csv.QUOTE_ALL)
    print('exported to', OUT, {p.name: f"{p.stat().st_size // 1024} KB" for p in sorted(OUT.iterdir())})


def _conn():
    import pyTigerGraph as tg
    conn = tg.TigerGraphConnection(host=config.TG_HOST, graphname=config.TG_GRAPH, username=config.TG_USERNAME or 'tigergraph',
                                   password=config.TG_PASSWORD, gsqlSecret=config.TG_SECRET or None)
    if config.TG_SECRET:
        conn.getToken(config.TG_SECRET)
    else:
        conn.getToken(conn.createSecret())
    return conn


def schema():
    conn = _conn()
    for f in ['schema.gsql', 'load.gsql', 'queries/investigation.gsql']:
        print('>>', f)
        print(conn.gsql((ROOT / 'graph' / f).read_text()))


def load():
    conn = _conn()
    files = {'f_customers': 'customers.csv', 'f_cards': 'cards.csv', 'f_txns': 'transactions.csv', 'f_devices': 'devices.csv',
             'f_next': 'next.csv', 'f_closed': 'closed_cases.csv', 'f_closed_txn': 'closed_case_txns.csv',
             'f_closed_conn': 'closed_case_connected.csv', 'f_policy': 'policy_chunks.csv'}
    for fname, csvname in files.items():
        print('loading', csvname)
        print(conn.runLoadingJobWithFile(str(OUT / csvname), fname, 'load_fraud', sep=',', eol='\n', timeout=600000))


def embed_vertices():
    """Populate the TigerVector attributes (ClosedCase / PolicyChunk) so similar_cases & retrieve_policy run in-graph."""
    from fraud_agent.store.embeddings import embed
    conn = _conn()
    n = 0
    for v in conn.getVertices('PolicyChunk', limit=1000):
        conn.runInstalledQuery('set_embedding', {'vtype': 'PolicyChunk', 'id': v['v_id'], 'vec': embed(v['attributes']['text'])})
        n += 1
    for v in conn.getVertices('ClosedCase', limit=100000):
        a = v['attributes']
        conn.runInstalledQuery('set_embedding', {'vtype': 'ClosedCase', 'id': v['v_id'],
                                                 'vec': embed(f"{a.get('pattern')} {a.get('outcome')} {a.get('analyst_notes')}")})
        n += 1
        if n % 500 == 0:
            print('embedded', n, flush=True)
    print('embedded', n, 'vertices')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'export'
    {'export': export, 'schema': schema, 'load': load, 'embed': embed_vertices,
     'all': lambda: (export(), schema(), load(), embed_vertices())}[cmd]()
