"""Optional monitoring track: sweep a period for the riskiest alerts, run the agent on each, and write a period summary.

    python monitoring/monitor.py --start 2016-12-01 --end 2016-12-02 --limit 8

Outputs monitoring/<start>_<end>/summary.json + one answer JSON per alert. The MCP tool `alerts_in_period` and
`investigate_transaction` expose the same two steps to an external scheduler / LLM client.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraud_agent.agent import investigate  # noqa: E402
from fraud_agent.tools import store  # noqa: E402


def run(start: str, end: str, limit: int, min_score: float):
    out_dir = ROOT / 'monitoring' / f"{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)
    st = store()
    alerts = st.alerts_in_period(start, end, min_score=min_score, limit=limit)
    t0 = time.time()
    results = []
    for t in alerts:
        trig = {'case_id': f"MON-{t['txn_id']}", 'trigger_type': 'risk_score', 'customer_id': t['customer_id'], 'card_id': t['card_id'],
                'flagged_txn_id': int(t['txn_id']), 'opened_at': t['ts'], 'risk_score': t.get('risk_score'),
                'trigger_text': f"Monitoring sweep: transaction {t['txn_id']} (${t['amount']}) scored {t.get('risk_score')}"}
        ans, _ = investigate(trig)
        (out_dir / f"{trig['case_id']}.json").write_text(json.dumps(ans, indent=2, default=str))
        c = ans['case']
        results.append({'txn_id': t['txn_id'], 'customer_id': t['customer_id'], 'risk_score': t.get('risk_score'), 'amount': t['amount'],
                        'verdict': c['verdict'], 'p': c['fraud_probability'], 'pattern': c['pattern'], 'exposure_usd': c['exposure_usd'],
                        'sar': ans['sar']['file'], 'final': [a['action'] for a in ans['next_best_actions']['final']],
                        'connected_cards': len(c['connected_card_ids'])})
    summary = {
        'period': [start, end], 'alerts_reviewed': len(results), 'seconds': round(time.time() - t0, 1),
        'verdicts': dict(Counter(r['verdict'] for r in results)), 'patterns': dict(Counter(r['pattern'] for r in results)),
        'exposure_usd': round(sum(r['exposure_usd'] for r in results), 2), 'sars': sum(r['sar'] for r in results),
        'auto_closed_legitimate': sum(r['verdict'] == 'legitimate' for r in results),
        'needs_human': sum(any(a in ('BLOCK_CARD', 'FILE_REPORT', 'ESCALATE_TO_ANALYST') for a in r['final']) for r in results),
        'alerts': results,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != 'alerts'}, indent=1))
    for r in results:
        print(f"{r['txn_id']} {r['customer_id']} rs={r['risk_score']} ${r['amount']:<8} -> {r['verdict']:<10} {r['p']:.2f} {r['pattern']:<28} {'+'.join(r['final'])}")
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2016-12-01'); ap.add_argument('--end', default='2016-12-02')
    ap.add_argument('--limit', type=int, default=8); ap.add_argument('--min-score', type=float, default=0.9)
    a = ap.parse_args()
    run(a.start, a.end, a.limit, a.min_score)
