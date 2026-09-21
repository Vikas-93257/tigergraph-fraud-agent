"""Run the agent on every case in case_pack.csv and write cases/<case_id>.json.

    python run_cases.py            # all 20
    python run_cases.py HHG-006    # one case, verbose
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from fraud_agent import config
from fraud_agent.agent import investigate
from fraud_agent.tools import store


def load_case_pack() -> list[dict]:
    cp = pd.read_csv(config.DATA_DIR / 'case_pack.csv')
    out = []
    for _, r in cp.iterrows():
        trig = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
        trig['flagged_txn_id'] = int(trig['flagged_txn_id'])
        out.append(trig)
    return out


def main(only: list[str] | None = None):
    cp = pd.read_csv(config.DATA_DIR / 'case_pack.csv')
    config.CASES_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    print('loading graph store…', flush=True)
    print(store().describe(), flush=True)
    rows = []
    for _, r in cp.iterrows():
        if only and r.case_id not in only:
            continue
        trig = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
        trig['flagged_txn_id'] = int(trig['flagged_txn_id'])
        ans, st = investigate(trig)
        (config.CASES_DIR / f"{r.case_id}.json").write_text(json.dumps(ans, indent=2, default=str))
        trace_dir = config.ROOT / 'traces'; trace_dir.mkdir(exist_ok=True)
        (trace_dir / f"{r.case_id}.json").write_text(json.dumps({
            'case_id': r.case_id, 'trigger': trig, 'txn': st.get('txn'), 'window': st.get('window'), 'profile': st.get('profile'),
            'device': st.get('device'), 'peers': st.get('peers'), 'signals': [x.as_evidence() | {'weight': x.weight, 'key': x.key} for x in st.get('signals', [])],
            'initial_prob': st.get('initial_prob'), 'tool_log': st['tool_log']}, indent=1, default=str))
        c = ans['case']
        rows.append({'case': r.case_id, 'trigger': r.trigger_type, 'verdict': c['verdict'], 'p': c['fraud_probability'],
                     'pattern': c['pattern'], 'exposure': c['exposure_usd'], 'n_txn': len(c['affected_txn_ids']),
                     'sar': ans['sar']['file'], 'initial': '+'.join(a['action'] for a in ans['next_best_actions']['initial']),
                     'final': '+'.join(a['action'] for a in ans['next_best_actions']['final']), 'tools': ans['tool_calls']})
        if only:
            print(json.dumps(ans, indent=2, default=str))
            print('\n--- tool log ---')
            for c_ in st['tool_log']:
                print(f"{c_['step']:>2} {c_['tool']:<26} {c_['ms']:>7.1f}ms  {json.dumps(c_['args'], default=str)[:110]}")
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
    print(df.to_string(index=False))
    print(f"\n{len(rows)} cases in {time.time()-t0:.1f}s -> {config.CASES_DIR}")


if __name__ == '__main__':
    main(sys.argv[1:] or None)
