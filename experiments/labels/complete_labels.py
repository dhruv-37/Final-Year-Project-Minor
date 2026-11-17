"""PROTOCOL_LABELS.md: complete the paper's fraud labels for fiscal 1991-2014 with later AAERs.

Sources: aaer_labels_pre2015.csv (read by hand from AAERs released 2019-2024) and the fiscal <= 2014 rows of
experiments/fresh/aaer_labels_fresh.csv. Filers are linked to GVKEY through the farr GVKEY-CIK table, within
the link's dates. The original data file is not changed.

    python experiments/labels/complete_labels.py
Output: experiments/labels/labels_completed.csv (gvkey, fyear, misstate, p_aaer for every firm-year of the
data) and experiments/labels/labels_added.csv (every label considered, and whether it was added).
"""
import io
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def read(path):
    return pd.read_csv(io.StringIO(''.join(ln for ln in open(path, encoding='utf-8') if not ln.startswith('#'))))


def candidates():
    rows = []
    for src, path in (('pre2015', os.path.join(HERE, 'aaer_labels_pre2015.csv')),
                      ('fresh', os.path.join(ROOT, 'experiments', 'fresh', 'aaer_labels_fresh.csv'))):
        for r in read(path).itertuples():
            for y in map(int, str(r.fyears).split(';')):
                if y <= 2014:
                    rows.append({'source': src, 'company': r.company, 'ciks': [int(c) for c in str(r.cik).split(';')],
                                 'fyear': y, 'tier': r.tier, 'case': int(str(r.aaer).split(';')[0])})
    return rows


def main():
    data = pd.read_csv(os.path.join(ROOT, 'data', 'data_FraudDetection_JAR2020.csv'), usecols=['gvkey', 'fyear', 'misstate', 'p_aaer'])
    link = pd.read_csv(os.path.join(ROOT, 'data', 'external', 'farr', 'gvkey_ciks.csv')).dropna(subset=['cik'])
    link['cik'] = link['cik'].astype(int)
    present = set(zip(data['gvkey'], data['fyear']))
    out, log = data.copy(), []
    for c in candidates():
        # a fiscal year ends between mid-year t and mid-year t+1; accept a link that is valid in that window
        lo = (pd.Timestamp(f'{c["fyear"]}-06-01') - pd.Timestamp('1970-01-01')).days
        hi = (pd.Timestamp(f'{c["fyear"] + 1}-05-31') - pd.Timestamp('1970-01-01')).days
        gv = link[link['cik'].isin(c['ciks']) & (link['first_date'] <= hi) & (link['last_date'].fillna(1e9) >= lo)]['gvkey'].unique()
        gv = [g for g in gv if (g, c['fyear']) in present]
        status = 'no GVKEY link' if not len(link[link['cik'].isin(c['ciks'])]) else 'firm-year not in data' if not gv else ''
        for g in gv:
            m = (out['gvkey'] == g) & (out['fyear'] == c['fyear'])
            if out.loc[m, 'misstate'].iloc[0] == 1:
                status = 'already fraud in the data'
            else:
                out.loc[m, 'misstate'], out.loc[m, 'p_aaer'] = 1, c['case']
                status = 'added'
        log.append({**{k: v for k, v in c.items() if k != 'ciks'}, 'ciks': ';'.join(map(str, c['ciks'])),
                    'gvkey': ';'.join(map(str, gv)), 'status': status})
    log = pd.DataFrame(log)
    log.to_csv(os.path.join(HERE, 'labels_added.csv'), index=False)
    out.to_csv(os.path.join(HERE, 'labels_completed.csv'), index=False)
    print(log['status'].value_counts().to_string())
    print(log[log.status != 'added'][['company', 'fyear', 'status']].to_string(index=False))
    before = data[data.misstate == 1].groupby('fyear').size()
    after = out[out.misstate == 1].groupby('fyear').size()
    print('frauds by fiscal year, 2008-2014: before -> after')
    print({int(y): f'{before.get(y, 0)} -> {after.get(y, 0)}' for y in range(2008, 2015)})
    print(f'total frauds {int(data.misstate.sum())} -> {int(out.misstate.sum())}')
    # the full data file with the completed labels, for retraining (step 2); git-ignored like the original
    full = pd.read_csv(os.path.join(ROOT, 'data', 'data_FraudDetection_JAR2020.csv'))
    full = full.drop(columns=['misstate', 'p_aaer']).merge(out, on=['gvkey', 'fyear'], how='left')[full.columns]
    full.to_csv(os.path.join(ROOT, 'data', 'data_completed_labels.csv'), index=False)


if __name__ == '__main__':
    main()
