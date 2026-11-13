"""Step 4: how close is the XBRL build to Compustat on the years both cover (fiscal 2011-2014)?

Links filers to GVKEYs with the public `farr` link table, then compares each raw item and each paper
ratio with the paper's dataset. Prints, per item: rows compared, share within 1%, share within 10%,
and the rank correlation.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRESH = os.path.join(ROOT, 'data', 'external', 'fresh')
ITEMS = ['at', 'act', 'lct', 'lt', 'ceq', 'che', 'ivst', 'rect', 'invt', 'ppegt', 'ivao', 'ap', 'txp', 'dlc', 'dltt', 'pstk',
         're', 'sale', 'cogs', 'dp', 'xint', 'txt', 'ib', 'ni', 'sstk', 'dltis', 'csho', 'prcc_f',
         'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa', 'issue', 'bm', 'dpi', 'reoa',
         'EBIT', 'ch_fcf']


def linked():
    x = pd.read_csv(os.path.join(FRESH, 'xbrl_firm_years.csv'))
    link = pd.read_csv(os.path.join(ROOT, 'data', 'external', 'farr', 'gvkey_ciks.csv')).dropna(subset=['cik'])
    link['cik'] = link['cik'].astype(int)
    day = (pd.to_datetime(x['fye']) - pd.Timestamp('1970-01-01')).dt.days
    m = x.assign(day=day).merge(link[['gvkey', 'cik', 'first_date', 'last_date']].rename(columns={'gvkey': 'gv'}), on='cik')
    m = m[(m['day'] >= m['first_date'] - 370) & (m['day'] <= m['last_date'] + 370)]
    return m.drop_duplicates(['cik', 'fyear']).drop(columns=['gvkey']).rename(columns={'gv': 'gvkey'})


if __name__ == '__main__':
    x = linked()
    ref = pd.read_csv(os.path.join(ROOT, 'data', 'data_FraudDetection_JAR2020.csv'))
    both = x.merge(ref, on=['gvkey', 'fyear'], suffixes=('', '_c'))
    both = both[both['fyear'].between(2011, 2014)]
    print(f'{len(both)} firm-years in both sources, fiscal 2011-2014; {both["gvkey"].nunique()} firms')
    print(f'Compustat firm-years 2011-2014: {ref["fyear"].between(2011, 2014).sum()}')
    for c in ITEMS:
        a, b = both[c], both[c + '_c']
        m = a.notna() & b.notna()
        rel = np.abs(a[m] - b[m]) / (np.abs(b[m]) + 1e-6)
        zero = (np.abs(b[m]) < 1e-9) & (np.abs(a[m]) < 1e-9)
        print(f'{c:12s} n={m.sum():6d} xbrl_missing={a.isna().mean():.3f} within1%={((rel <= .01) | zero).mean():.3f} '
              f'within10%={((rel <= .10) | zero).mean():.3f} spearman={a[m].corr(b[m], method="spearman"):.3f}')
