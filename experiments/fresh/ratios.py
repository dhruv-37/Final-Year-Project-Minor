"""The 14 ratios of Bao et al. (2020), rebuilt from raw items (definitions from Dechow et al. 2011).

`ratios14(df)` needs gvkey, fyear, the 28 raw items and, for soft_assets and dpi, net PP&E (`ppent`).
Running this file checks the formulas against the ratio columns shipped in the paper's dataset.
"""
import numpy as np
import pandas as pd


def _div(a, b):
    return (a / b.where(b != 0)).replace([np.inf, -np.inf], np.nan)


def ratios14(df):
    d = df.sort_values(['gvkey', 'fyear']).reset_index(drop=True)
    g = d.groupby('gvkey')
    ok = g['fyear'].shift(1) == d['fyear'] - 1           # previous row is the previous fiscal year

    def lag(s):
        return s.groupby(d['gvkey']).shift(1).where(ok)

    z = lambda c: d[c].fillna(0)                           # noqa: E731  (Compustat-style: missing component = 0)
    at, at_l = d['at'], lag(d['at'])
    avg_at = (at + at_l) / 2
    wc = (d['act'] - d['che']) - (d['lct'] - z('dlc') - z('txp'))
    nco = (at - d['act'] - z('ivao')) - (d['lt'] - d['lct'] - z('dltt'))
    fin = (z('ivst') + z('ivao')) - (z('dltt') + z('dlc') + z('pstk'))
    rsst = (wc - lag(wc)) + (nco - lag(nco)) + (fin - lag(fin))
    d_rect, d_invt, d_ap = d['rect'] - lag(d['rect']), d['invt'] - lag(d['invt']), d['ap'] - lag(d['ap'])
    cs = d['sale'] - d_rect
    cm = 1 - _div(d['cogs'] - d_invt + d_ap, cs)
    roa = _div(d['ni'], avg_at)
    out = pd.DataFrame({
        'dch_wc': _div(wc - lag(wc), avg_at),
        'ch_rsst': _div(rsst, avg_at),
        'dch_rec': _div(d_rect, avg_at),
        'dch_inv': _div(d_invt, avg_at),
        'ch_cs': _div(cs - lag(cs), lag(cs)),
        'ch_cm': _div(cm - lag(cm), lag(cm)),
        'ch_roa': roa - lag(roa),
        'issue': ((z('sstk') > 0) | (z('dltis') > 0)).astype(float),
        'bm': _div(d['ceq'], d['csho'] * d['prcc_f']),
        'reoa': _div(d['re'], at),
        'EBIT': _div(d['ni'] + z('xint') + z('txt'), at),
        'ch_fcf': _div(d['ib'] - lag(d['ib']) - rsst, avg_at),
    })
    if 'ppent' in d:
        out['soft_assets'] = _div(at - d['ppent'] - d['che'], at)
        r = _div(d['dp'], d['dp'] + d['ppent'])
        out['dpi'] = _div(lag(r), r)
    return pd.concat([d[['gvkey', 'fyear']], out], axis=1)


if __name__ == '__main__':
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ref = pd.read_csv(os.path.join(root, 'data', 'data_FraudDetection_JAR2020.csv'))
    mine = ratios14(ref).merge(ref, on=['gvkey', 'fyear'], suffixes=('', '_paper'))
    for c in [c for c in mine.columns if c + '_paper' in mine.columns]:
        a, b = mine[c], mine[c + '_paper']
        m = a.notna() & b.notna()
        close = (np.abs(a[m] - b[m]) <= 1e-3 * (1 + np.abs(b[m]))).mean()
        print(f'{c:12s} both={m.sum():6d}  mine_nan/paper_ok={(a.isna() & b.notna()).sum():6d}  '
              f'paper_nan/mine_ok={(a.notna() & b.isna()).sum():6d}  match={close:.3f}  '
              f'spearman={a[m].corr(b[m], method="spearman"):.3f}')
