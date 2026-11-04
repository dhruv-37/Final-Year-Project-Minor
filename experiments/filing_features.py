"""Filing features for PROTOCOL_TEXT.md, built from the Loughran-McDonald 10-X summaries and farr's link table.
Rows of the returned frames line up with the project's firm-year panel."""
import numpy as np
import pandas as pd

ORIGINAL_FORMS = {'10-K', '10-K405', '10KSB', '10KSB40', '10-KT', '10-KSB', '10-K40'}
SMALL_FORMS = {'10KSB', '10KSB40', '10-KSB'}
SHARES = ['N_Negative', 'N_Positive', 'N_Uncertainty', 'N_Litigious', 'N_StrongModal', 'N_WeakModal', 'N_Constraining',
          'N_Complexity', 'N_Negation']
MAX_LAG, LINK_SLACK, AMEND_YEARS = 540, 90, 3


def load_filings(lm_csv):
    cols = ['CIK', 'FILING_DATE', 'CPR', 'FORM_TYPE', 'FFInd', 'N_Words', 'N_Unique_Words', 'NetFileSize', 'N_Exhibits'] + SHARES
    lm = pd.read_csv(lm_csv, usecols=cols, low_memory=False)
    lm['filed'] = pd.to_datetime(lm['FILING_DATE'].astype(str), format='%Y%m%d', errors='coerce')
    lm['period'] = pd.to_datetime(lm['CPR'].astype(str), format='%Y%m%d', errors='coerce')
    return lm.dropna(subset=['filed', 'period'])


def load_links(links_csv):
    L = pd.read_csv(links_csv, usecols=['gvkey', 'cik', 'first_date', 'last_date'], dtype={'gvkey': str})
    L['gvkey'] = pd.to_numeric(L['gvkey'], errors='coerce')
    L['first'] = pd.to_datetime(L['first_date'], unit='D', origin='unix', errors='coerce')
    L['last'] = pd.to_datetime(L['last_date'], unit='D', origin='unix', errors='coerce')
    L = L.dropna(subset=['gvkey', 'cik', 'first', 'last'])
    L['cik'] = L['cik'].astype(np.int64)
    return L[['gvkey', 'cik', 'first', 'last']].drop_duplicates()


def match_filings(lm, links, gvkey, years):
    """One original filing per firm-year. Returns a frame aligned with the panel; NaN where none matched."""
    f = lm[lm['FORM_TYPE'].isin(ORIGINAL_FORMS)].copy()
    f['fyear'] = f['period'].dt.year - (f['period'].dt.month < 6).astype(int)
    f['lag'] = (f['filed'] - f['period']).dt.days
    f = f[f['lag'].between(0, MAX_LAG)]
    m = f.merge(links, left_on='CIK', right_on='cik')
    slack = pd.Timedelta(days=LINK_SLACK)
    m = m[(m['filed'] >= m['first'] - slack) & (m['filed'] <= m['last'] + slack)]
    m = m.sort_values(['gvkey', 'fyear', 'filed', 'CIK']).drop_duplicates(['gvkey', 'fyear'])
    panel = pd.DataFrame({'gvkey': gvkey.astype(np.int64), 'fyear': years.astype(np.int64)})
    m['gvkey'] = m['gvkey'].astype(np.int64)
    m['fyear'] = m['fyear'].astype(np.int64)
    return panel.merge(m, on=['gvkey', 'fyear'], how='left', validate='one_to_one')


def donors(linked, years, seed=0):
    """Row index whose values each row takes: itself if linked, else a random linked row of its fiscal year.
    Years with hardly any filings (before EDGAR) draw from the first well-covered years."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(linked))
    wide = np.where(linked & (years >= 1995) & (years <= 1997))[0]
    for yr in np.unique(years):
        rows = np.where(years == yr)[0]
        pool = rows[linked[rows]]
        if len(pool) < 500:
            pool = wide
        missing = rows[~linked[rows]]
        idx[missing] = rng.choice(pool, len(missing))
    return idx


def _rank(values, years, linked):
    return pd.Series(values).where(linked).groupby(years).rank(pct=True).values


def spell_prev(gvkey, years):
    prev = np.r_[-1, np.arange(len(years) - 1)]
    prev[np.r_[True, (gvkey[1:] != gvkey[:-1]) | (years[1:] != years[:-1] + 1)]] = -1
    return prev


def text_block(m, gvkey, years, linked):
    words = m['N_Words'].where(lambda s: s > 0)
    raw = {'txt_words': np.log(words), 'txt_unique': m['N_Unique_Words'] / words, 'txt_size': np.log(m['NetFileSize'].where(lambda s: s > 0)),
           'txt_exhibits': m['N_Exhibits']}
    raw.update({'txt_' + c[2:].lower(): m[c] / words for c in SHARES})
    out = {}
    prev = spell_prev(gvkey, years)
    for name, s in raw.items():
        r = _rank(s.values, years, linked)
        out[name + '_yr'] = r
        d = r - r[np.maximum(prev, 0)]
        d[prev < 0] = np.nan
        out[name + '_chg'] = d
    return pd.DataFrame(out)


def industry_block(m, ranks, years, linked):
    d = ranks.reset_index(drop=True).where(pd.Series(linked), np.nan)
    med = d.groupby([years, m['FFInd'].values]).transform('median')
    out = (d - med).add_suffix('_ind')
    out['ff48'] = m['FFInd'].values
    return out


def timing_block(m, years, linked):
    return pd.DataFrame({'file_lag_yr': _rank(m['lag'].values, years, linked), 'file_late': (m['lag'].values > 90).astype(float),
                         'file_small': m['FORM_TYPE'].isin(SMALL_FORMS).astype(float).values})


def amend_block(m, lm, linked):
    a = lm[lm['FORM_TYPE'].astype(str).str.endswith('-A')]
    by_cik = {k: np.sort(v.values) for k, v in a.groupby('CIK')['filed']}
    q = a[a['FORM_TYPE'].astype(str).str.startswith('10-K') | a['FORM_TYPE'].astype(str).str.startswith('10K')]
    k_by_cik = {k: np.sort(v.values) for k, v in q.groupby('CIK')['filed']}
    win = np.timedelta64(365 * AMEND_YEARS, 'D')

    def count(table):
        out = np.full(len(m), np.nan)
        for i in np.where(linked)[0]:
            t = table.get(int(m['CIK'].values[i]))
            filed = m['filed'].values[i]
            out[i] = 0 if t is None else np.searchsorted(t, filed, 'left') - np.searchsorted(t, filed - win, 'left')
        return out
    return pd.DataFrame({'amend_any_3y': count(by_cik), 'amend_10k_3y': count(k_by_cik)})


def build_blocks(lm, links, gvkey, years, ranks, seed=0):
    """{'T1'..'T4': frame} with the no-missingness rule applied, plus the linked mask."""
    m = match_filings(lm, links, gvkey, years)
    linked = m['filed'].notna().values
    blocks = {'T1': text_block(m, gvkey, years, linked), 'T2': industry_block(m, ranks, years, linked),
              'T3': timing_block(m, years, linked), 'T4': amend_block(m, lm, linked)}
    take = donors(linked, years, seed)
    return {k: v.iloc[take].reset_index(drop=True).astype(np.float32) for k, v in blocks.items()}, linked


def shuffled_block(b, years, linked, seed):
    """Placebo: linked rows trade filing features within their fiscal year; other rows are redrawn likewise."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(b))
    for yr in np.unique(years):
        rows = np.where(years == yr)[0]
        idx[rows] = rng.permutation(rows)
    return b.iloc[idx].reset_index(drop=True)
