"""MD&A features for PROTOCOL_MDA.md. Rows of every returned block line up with the project's panel (ctx order).

Steps, each cached under results/experiments/cache/mda/:
  link()       MD&A -> fiscal year (Loughran-McDonald period of report) -> GVKEY (farr), usable if >= 1,000 words
  counts()     fixed unigram+bigram vocabulary from 8,000 MD&As of fiscal <= 1997; count matrix of every usable MD&A
  m2_block()   change and peer features (same or earlier fiscal years only)
  m1_block(k)  50 SVD components of TF-IDF, IDF and SVD fitted on window k's training MD&As
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import ROOT, ctx, load, path, split  # noqa: E402

MDA_DIR = os.path.join(ROOT, 'data', 'external', 'mda')
LM_CSV = os.path.join(ROOT, 'data', 'external', 'lm10x', 'Loughran-McDonald_10X_Summaries_1993-2025.csv')
LINKS_CSV = os.path.join(ROOT, 'data', 'external', 'farr', 'gvkey_ciks.csv')
ORIGINAL_FORMS = {'10-K', '10-K405', '10KSB', '10KSB40', '10-KT', '10-KSB', '10-K40'}
MAX_LAG, LINK_SLACK, MIN_WORDS, MAX_WORDS = 540, 90, 1000, 10000
VOCAB_YEAR, VOCAB_SAMPLE, N_VOCAB, MIN_DF, N_SVD, MIN_PEERS = 1997, 8000, 20000, 20, 50, 5
TOKENS = r'(?u)\b[a-zA-Z]{3,}\b'


def _save(p, obj):
    import pickle
    tmp = p + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, p)


# ------------------------------------------------------------------ 1. link
def _filings():
    lm = pd.read_csv(LM_CSV, usecols=['CIK', 'FILING_DATE', 'CPR', 'FORM_TYPE', 'SIC'], low_memory=False)
    lm = lm[lm['FORM_TYPE'].isin(ORIGINAL_FORMS)].copy()
    lm['filed'] = pd.to_datetime(lm['FILING_DATE'].astype(str), format='%Y%m%d', errors='coerce')
    lm['period'] = pd.to_datetime(lm['CPR'].astype(str), format='%Y%m%d', errors='coerce')
    lm = lm.dropna(subset=['filed', 'period'])
    lm['fyear'] = lm['period'].dt.year - (lm['period'].dt.month < 6).astype(int)
    lm['lag'] = (lm['filed'] - lm['period']).dt.days
    lm['fy_filed'] = lm['filed'].dt.year
    n = lm.groupby(['CIK', 'fy_filed']).size()
    lm = lm.set_index(['CIK', 'fy_filed'])
    lm = lm[n.reindex(lm.index).values == 1]                     # one original 10-K in that filing year
    lm = lm[lm['lag'].between(0, MAX_LAG)]
    return {k: (r.filed, int(r.fyear), int(r.SIC) // 100 if pd.notna(r.SIC) else -1)
            for k, r in zip(lm.index, lm.itertuples(index=False))}


def _links():
    L = pd.read_csv(LINKS_CSV, usecols=['gvkey', 'cik', 'first_date', 'last_date'], dtype={'gvkey': str})
    L['gvkey'] = pd.to_numeric(L['gvkey'], errors='coerce')
    L['first'] = pd.to_datetime(L['first_date'], unit='D', origin='unix', errors='coerce')
    L['last'] = pd.to_datetime(L['last_date'], unit='D', origin='unix', errors='coerce')
    L = L.dropna(subset=['gvkey', 'cik', 'first', 'last'])
    L['cik'] = L['cik'].astype(np.int64)
    L['gvkey'] = L['gvkey'].astype(np.int64)
    return L[['gvkey', 'cik', 'first', 'last']].drop_duplicates()


def link():
    """Index of usable MD&As in the panel (one per firm-year) and their texts, in the same order."""
    p_idx, p_txt = path('mda', 'index.pkl'), os.path.join(MDA_DIR, 'linked.jsonl.gz')
    if os.path.exists(p_idx) and os.path.exists(p_txt):
        return load(p_idx)
    c = ctx()
    panel = pd.DataFrame({'gvkey': c.gvkey.astype(np.int64), 'fyear': c.years.astype(np.int64)})
    panel['row'] = np.arange(len(panel))
    rowof = panel.set_index(['gvkey', 'fyear'])['row']
    lm, L = _filings(), _links()
    by_cik = {k: g for k, g in L.groupby('cik')}
    slack = pd.Timedelta(days=LINK_SLACK)
    rows, stats = [], {'records': 0, 'matched_10k': 0, 'usable_words': 0, 'linked': 0}
    tmp_txt = os.path.join(MDA_DIR, '_texts.jsonl.gz')
    sk = os.path.join(MDA_DIR, 'skipped.json')
    skipped = json.load(open(sk))['skipped_filing_years'] if os.path.exists(sk) else []
    stats['skipped_filing_years'] = skipped
    with gzip.open(tmp_txt, 'wt', encoding='utf-8') as out:
        for y in range(1993, 2021):
            f = os.path.join(MDA_DIR, f'mda_{y}.jsonl.gz')
            if y in skipped:                       # download rule 4: these firm-years count as having no MD&A
                continue
            if not os.path.exists(f):
                raise FileNotFoundError(f'{f}: run download_mda.py to the end first')
            for line in gzip.open(f, 'rt', encoding='utf-8'):
                d = json.loads(line)
                stats['records'] += 1
                fl = lm.get((d['cik'], d['year']))
                if fl is None:
                    continue
                stats['matched_10k'] += 1
                filed, fyear, sic2 = fl
                words = d['mda'].split()
                if len(words) < MIN_WORDS:
                    continue
                stats['usable_words'] += 1
                g = by_cik.get(d['cik'])
                if g is None:
                    continue
                g = g[(filed >= g['first'] - slack) & (filed <= g['last'] + slack)]
                hit = False
                for gv in g['gvkey'].unique():
                    r = rowof.get((int(gv), fyear))
                    if r is None:
                        continue
                    hit = True
                    rows.append({'row': int(r), 'gvkey': int(gv), 'fyear': fyear, 'cik': d['cik'], 'filed': filed,
                                 'sic2': sic2, 'n_words': len(words), 'tid': len(rows)})
                    out.write(json.dumps(' '.join(words[:MAX_WORDS])) + '\n')
                stats['linked'] += hit
            print(f'  link {y}: {stats}', flush=True)
    idx = pd.DataFrame(rows).sort_values(['row', 'filed', 'cik']).drop_duplicates('row')     # earliest filing per firm-year
    idx = idx.sort_values('tid')                                          # texts are written in this order
    keep = set(idx['tid'])
    with gzip.open(tmp_txt, 'rt', encoding='utf-8') as src, gzip.open(p_txt + '.tmp', 'wt', encoding='utf-8') as dst:
        for i, line in enumerate(src):
            if i in keep:
                dst.write(line)
    os.replace(p_txt + '.tmp', p_txt)
    os.remove(tmp_txt)
    idx = idx.drop(columns='tid').reset_index(drop=True)
    stats['firm_years'] = len(idx)
    _save(p_idx, {'index': idx, 'stats': stats})
    return load(p_idx)


def texts():
    with gzip.open(os.path.join(MDA_DIR, 'linked.jsonl.gz'), 'rt', encoding='utf-8') as fh:
        for line in fh:
            yield json.loads(line)


def usable():
    """Boolean per panel row: has a usable MD&A."""
    c = ctx()
    u = np.zeros(len(c.years), dtype=bool)
    u[link()['index']['row'].values] = True
    return u


# ------------------------------------------------------------------ 2. counts
def counts():
    p = path('mda', 'counts.npz')
    if not os.path.exists(p):
        from sklearn.feature_extraction.text import CountVectorizer
        idx = link()['index']
        early = np.where(idx['fyear'].values <= VOCAB_YEAR)[0]
        pick = set(np.random.RandomState(0).choice(early, min(VOCAB_SAMPLE, len(early)), replace=False))
        sample = [t for i, t in enumerate(texts()) if i in pick]
        cv = CountVectorizer(token_pattern=TOKENS, lowercase=True, stop_words='english', ngram_range=(1, 2),
                             min_df=MIN_DF, max_features=N_VOCAB, dtype=np.float32)
        cv.fit(sample)
        del sample
        vocab = cv.get_feature_names_out()
        cv2 = CountVectorizer(token_pattern=TOKENS, lowercase=True, stop_words='english', ngram_range=(1, 2),
                              vocabulary=vocab, dtype=np.float32)
        parts, batch = [], []
        for t in texts():
            batch.append(t)
            if len(batch) == 2000:
                parts.append(cv2.transform(batch))
                batch = []
                print(f'  counts: {sum(x.shape[0] for x in parts)} MD&As', flush=True)
        if batch:
            parts.append(cv2.transform(batch))
        X = sp.vstack(parts).tocsr()
        sp.save_npz(p, X)
        _save(path('mda', 'vocab.pkl'), list(vocab))
    return sp.load_npz(p)


def _tfidf(X, fit_rows):
    """Sublinear TF-IDF with the IDF from fit_rows only, rows L2-normalised."""
    from sklearn.preprocessing import normalize
    n = len(fit_rows)
    df = np.asarray((X[fit_rows] > 0).sum(axis=0)).ravel()
    idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
    T = X.copy()
    T.data = np.log1p(T.data)
    return normalize(T.multiply(idf).tocsr(), norm='l2', copy=False)


# ------------------------------------------------------------------ 3. blocks
def donors(has, years, seed=0):
    """Row whose values each row takes: itself if it has an MD&A, else a random row of its fiscal year that has
    one. Years with fewer than 500 (before EDGAR) draw from fiscal 1995-1997."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(has))
    wide = np.where(has & (years >= 1995) & (years <= 1997))[0]
    for yr in np.unique(years):
        rows = np.where(years == yr)[0]
        pool = rows[has[rows]]
        if len(pool) < 500:
            pool = wide
        missing = rows[~has[rows]]
        idx[missing] = rng.choice(pool, len(missing))
    return idx


def to_panel(doc_values, placebo=False):
    """Doc-level block (index order) -> panel-length block with donors. Placebo: values shuffled among the
    firm-years that have an MD&A, within each fiscal year, before the donors are drawn."""
    c = ctx()
    idx = link()['index']
    out = np.full((len(c.years), doc_values.shape[1]), np.nan, dtype=np.float32)
    vals = doc_values.copy()
    if placebo:
        rng = np.random.RandomState(7)
        fy = idx['fyear'].values
        for yr in np.unique(fy):
            r = np.where(fy == yr)[0]
            vals[r] = vals[rng.permutation(r)]
    out[idx['row'].values] = vals
    return out[donors(usable(), c.years)]


def m2_docs():
    """Per MD&A: log words, similarity to the firm's previous-year MD&A, distance to its year x SIC-2 centroid."""
    p = path('mda', 'm2_docs.pkl')
    if not os.path.exists(p):
        idx = link()['index']
        X = counts()
        T = _tfidf(X, np.where(idx['fyear'].values <= VOCAB_YEAR)[0])
        pos = {(g, f): i for i, (g, f) in enumerate(zip(idx['gvkey'], idx['fyear']))}
        prev = np.array([pos.get((g, f - 1), -1) for g, f in zip(idx['gvkey'], idx['fyear'])])
        sim = np.full(len(idx), np.nan, dtype=np.float32)
        has = prev >= 0
        a, b = np.where(has)[0], prev[has]
        sim[a] = np.asarray(T[a].multiply(T[b]).sum(axis=1)).ravel()
        dist = np.zeros(len(idx), dtype=np.float32)
        for fy, g in idx.groupby('fyear'):
            yr_rows = g.index.values
            cen_year = np.asarray(T[yr_rows].mean(axis=0)).ravel()
            for s2, h in g.groupby('sic2'):
                r = h.index.values
                cen = np.asarray(T[r].mean(axis=0)).ravel() if (len(r) >= MIN_PEERS and s2 >= 0) else cen_year
                cen = cen / (np.linalg.norm(cen) + 1e-12)
                dist[r] = 1 - T[r] @ cen
        # no prior-year MD&A: the year's median similarity (no flag, no NaN)
        fy = idx['fyear'].values
        for yr in np.unique(fy):
            r = np.where(fy == yr)[0]
            med = np.nanmedian(sim[r]) if np.isfinite(sim[r]).any() else np.nanmedian(sim)
            sim[r] = np.where(np.isnan(sim[r]), med, sim[r])
        vals = np.c_[np.log(idx['n_words'].values), sim, dist].astype(np.float32)
        _save(p, vals)
    return load(p)


M2_COLS = ['mda_log_words', 'mda_sim_prev', 'mda_peer_dist']


def m2_block(placebo=False):
    return pd.DataFrame(to_panel(m2_docs(), placebo), columns=M2_COLS)


def m1_docs(key):
    """50 SVD components per MD&A, TF-IDF and SVD fitted on the window's training firm-years only."""
    p = path('mda', 'm1', f'{key}.pkl')
    if not os.path.exists(p):
        from sklearn.decomposition import TruncatedSVD
        idx = link()['index']
        tr, _ = split(key)
        in_tr = np.isin(idx['row'].values, tr)
        fit_rows = np.where(in_tr)[0]
        T = _tfidf(counts(), fit_rows)
        svd = TruncatedSVD(N_SVD, algorithm='randomized', n_iter=5, random_state=0).fit(T[fit_rows])
        _save(p, svd.transform(T).astype(np.float32))
        print(f'  m1 {key}: fitted on {len(fit_rows)} MD&As, explained variance {svd.explained_variance_ratio_.sum():.3f}', flush=True)
    return load(p)


M1_COLS = [f'mda_c{i:02d}' for i in range(N_SVD)]


def m1_block(key, placebo=False):
    return pd.DataFrame(to_panel(m1_docs(key), placebo), columns=M1_COLS)
