"""Shared setup: the same data, folds, windows and reference model as run_pipeline.py, plus a resumable
task runner. Every result is cached as one file, so an interrupted run continues where it stopped."""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from features import Winsorizer, engineer  # noqa: E402
from models import BalancedBagger, LogReg, lgbm_factory, make_folds, mask_serial, xgb_factory  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'experiments')
CACHE = os.path.join(OUT, 'cache')
SEEDS = (42, 43, 44)
TEST_YEARS = tuple(range(2003, 2013))         # 2013-2014 have < 15 frauds: excluded from every mean
CV_VAL_YEARS = (1999, 2000, 2001)
N_BAGS = 10
_CTX = None


class Ctx:
    pass


def ctx():
    """Engineered data, cached on disk and once per process."""
    global _CTX
    if _CTX is not None:
        return _CTX
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'ctx.pkl')
    if not os.path.exists(path):
        df, groups = engineer(pd.read_csv(os.path.join(ROOT, 'data', 'data_FraudDetection_JAR2020.csv')))
        df = df[df['fyear'] >= 1991].reset_index(drop=True)
        cols = groups['raw28'] + groups['enhanced_rank']
        blob = dict(DW=df[cols].astype(np.float32), groups=groups, years=df['fyear'].values, y=df['misstate'].values,
                    paaer=df['p_aaer'].values, gvkey=df['gvkey'].values)
        with open(path, 'wb') as fh:
            pickle.dump(blob, fh)
    with open(path, 'rb') as fh:
        blob = pickle.load(fh)
    c = Ctx()
    c.__dict__.update(blob)
    c.feats = c.groups['enhanced_rank']
    s = json.load(open(os.path.join(ROOT, 'results', 'summary.json')))
    c.bestL, c.bestX = s['best_lgbm'], s['best_xgb']
    _CTX = c
    return c


def split(key):
    """key = 'val1999' (rolling-origin fold) or 'test2003' -> (train rows, evaluation rows)."""
    c = ctx()
    kind, year = key[:-4], int(key[-4:])
    if kind == 'val':
        return next((tr, va) for tr, va in make_folds(c.years) if c.years[va][0] == year)
    return np.where(c.years <= year - 2)[0], np.where(c.years == year)[0]


def keys(stage):
    return [f'val{y}' for y in CV_VAL_YEARS] if stage == 'cv' else [f'test{y}' for y in TEST_YEARS]


def winsorized(tr):
    c = ctx()
    return Winsorizer().fit(c.DW.iloc[tr]).transform(c.DW)


def fit_member(member, D, feats, tr, te, ytr, seed, n_bags, n_jobs):
    """Fit one reference member on `feats` and score the evaluation rows."""
    c = ctx()
    X = D[feats].values
    if member == 'LGBM':
        m = BalancedBagger(lgbm_factory(c.bestL, n_jobs), n_bags, c.bestL['ratio'], seed)
    elif member == 'XGB':
        m = BalancedBagger(xgb_factory(c.bestX, n_jobs), n_bags, c.bestX['ratio'], seed + 1)
    else:
        m = LogReg()
    return m.fit(X[tr], ytr).predict_proba(X[te])


def path(*parts):
    p = os.path.join(CACHE, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _run_one(fn, out_path, kw, threads):
    if os.path.exists(out_path):
        return 0.0
    os.environ['OMP_NUM_THREADS'] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass
    t0 = time.time()
    res = fn(**kw, n_jobs=threads)
    tmp = out_path + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(res, fh)
    os.replace(tmp, out_path)
    print(f'  done {os.path.relpath(out_path, CACHE)} [{time.time() - t0:.0f}s]', flush=True)
    return time.time() - t0


def run_tasks(tasks, n_workers=4, threads=4):
    """tasks: [(fn, out_path, kwargs)]; fn must be a module-level function taking n_jobs."""
    todo = [t for t in tasks if not os.path.exists(t[1])]
    print(f'{len(todo)} of {len(tasks)} tasks to run ({n_workers} workers x {threads} threads)', flush=True)
    Parallel(n_jobs=n_workers, backend='loky')(delayed(_run_one)(f, p, k, threads) for f, p, k in todo)


def load(p):
    with open(p, 'rb') as fh:
        return pickle.load(fh)
