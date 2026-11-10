"""Checks A to H of PROTOCOL_SHAP.md on the deployed Temporal Transformer (results/final_models.joblib).

    python experiments/audit_temporal_shap.py
"""
import json
import time
from itertools import combinations
from math import comb

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn

from common import OUT, ROOT, ctx, os
from deep_models import ARCH, HISTORY_YEARS, TEMPORAL_MEMBER, DeepClassifier
from explain import _sample, expected_gradients, temporal_reference, temporal_shap

SEED, N_ROWS, N_PERM_ROWS, TOP_K = 42, 3000, 300, 20
EX_YEARS = (2012, 2013, 2014)
TRAIN_END = 2010


class Linear(nn.Module):
    """Known answer: logit = sum of w * x, so attributions must equal w (x - mean reference)."""

    def __init__(self, shape):
        super().__init__()
        self.w = nn.Parameter(torch.randn(*shape, generator=torch.Generator().manual_seed(0)))

    def forward(self, x, m, p):
        return (self.w * x).sum((1, 2))


def verdict(ok):
    return 'PASS' if ok else 'FAIL'


def global_imp(values):
    return np.abs(values.sum(1)).mean(0)


def logits(net, x, m, p, batch=4096):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(*[torch.as_tensor(a[s:s + batch]) for a in (x, m, p)]).numpy() for s in range(0, len(x), batch)])


def check_a():
    """A linear network with some cells missing and some years absent."""
    rng = np.random.RandomState(0)
    x = rng.randn(40, HISTORY_YEARS, 6).astype(np.float32)
    m = (rng.rand(*x.shape) < 0.1).astype(np.float32)
    p = np.ones(x.shape[:2], dtype=np.float32)
    p[:10, :2] = 0
    x, m = x * p[..., None], m * p[..., None]
    base = rng.randn(25, HISTORY_YEARS, 6).astype(np.float32)
    net = Linear(x.shape[1:])
    phi, out, ref = expected_gradients(net, x, m, p, base, n_steps=4)
    observed = p[..., None] * (1 - m)
    want = net.w.detach().numpy() * (x - base.mean(0)) * observed
    return {'max_error': float(np.abs(phi - want).max()), 'max_unobserved': float(np.abs(phi[observed == 0]).max())}


def coalition_value(net, x, m, p, base):
    """value(mask): mean output of each row with its masked cells kept and the rest from the reference.
    Missing values and absent years always stay as they are, as in the attributions. mask: (K, f)."""
    obs = p[..., None] * (1 - m)

    def value(mask):
        keep = np.where(obs > 0, mask[None], True)
        return np.mean([logits(net, np.where(keep, x, b[None]), m, p) for b in base], axis=0)
    return value


def permutation_shapley(net, x, m, p, base, groups, n_perm, seed):
    """Shapley values by group: walk random orderings, switching a group on at a time. With antithetic
    orderings, as in shap.PermutationExplainer. groups: list of (K, f) boolean masks."""
    rng = np.random.RandomState(seed)
    value = coalition_value(net, x, m, p, base)
    phi = np.zeros((len(x), len(groups)))
    for _ in range(n_perm):
        order = rng.permutation(len(groups))
        for seq in (order, order[::-1]):
            mask = np.zeros(x.shape[1:], dtype=bool)
            prev = value(mask)
            for g in seq:
                mask = mask | groups[g]
                cur = value(mask)
                phi[:, g] += cur - prev
                prev = cur
    return phi / (2 * n_perm)


def exact_year_shapley(net, x, m, p, base):
    """All 32 coalitions of the 5 years. Rows are treated together; absent years have no effect."""
    k, f = x.shape[1:]
    value = coalition_value(net, x, m, p, base)
    v = {}
    for r in range(k + 1):
        for s in combinations(range(k), r):
            mask = np.zeros((k, f), dtype=bool)
            mask[list(s)] = True
            v[s] = value(mask)
    phi = np.zeros((len(x), k))
    for j in range(k):
        for s, val in v.items():
            if j in s:
                continue
            with_j = tuple(sorted(s + (j,)))
            phi[:, j] += (v[with_j] - val) / (k * comb(k - 1, len(s)))
    return phi


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    c = ctx()
    art = joblib.load(os.path.join(ROOT, 'results', 'final_models.joblib'))
    tm = art['temporal']
    feats = tm.features
    D = art['winsorizer'].transform(c.DW[art['winsor_cols']])
    ex = np.where(np.isin(c.years, EX_YEARS))[0]
    pool = np.where(c.years <= TRAIN_END)[0]
    rows = ex[_sample(np.random.RandomState(SEED), len(ex), N_ROWS)]
    base = temporal_reference(tm, D, c.gvkey, c.years, pool, seed=SEED)
    res = {'method': dict(n_rows=len(rows), n_reference=len(base), n_steps=16, networks=len(tm.clf_.nets_), years=list(EX_YEARS))}

    res['A'] = check_a()
    res['A']['pass'] = bool(res['A']['max_error'] < 1e-4 and res['A']['max_unobserved'] == 0)
    print('A', verdict(res['A']['pass']), res['A'], flush=True)

    t1 = time.time()
    r = temporal_shap(tm, D, c.gvkey, c.years, rows, base)
    res['method']['seconds_for_attributions'] = round(time.time() - t1, 1)
    print('attributions done', res['method']['seconds_for_attributions'], 's', flush=True)
    x, m, p, sv = r['x'], r['m'], r['p'], r['values']
    obs = p[..., None] * (1 - m)
    res['coverage'] = {'share_missing_of_present_cells': float(m[p > 0].mean()), 'rows_with_full_history': float((p > 0).all(1).mean())}

    # B: same network with and without gradients; ranking against the deployed probabilities
    gaps = []
    for n in tm.clf_.nets_:
        n.eval()
        with torch.enable_grad():
            on = n(*[torch.as_tensor(a) for a in (x, m, p)]).detach().numpy()
        gaps.append(float(np.abs(on - logits(n, x, m, p)).max()))
    deployed = tm.predict_scores(D, c.gvkey, c.years, rows)
    rho = float(spearmanr(r['output'], deployed)[0])
    res['B'] = {'max_gap': max(gaps), 'spearman_mean_logit_vs_deployed': rho, 'pass': bool(max(gaps) < 1e-4 and rho >= 0.99)}
    print('B', verdict(res['B']['pass']), res['B'], flush=True)

    # C: completeness
    err = np.abs(sv.sum((1, 2)) - (r['output'] - r['reference'])) / r['output'].std()
    res['C'] = {'median_error_in_sd': float(np.median(err)), 'p99_error_in_sd': float(np.percentile(err, 99)), 'output_sd': float(r['output'].std())}
    res['C']['pass'] = bool(res['C']['median_error_in_sd'] <= 0.01 and res['C']['p99_error_in_sd'] <= 0.05)
    print('C', verdict(res['C']['pass']), res['C'], flush=True)

    # D: permutation Shapley on a subsample, averaged over the networks, with a small reference set
    sub = np.random.RandomState(SEED + 1).choice(len(rows), N_PERM_ROWS, replace=False)
    small = base[:10]
    groups = [np.eye(len(feats), dtype=bool)[i][None].repeat(HISTORY_YEARS, 0) for i in range(len(feats))]
    t1 = time.time()
    perm = np.mean([permutation_shapley(n, x[sub], m[sub], p[sub], small, groups, 5, SEED) for n in tm.clf_.nets_], axis=0)
    print('permutation Shapley', round(time.time() - t1), 's', flush=True)
    eg = temporal_shap(tm, D, c.gvkey, c.years, rows[sub], small)
    eg_f = eg['values'].sum(1)
    row_r = [np.corrcoef(a, b)[0, 1] for a, b in zip(eg_f, perm) if a.std() > 0 and b.std() > 0]
    t1 = time.time()
    year_exact = np.mean([exact_year_shapley(n, x[sub], m[sub], p[sub], small) for n in tm.clf_.nets_], axis=0)
    print('exact year Shapley', round(time.time() - t1), 's', flush=True)
    share = lambda a: np.abs(a).mean(0) / np.abs(a).mean(0).sum()
    share_eg, share_ex = share(eg['values'].sum(2)), share(year_exact)
    res['D'] = {'global_spearman': float(spearmanr(np.abs(eg_f).mean(0), np.abs(perm).mean(0))[0]), 'median_row_pearson': float(np.median(row_r)),
                'year_share_expected_gradients': share_eg.tolist(), 'year_share_exact_shapley': share_ex.tolist(),
                'largest_year_share_gap': float(np.abs(share_eg - share_ex).max()),
                'top10_overlap': len(set(np.argsort(-np.abs(eg_f).mean(0))[:10]) & set(np.argsort(-np.abs(perm).mean(0))[:10]))}
    res['D']['pass'] = bool(res['D']['global_spearman'] >= 0.80 and res['D']['median_row_pearson'] >= 0.80 and res['D']['largest_year_share_gap'] <= 0.05)
    print('D', verdict(res['D']['pass']), res['D'], flush=True)

    # E: other references, across networks
    other = temporal_reference(tm, D, c.gvkey, c.years, pool, seed=SEED + 7)
    r2 = temporal_shap(tm, D, c.gvkey, c.years, rows, other)
    net_imp = [global_imp(v) for v in r['per_net']]
    pairs = [float(spearmanr(net_imp[i], net_imp[j])[0]) for i in range(len(net_imp)) for j in range(i + 1, len(net_imp))]
    res['E'] = {'spearman_other_reference': float(spearmanr(global_imp(sv), global_imp(r2['values']))[0]), 'network_pairs_spearman': pairs, 'mean_network_spearman': float(np.mean(pairs))}
    res['E']['pass'] = bool(res['E']['spearman_other_reference'] >= 0.95 and res['E']['mean_network_spearman'] >= 0.70)
    print('E', verdict(res['E']['pass']), res['E'], flush=True)

    # F: deletion. Reset cells to the reference and read the mean logit
    def after_reset(mask):
        return np.mean([np.mean([logits(n, np.where(mask, b[None], x), m, p) for b in base[:10]], axis=0) for n in tm.clf_.nets_], axis=0)

    rng = np.random.RandomState(SEED + 2)
    top_mask, rand_mask = np.zeros(sv.shape, dtype=bool), np.zeros(sv.shape, dtype=bool)
    for i in range(len(rows)):
        cand = np.flatnonzero(obs[i].ravel() > 0)
        pos = cand[np.argsort(-sv[i].ravel()[cand])][:TOP_K]
        np.put(top_mask[i], pos[sv[i].ravel()[pos] > 0], True)
        np.put(rand_mask[i], rng.choice(cand, min(TOP_K, len(cand)), replace=False), True)
    d_top, d_rand = r['output'] - after_reset(top_mask), r['output'] - after_reset(rand_mask)
    boot = np.random.RandomState(SEED + 3).randint(0, len(rows), (2000, len(rows)))
    diffs = (d_top - d_rand)[boot].mean(1)
    res['F'] = {'top_k': TOP_K, 'mean_drop_top': float(d_top.mean()), 'mean_drop_random': float(d_rand.mean()), 'ratio': float(d_top.mean() / d_rand.mean()) if d_rand.mean() > 0 else None,
                'difference_ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    res['F']['pass'] = bool(d_top.mean() >= 2 * max(d_rand.mean(), 1e-9) and res['F']['difference_ci95'][0] > 0)
    print('F', verdict(res['F']['pass']), res['F'], flush=True)

    # G: re-initialised weights, and a network trained on labels shuffled within each year
    arch = dict(TEMPORAL_MEMBER['arch_kw'], seq_len=HISTORY_YEARS)
    torch.manual_seed(SEED + 4)
    rand_net = ARCH[TEMPORAL_MEMBER['arch']](len(feats), **arch)
    g_rand = temporal_shap(tm, D, c.gvkey, c.years, rows, base, nets=[rand_net])['values']
    ytr = c.y[pool].copy()
    rng = np.random.RandomState(SEED + 5)
    for yr in np.unique(c.years[pool]):
        w = np.where(c.years[pool] == yr)[0]
        ytr[w] = rng.permutation(ytr[w])
    kw = {k: v for k, v in TEMPORAL_MEMBER.items() if k != 'n_members'}
    shuffled = DeepClassifier(seed=SEED + 6, n_members=1, **kw).fit(tm._history(D, c.gvkey, c.years, pool), ytr)
    g_lab = temporal_shap(tm, D, c.gvkey, c.years, rows, base, nets=shuffled.nets_)['values']
    placebo_auc = roc_auc_score(c.y[ex], shuffled.predict_proba(tm._history(D, c.gvkey, c.years, ex)))
    res['G'] = {'spearman_reinitialised': float(spearmanr(global_imp(sv), global_imp(g_rand))[0]), 'spearman_shuffled_labels': float(spearmanr(global_imp(sv), global_imp(g_lab))[0]),
                'shuffled_labels_network_auc_2012_2014': float(placebo_auc)}
    res['G']['pass'] = bool(abs(res['G']['spearman_reinitialised']) <= 0.30 and abs(res['G']['spearman_shuffled_labels']) <= 0.50)
    print('G', verdict(res['G']['pass']), res['G'], flush=True)

    # H: repeat
    again = temporal_shap(tm, D, c.gvkey, c.years, rows, base)['values']
    res['H'] = {'max_difference': float(np.abs(again - sv).max())}
    res['H']['pass'] = bool(res['H']['max_difference'] < 1e-3)
    print('H', verdict(res['H']['pass']), res['H'], flush=True)

    res['all_pass'] = all(res[k]['pass'] for k in 'ABCDEFGH')
    res['minutes'] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(os.path.join(OUT, 'audit_temporal_shap.json'), 'w'), indent=2)
    print('all pass:', res['all_pass'], res['minutes'], 'min', flush=True)


if __name__ == '__main__':
    main()
