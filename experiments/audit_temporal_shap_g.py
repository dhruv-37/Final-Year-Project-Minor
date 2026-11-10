"""Follow-up to the failed check G of PROTOCOL_SHAP.md. Chosen after seeing that result, so it explains the
failure and does not overturn it.

    python experiments/audit_temporal_shap_g.py

1. How much 10 random networks agree with the real model on global feature importance.
2. Whether that agreement comes from how much each input varies, which any network inherits.
3. Within a firm-year, whether a random network gives the same attributions as the real model.
4. Whether cells picked by a random network's attributions change the real model's output.
"""
import json

import joblib
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from audit_temporal_shap import EX_YEARS, N_ROWS, SEED, TOP_K, TRAIN_END, global_imp, logits
from common import OUT, ROOT, ctx, os
from deep_models import ARCH, HISTORY_YEARS, TEMPORAL_MEMBER
from explain import _sample, temporal_reference, temporal_shap

N_RANDOM = 10


def residual_rank(a, on):
    """Rank of a with the rank of `on` regressed out."""
    ra, ro = rankdata(a), rankdata(on)
    return ra - np.polyval(np.polyfit(ro, ra, 1), ro)


def top_positive_cells(sv, obs, k):
    mask = np.zeros(sv.shape, dtype=bool)
    for i in range(len(sv)):
        cand = np.flatnonzero(obs[i].ravel() > 0)
        pick = cand[np.argsort(-sv[i].ravel()[cand])][:k]
        np.put(mask[i], pick[sv[i].ravel()[pick] > 0], True)
    return mask


def main():
    c = ctx()
    art = joblib.load(os.path.join(ROOT, 'results', 'final_models.joblib'))
    tm = art['temporal']
    D = art['winsorizer'].transform(c.DW[art['winsor_cols']])
    ex = np.where(np.isin(c.years, EX_YEARS))[0]
    rows = ex[_sample(np.random.RandomState(SEED), len(ex), N_ROWS)]
    base = temporal_reference(tm, D, c.gvkey, c.years, np.where(c.years <= TRAIN_END)[0], seed=SEED)
    r = temporal_shap(tm, D, c.gvkey, c.years, rows, base)
    sv, x, m, p = r['values'], r['x'], r['m'], r['p']
    obs = p[..., None] * (1 - m)
    real = global_imp(sv)
    # how much each input moves from the reference, averaged over its observed cells
    spread = (np.abs(x - base.mean(0)) * obs).sum((0, 1)) / np.maximum(obs.sum((0, 1)), 1)
    observed_share = obs.mean((0, 1))

    def after_reset(mask):
        return np.mean([np.mean([logits(n, np.where(mask, b[None], x), m, p) for b in base[:10]], axis=0) for n in tm.clf_.nets_], axis=0)

    arch = dict(TEMPORAL_MEMBER['arch_kw'], seq_len=HISTORY_YEARS)
    draws = []
    kept = []
    for i in range(N_RANDOM):
        torch.manual_seed(SEED + 100 + i)
        net = ARCH[TEMPORAL_MEMBER['arch']](len(tm.features), **arch)
        g = temporal_shap(tm, D, c.gvkey, c.years, rows, base, nets=[net])['values']
        imp = global_imp(g)
        row_r = [np.corrcoef(a, b)[0, 1] for a, b in zip(sv.sum(1), g.sum(1)) if a.std() > 0 and b.std() > 0]
        kept.append(imp)
        draws.append({'spearman_global': float(spearmanr(real, imp)[0]),
                      'partial_given_spread': float(np.corrcoef(residual_rank(real, spread), residual_rank(imp, spread))[0, 1]),
                      'spearman_with_spread': float(spearmanr(imp, spread)[0]),
                      'median_row_pearson': float(np.median(row_r)),
                      'median_abs_row_pearson': float(np.median(np.abs(row_r))),
                      'top15_overlap': len(set(np.argsort(-real)[:15]) & set(np.argsort(-imp)[:15])),
                      'drop_in_real_model_from_its_top_cells': float((r['output'] - after_reset(top_positive_cells(g, obs, TOP_K))).mean())})
        print(i, draws[-1], flush=True)
    t = {k: [d[k] for d in draws] for k in draws[0]}
    res = {'n_random_networks': N_RANDOM, 'real_spearman_with_spread': float(spearmanr(real, spread)[0]),
           'real_spearman_with_observed_share': float(spearmanr(real, observed_share)[0]),
           'real_partial_vs_mean_random_given_spread': float(np.corrcoef(residual_rank(real, spread), residual_rank(np.mean(kept, 0), spread))[0, 1]),
           'summary': {k: {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v))} for k, v in t.items()}, 'draws': draws,
           'drop_in_real_model_from_real_top_cells': float((r['output'] - after_reset(top_positive_cells(sv, obs, TOP_K))).mean())}
    json.dump(res, open(os.path.join(OUT, 'audit_temporal_shap_g.json'), 'w'), indent=2)
    print({k: v for k, v in res.items() if k != 'draws'}, flush=True)


if __name__ == '__main__':
    main()
