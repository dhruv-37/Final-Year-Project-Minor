"""Scoring helpers shared by the runners: CV summaries and stored candidate scores."""
import numpy as np
from sklearn.metrics import roc_auc_score

from common import SEEDS, ctx, keys, load, path, pickle, split
from evaluate import ndcg_at_k, sens_at_k
from models import ensemble_scores
from run_reference import MEMBERS, ref_path


def reference_ensemble(key, seed, members=MEMBERS):
    return ensemble_scores([load(ref_path(key, m, seed)) for m in members])


def cv_report(scores_by_seed):
    """{seed: {key: scores}} -> mean CV AUC / NDCG@5% / Sens@5% across folds and seeds, plus fold AUCs."""
    c = ctx()
    rows = []
    for s, by_key in scores_by_seed.items():
        for k, p in by_key.items():
            _, va = split(k)
            rows.append((k, roc_auc_score(c.y[va], p), ndcg_at_k(c.y[va], p, 0.05), sens_at_k(c.y[va], p, 0.05)))
    aucs = {k: float(np.mean([r[1] for r in rows if r[0] == k])) for k in sorted({r[0] for r in rows})}
    return {'cv_auc': float(np.mean([r[1] for r in rows])), 'cv_ndcg5': float(np.mean([r[2] for r in rows])),
            'cv_sens5': float(np.mean([r[3] for r in rows])), 'fold_auc': aucs}


def reference_cv(seeds=SEEDS, members=MEMBERS):
    return cv_report({s: {k: reference_ensemble(k, s, members) for k in keys('cv')} for s in seeds})


def save_candidate(name, seed, by_key):
    with open(path('candidates', f'{name}_s{seed}.pkl'), 'wb') as fh:
        pickle.dump(by_key, fh)
