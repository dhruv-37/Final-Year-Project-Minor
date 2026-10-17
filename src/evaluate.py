"""Metrics used in the fraud-detection literature (AUC, PR-AUC, NDCG@k, sensitivity@k) plus
decision metrics for an investigation budget (precision@k, lift, confusion matrix)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

BOOT_METRICS = ('AUC', 'NDCG@5%', 'Sens@5%')


def rank_norm(p):
    """Map scores to (0,1] by rank so differently-scaled models can be averaged."""
    return rankdata(p) / len(p)


def top_k(y, p, k_frac):
    """(k, frauds among the top k) for the top k_frac highest-risk firm-years."""
    y = np.asarray(y)
    k = max(1, int(np.ceil(k_frac * len(y))))
    return k, int(y[np.argsort(-np.asarray(p), kind='stable')[:k]].sum())


def ndcg_at_k(y, p, k_frac=0.01):
    """NDCG@k with binary relevance (as in Bao et al. 2020), k = top k_frac of the sample."""
    y = np.asarray(y)
    n = len(y)
    k = max(1, int(np.ceil(k_frac * n)))
    order = np.argsort(-np.asarray(p), kind='stable')[:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (y[order] * disc).sum()
    ideal = disc[:min(int(y.sum()), k)].sum()
    return dcg / ideal if ideal > 0 else np.nan


def sens_at_k(y, p, k_frac=0.01):
    """Share of all frauds captured in the top k_frac highest-risk firm-years."""
    return top_k(y, p, k_frac)[1] / max(1, np.sum(y))


# ---------------------------------------------------------------- paper-style yearly evaluation
def per_year_metrics(y, p, years):
    """One row per test year: AUC, PR-AUC, NDCG@1/5%, sensitivity@1/5/10% (as in Bao et al.)
    + precision and lift (precision / base rate) when investigating the top 1%."""
    y, p, years = map(np.asarray, (y, p, years))
    rows = []
    for yr in np.unique(years):
        m = years == yr
        if y[m].sum() == 0:
            continue
        k, tp = top_k(y[m], p[m], 0.01)
        rows.append({'year': int(yr), 'frauds': int(y[m].sum()),
                     'AUC': roc_auc_score(y[m], p[m]),
                     'PR_AUC': average_precision_score(y[m], p[m]),
                     'NDCG@1%': ndcg_at_k(y[m], p[m], 0.01),
                     'NDCG@5%': ndcg_at_k(y[m], p[m], 0.05),
                     'Sens@1%': sens_at_k(y[m], p[m], 0.01),
                     'Sens@5%': sens_at_k(y[m], p[m], 0.05),
                     'Sens@10%': sens_at_k(y[m], p[m], 0.10),
                     'Prec@1%': tp / k, 'Lift@1%': tp / k / y[m].mean()})
    return pd.DataFrame(rows)


def decision_metrics(y, p, years, ks=(0.01, 0.02, 0.05)):
    """Per test year and budget k (share of that year's firms investigated): confusion matrix,
    precision@k, recall@k (= Sens@k) and lift = precision / base rate."""
    y, p, years = map(np.asarray, (y, p, years))
    rows = []
    for yr in np.unique(years):
        m = years == yr
        n, fr = int(m.sum()), int(y[m].sum())
        for kf in ks:
            k, tp = top_k(y[m], p[m], kf)
            rows.append({'year': int(yr), 'k': f'{kf:.0%}', 'n': n, 'frauds': fr, 'investigated': k,
                         'TP': tp, 'FP': k - tp, 'FN': fr - tp, 'TN': n - k - (fr - tp),
                         'precision': tp / k, 'recall': tp / fr if fr else np.nan,
                         'lift': tp / k / (fr / n) if fr else np.nan})
    return pd.DataFrame(rows)


def period_mean(pym, lo, hi, min_frauds=0):
    """Average the yearly metrics over test years lo..hi with >= min_frauds frauds."""
    d = pym[(pym['year'] >= lo) & (pym['year'] <= hi) & (pym['frauds'] >= min_frauds)]
    return d.drop(columns=['year', 'frauds']).mean()


# ---------------------------------------------------------------- clustered paired bootstrap
def _boot_stats(y, p):
    """AUC, NDCG@5%, Sens@5% (same definitions as above, one sort each)."""
    n, npos = len(y), int(y.sum())
    auc = (rankdata(p)[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (n - npos))
    k = max(1, int(np.ceil(0.05 * n)))
    hit = y[np.argsort(-p, kind='stable')[:k]]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    return np.array([auc, (hit * disc).sum() / disc[:min(npos, k)].sum(), hit.sum() / npos])


def _clusters(keys):
    _, inv = np.unique(keys, return_inverse=True)
    sizes = np.bincount(inv)
    return np.argsort(inv, kind='stable'), np.cumsum(sizes) - sizes, sizes


def _draw(rng, rows, cl):
    """Resample whole clusters with replacement; returns the row indices of the drawn clusters."""
    order, starts, sizes = cl
    pick = rng.randint(len(sizes), size=len(sizes))
    ln = sizes[pick]
    off = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
    return rows[order[np.repeat(starts[pick], ln) + off]]


def bootstrap_period(y, preds, years, pos_key, neg_key, use_years, n_boot=1000, seed=0):
    """Paired bootstrap of the year-averaged BOOT_METRICS for every model in `preds`. Inside each
    test year, frauds are resampled by AAER case (pos_key) and non-frauds by firm (neg_key).
    Returns {model: array (n_boot, len(BOOT_METRICS))}."""
    rng = np.random.RandomState(seed)
    y, years = np.asarray(y), np.asarray(years)
    groups = []
    for yr in use_years:
        m = np.where(years == yr)[0]
        pos, neg = m[y[m] == 1], m[y[m] == 0]
        groups.append((pos, _clusters(pos_key[pos]), neg, _clusters(neg_key[neg])))
    out = {k: np.zeros((n_boot, len(BOOT_METRICS))) for k in preds}
    for b in range(n_boot):
        for pos, cp, neg, cn in groups:
            idx = np.concatenate([_draw(rng, pos, cp), _draw(rng, neg, cn)])
            for k, p in preds.items():
                out[k][b] += _boot_stats(y[idx], p[idx])
    return {k: v / len(groups) for k, v in out.items()}
