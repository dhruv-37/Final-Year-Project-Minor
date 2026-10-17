"""Explainability: global + local SHAP for the deployed ensemble (its actual fitted members).

Outputs (in results/explain/):
  shap_beeswarm.png, shap_bar.png   - global importance / direction (tree members, log-odds)
  shap_dependence_*.png             - how the top features push risk up or down
  waterfall_*.png                   - why a specific firm-year was flagged
  top_flagged_explained.csv         - highest-risk firm-years (deployed score) + top-3 drivers in words
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from features import describe


def _shap_pos(sv):
    """Normalise the different shapes shap returns for binary models to (n, f)."""
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    return sv


def member_shap(m, X, rng):
    """Balanced bagger of trees -> TreeExplainer on EVERY bag: (bags, n, f) log-odds values, base.
    Anything else (RUSBoost, LR) -> PermutationExplainer on <=300 rows: (1, n_sub, f), row idx."""
    if hasattr(m, 'models_'):
        vals, bases = [], []
        for b in m.models_:
            ex = shap.TreeExplainer(b)
            vals.append(_shap_pos(ex.shap_values(X)))
            bases.append(float(np.atleast_1d(ex.expected_value)[-1]))
        return np.stack(vals), float(np.mean(bases)), None
    sub = rng.choice(len(X), min(300, len(X)), replace=False)
    bg = shap.maskers.Independent(X[rng.choice(len(X), 10, replace=False)])
    sv = shap.PermutationExplainer(m.predict_proba, bg)(X[sub], max_evals=2 * X.shape[1] + 1).values
    return sv[None], None, sub


def explain_all(models, X_df, y, meta, scores, out_dir, n_sample=3000, top_n=15, seed=0, log=print):
    """models: {name: fitted deployed member}; X_df: winsorised features of the explain years;
    y: labels; meta: DataFrame(gvkey, fyear); scores: DataFrame(risk_percentile, risk_score) = the
    deployed ensemble's output for the same rows. Global plots use a random sample (real base rate)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(X_df), min(n_sample, len(X_df)), replace=False))
    Xs = X_df.iloc[idx]
    Xv = Xs.values.astype(np.float32)
    log(f'  computing SHAP on a random sample of {len(idx)} rows ({int(y[idx].sum())} frauds, real base rate)')
    imp = pd.DataFrame({'feature': X_df.columns})
    tree_sv, tree_base = [], []
    for name, m in models.items():
        sv_m, base_m, sub = member_shap(m, Xv, rng)
        imp[f'mean_abs_shap_{name}'] = np.abs(sv_m).mean(axis=(0, 1))
        if sub is None:
            log(f'  {name}: TreeExplainer on all {len(sv_m)} bags')
            tree_sv.append(sv_m.mean(0))
            tree_base.append(base_m)
        else:
            log(f'  {name}: not a tree model -> PermutationExplainer on {len(sub)} rows '
                f'(probability units; importance column only, not in the plots)')
    sv, base = np.mean(tree_sv, axis=0), float(np.mean(tree_base))

    # ---- global importance ------------------------------------------------------
    tree_cols = [f'mean_abs_shap_{k}' for k, m in models.items() if hasattr(m, 'models_')]
    imp.insert(1, 'mean_abs_shap', imp[tree_cols].mean(axis=1))
    imp['meaning'] = imp['feature'].map(describe)
    imp = imp.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(out_dir, 'feature_importance_shap.csv'), index=False)

    expl = shap.Explanation(values=sv, base_values=np.full(len(sv), base),
                            data=Xs.values, feature_names=list(X_df.columns))
    plt.figure()
    shap.plots.beeswarm(expl, max_display=top_n, show=False)
    plt.title('SHAP summary - what drives fraud risk')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_beeswarm.png'), dpi=150)
    plt.close()

    plt.figure()
    shap.plots.bar(expl, max_display=top_n, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_bar.png'), dpi=150)
    plt.close()

    # ---- dependence plots for the top-4 features -------------------------------------
    for f in imp['feature'].head(4):
        j = list(X_df.columns).index(f)
        plt.figure(figsize=(5.5, 4))
        plt.scatter(Xs.values[:, j], sv[:, j], s=6, alpha=0.4)
        plt.axhline(0, color='grey', lw=0.8)
        plt.xlabel(f)
        plt.ylabel('SHAP value (log-odds)')
        plt.title(f'{f}: {describe(f)[:60]}', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'shap_dependence_{f}.png'), dpi=150)
        plt.close()

    # ---- local explanations: top-risk firm-years in the sample (deployed score) ----------
    sc = scores.iloc[idx].reset_index(drop=True)
    ranked = np.argsort(-sc['risk_percentile'].values, kind='stable')
    order = ranked[:30]
    rows = []
    cols = np.array(X_df.columns)
    for r in order:
        top = np.argsort(-np.abs(sv[r]))[:3]
        drivers = '; '.join(
            f"{cols[j]}={Xs.values[r, j]:.3g} ({'raises' if sv[r, j] > 0 else 'lowers'} risk, "
            f"{sv[r, j]:+.2f})" for j in top)
        m = meta.iloc[idx[r]]
        rows.append({'gvkey': int(m['gvkey']), 'fyear': int(m['fyear']),
                     'risk_percentile': sc['risk_percentile'][r], 'risk_score': sc['risk_score'][r],
                     'actual_fraud': int(y[idx[r]]), 'top_drivers': drivers})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'top_flagged_explained.csv'), index=False)

    # waterfall for the 3 highest-scoring true frauds + 1 high-scoring false alarm
    tp = [r for r in ranked if y[idx[r]] == 1][:3]
    fp = [r for r in ranked if y[idx[r]] == 0][:1]
    for tag, rs in (('fraud', tp), ('falsealarm', fp)):
        for k, r in enumerate(rs):
            plt.figure()
            shap.plots.waterfall(expl[int(r)], max_display=10, show=False)
            m = meta.iloc[idx[r]]
            plt.title(f'{tag}: gvkey {int(m["gvkey"])}, FY{int(m["fyear"])}', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'waterfall_{tag}_{k + 1}.png'), dpi=150)
            plt.close()
    return imp
