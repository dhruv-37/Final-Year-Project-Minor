"""Explainability: global + local SHAP for the deployed ensemble (its actual fitted members).

Outputs (in results/explain/):
  shap_beeswarm.png, shap_bar.png   - global importance / direction (tree members, log-odds)
  shap_dependence_*.png             - how the top features push risk up or down
  waterfall_*.png                   - why a specific firm-year was flagged
  top_flagged_explained.csv         - highest-risk firm-years (deployed score) + top-3 drivers in words,
                                      each member's rank and the Temporal Transformer's own drivers
  temporal_shap_*.png, temporal_shap_features.csv, temporal_shap_years.csv
                                    - the Temporal Transformer: which features and which past years
  waterfall_tt_*.png                - why it scored a specific firm-year as it did
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from deep_models import HISTORY_YEARS, history_index
from features import describe


def _sample(rng, n, n_sample):
    return np.sort(rng.choice(n, min(n_sample, n), replace=False))


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
    # without a seed the explainer reseeds numpy from the OS, and the values change on every run
    sv = shap.PermutationExplainer(m.predict_proba, bg, seed=int(rng.randint(2 ** 31 - 1)))(X[sub], max_evals=2 * X.shape[1] + 1).values
    return sv[None], None, sub


def explain_all(models, X_df, y, meta, scores, out_dir, n_sample=3000, top_n=15, seed=0, log=print):
    """models: {name: fitted deployed member}; X_df: winsorised features of the explain years;
    y: labels; meta: DataFrame(gvkey, fyear); scores: DataFrame(risk_percentile, risk_score) = the
    deployed ensemble's output for the same rows. Global plots use a random sample (real base rate)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    idx = _sample(rng, len(X_df), n_sample)
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


# ---------------------------------------------------------------------- Temporal Transformer
YEAR_LABELS = [f't-{HISTORY_YEARS - 1 - j}' if j < HISTORY_YEARS - 1 else 't' for j in range(HISTORY_YEARS)]


def expected_gradients(net, x, m, p, base, n_steps=16, chunk=16):
    """Expected gradients (the method of shap.GradientExplainer) of one network's logit, on a fixed grid so
    that attributions sum to output minus reference output. x, m: (rows, K, f); p: (rows, K); base:
    (refs, K, f). Only observed values move: a row's missing-value flags and absent years stay as they
    are, so they get no attribution. Returns attributions (rows, K, f), outputs and reference outputs."""
    net.eval()
    x, m, p, base = [torch.as_tensor(a, dtype=torch.float32) for a in (x, m, p, base)]
    alpha = ((torch.arange(n_steps) + 0.5) / n_steps)[None, None, :, None, None]
    n_ref = len(base)
    phi, out, ref = np.zeros(x.shape, dtype=np.float32), np.zeros(len(x)), np.zeros(len(x))
    for s in range(0, len(x), chunk):
        xb, mb, pb = x[s:s + chunk], m[s:s + chunk], p[s:s + chunk]
        c = len(xb)
        observed = (pb[..., None] * (1 - mb))[:, None]
        start = base[None] * observed + xb[:, None] * (1 - observed)                  # (c, refs, K, f)
        delta = xb[:, None] - start
        points = (start[:, :, None] + alpha * delta[:, :, None]).reshape(-1, *xb.shape[1:]).requires_grad_(True)
        mm, pp = [a.repeat_interleave(n_ref * n_steps, 0) for a in (mb, pb)]
        grads = torch.autograd.grad(net(points, mm, pp).sum(), points)[0].reshape(c, n_ref, n_steps, *xb.shape[1:])
        phi[s:s + c] = (grads.mean(2) * delta).mean(1).detach().numpy()
        out[s:s + c] = net(xb, mb, pb).detach().numpy()
        start_out = net(start.reshape(-1, *xb.shape[1:]), mb.repeat_interleave(n_ref, 0), pb.repeat_interleave(n_ref, 0))
        ref[s:s + c] = start_out.detach().reshape(c, n_ref).mean(1).numpy()
    return phi, out, ref


def temporal_reference(member, D, gvkey, years, pool, n_ref=50, seed=0):
    """Random firm-years with a complete history, drawn from the member's training rows."""
    complete = pool[(history_index(gvkey, years, HISTORY_YEARS)[pool] >= 0).all(1)]
    rows = np.random.RandomState(seed).choice(complete, n_ref, replace=False)
    return member._history(D, gvkey, years, rows)[0]


def temporal_shap(member, D, gvkey, years, rows, base, nets=None, n_steps=16):
    """Attributions over the member's networks, in log-odds as for the tree members: a dict of
    values (rows, K, f), output, reference, plus the row's inputs."""
    x, m, p = member._history(D, gvkey, years, rows)
    parts = [expected_gradients(n, x, m, p, base, n_steps) for n in (member.clf_.nets_ if nets is None else nets)]
    return {'values': np.mean([a[0] for a in parts], axis=0), 'output': np.mean([a[1] for a in parts], axis=0),
            'reference': np.mean([a[2] for a in parts], axis=0), 'x': x, 'm': m, 'p': p, 'per_net': [a[0] for a in parts]}


def _label(feat):
    """Within-year ranks end in _yr. Other inputs keep their own name, so log_at and its rank stay apart."""
    return feat.removesuffix('_yr') + ' (rank)' if feat.endswith('_yr') else feat


def explain_temporal(member, D, gvkey, years, ex, pool, y, member_ranks, risk, out_dir, n_sample=3000, top_n=15, seed=0, log=print):
    """SHAP for the Temporal Transformer on the same rows as explain_all, plus each member's share of the
    final score. D: winsorised frame of all rows; ex: rows explained; pool: its training rows; y: labels of
    ex; member_ranks: {member: within-year rank on ex}; risk: the deployed risk_percentile on ex, which is
    exactly the mean of those ranks."""
    assert np.allclose(np.mean(list(member_ranks.values()), axis=0), risk), 'risk is not the mean of the member ranks'
    idx = _sample(np.random.RandomState(seed), len(ex), n_sample)
    rows = ex[idx]
    feats = member.features
    res = temporal_shap(member, D, gvkey, years, rows, temporal_reference(member, D, gvkey, years, pool, seed=seed))
    sv, base = res['values'], res['reference']
    log(f'  Temporal Transformer: expected gradients on {len(rows)} rows, {sv.shape[1]} years x {sv.shape[2]} features, '
        f'{len(member.clf_.nets_)} networks')
    by_feature, by_year = sv.sum(1), sv.sum(2)
    imp = pd.DataFrame({'feature': feats, 'mean_abs_shap': np.abs(by_feature).mean(0)})
    for j, lab in enumerate(YEAR_LABELS):
        imp[f'mean_abs_shap_{lab}'] = np.abs(sv[:, j]).mean(0)
    imp['meaning'] = imp['feature'].map(describe)
    imp = imp.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(out_dir, 'temporal_shap_features.csv'), index=False)
    size, present = np.abs(by_year).mean(0), res['p'].mean(0)
    years_tab = pd.DataFrame({'year': YEAR_LABELS, 'mean_abs_shap': size, 'share': size / size.sum(), 'rows_with_this_year': present,
                              'mean_abs_shap_when_present': size / present})    # an absent year has exactly 0 attribution
    years_tab.to_csv(os.path.join(out_dir, 'temporal_shap_years.csv'), index=False)

    # the importance table of the other members gains this one's column
    p_imp = os.path.join(out_dir, 'feature_importance_shap.csv')
    other = pd.read_csv(p_imp).drop(columns='mean_abs_shap_TT', errors='ignore')
    other.insert(other.columns.get_loc('meaning'), 'mean_abs_shap_TT', other['feature'].map(imp.set_index('feature')['mean_abs_shap']))
    other.to_csv(p_imp, index=False)

    data = D[feats].values[rows]
    expl = shap.Explanation(values=by_feature, base_values=base, data=data, feature_names=[_label(f) for f in feats])
    plt.figure()
    shap.plots.beeswarm(expl, max_display=top_n, show=False)
    plt.title('Temporal Transformer: what drives its score\n(SHAP summed over 5 years; colour is the latest year)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'temporal_shap_beeswarm.png'), dpi=150)
    plt.close()

    top = imp['feature'].head(top_n).tolist()
    cols = [feats.index(f) for f in top]
    grid = np.abs(sv).mean(0)[:, cols].T
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(YEAR_LABELS)), YEAR_LABELS)
    ax.set_yticks(range(len(top)), [_label(f) for f in top])
    ax.set_xlabel('Fiscal year relative to the year scored')
    ax.set_title('Temporal Transformer: mean |SHAP| by feature and year')
    fig.colorbar(im, label='mean |SHAP| (log-odds)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'temporal_shap_heatmap.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(5.5, 3.5))
    plt.bar(YEAR_LABELS, 100 * years_tab['share'], color='#0B3C5D')
    plt.ylabel('Share of attribution (%)')
    plt.xlabel('Fiscal year relative to the year scored')
    plt.title(f'Temporal Transformer: which years it reads\n(t-4 exists for {100 * present[0]:.0f}% of firm-years, t for {100 * present[-1]:.0f}%)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'temporal_shap_years.png'), dpi=150)
    plt.close()

    # top-flagged table: which members flagged each firm, and what moved this member
    ranks = {m: r[idx] for m, r in member_ranks.items()}
    order = np.argsort(-risk[idx], kind='stable')
    p_top = os.path.join(out_dir, 'top_flagged_explained.csv')
    flagged = pd.read_csv(p_top)
    where = {(int(g), int(t)): i for i, (g, t) in enumerate(zip(gvkey[rows], years[rows]))}
    extra = []
    for g, t in zip(flagged['gvkey'], flagged['fyear']):
        i = where[(g, t)]
        cells = np.argsort(-np.abs(sv[i]), axis=None)[:3]
        words = '; '.join(f'{feats[f]} in {YEAR_LABELS[j]} ({"raises" if sv[i, j, f] > 0 else "lowers"} risk, {sv[i, j, f]:+.2f})'
                          for j, f in (np.unravel_index(c, sv[i].shape) for c in cells))
        extra.append({**{f'rank_{m}': r[i] for m, r in ranks.items()}, 'tt_top_drivers': words})
    flagged = flagged.drop(columns=[c for c in flagged if c.startswith('rank_') or c == 'tt_top_drivers'])
    pd.concat([flagged, pd.DataFrame(extra)], axis=1).to_csv(p_top, index=False)

    # waterfalls: the same firm-years as the tree members' plots
    tp = [r for r in order if y[idx[r]] == 1][:3]
    fp = [r for r in order if y[idx[r]] == 0][:1]
    for tag, picks in (('fraud', tp), ('falsealarm', fp)):
        for k, r in enumerate(picks):
            plt.figure()
            shap.plots.waterfall(expl[int(r)], max_display=10, show=False)
            plt.title(f'Temporal Transformer, {tag}: gvkey {int(gvkey[rows[r]])}, FY{int(years[rows[r]])}\n(SHAP summed over 5 years; values are the latest year)', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'waterfall_tt_{tag}_{k + 1}.png'), dpi=150)
            plt.close()
    log('  Temporal Transformer, share of attribution by year: ' + ', '.join(f'{a} {100 * b:.0f}%' for a, b in zip(YEAR_LABELS, years_tab['share'])))
    return imp, years_tab
