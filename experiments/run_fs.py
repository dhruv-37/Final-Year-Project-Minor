"""Dynamic feature selection candidates of PROTOCOL_FS.md, against the current ensemble (trees + LR + TT).

    python experiments/run_fs.py --stage screen            # all settings, seed 42, validation years
    python experiments/run_fs.py --stage check             # best of each family, fresh seeds, the gate
    python experiments/run_fs.py --stage test --cand F9_0.02_one_se ...
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import (N_BAGS, OUT, SEEDS, ctx, fit_member, keys, load, mask_serial, os, path, pickle, run_tasks, split,
                    winsorized)
from deep_models import DeepClassifier, Preprocessor, TEMPORAL_MEMBER, history_index, stack_history
from dynamic_selection import (SoftBagger, adversarial_drift, history_features, mrmr_order, oot_importance,
                               permutation_drop, shadow_frequency, stability_frequency)
from feature_selection import DynamicRFE, choose_subset
from models import BalancedBagger, ensemble_scores, lgbm_factory, xgb_factory
from run_deep import HIST_LEN, TEST_MEMBERS, CV_MEMBERS, cv_path, final_path
from run_drfe import p_path, path_task
from run_reference import MEMBERS, ref_path
from scoring import cv_report

SCREEN_SEED, CHECK_SEEDS = 42, (43, 44)
CV_GATE = 0.002
GATE_EPOCHS = (5, 10, 15, 20)
HIST_BASES = [b + '_yr' for b in ('soft_assets', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'ch_cs', 'ch_roa', 'bm', 'reoa',
                                  'EBIT', 'roa', 'margin', 'asset_turn', 'lev', 'cash_ta', 'rect_sale', 'inv_cogs',
                                  'mve_ta', 'fin_need', 'tata')]
TAUS = (0.02, 0.05, 0.10)
RULES = ('best', 'one_se')
SETTINGS = ([f'F1_{x}' for x in (0.6, 0.8, 0.95)] + [f'F2_{x}' for x in (1.0, 0.95, 0.75)] + ['F3_0', 'F3_se']
            + [f'F4_{x}' for x in (0.1, 0.2, 0.3)] + [f'F5_{x}' for x in (40, 70, 100)]
            + [f'F6_{t}_{r}' for t in TAUS for r in RULES] + [f'F7_{x}' for x in (0.5, 1.0)]
            + [f'F8_{t}' for t in TAUS] + ['F9_none'] + [f'F9_{t}_{r}' for t in (0.02, 0.05) for r in RULES]
            + [f'F10_{g}_{n}' for g in ('softmax', 'sigmoid') for n in (68, 133)])


def history_frame():
    p = path('fs', 'history.pkl')
    if os.path.exists(p):
        return load(p)
    c = ctx()
    h = history_features(c.DW, c.gvkey, c.years, HIST_BASES)
    with open(p, 'wb') as fh:
        pickle.dump(h, fh)
    return h


def window(key):
    c = ctx()
    tr, te = split(key)
    return c, tr, te, mask_serial(c.y, c.paaer, tr, te)


# ---------------------------------------------------------------- what each selector stores per window
def selector_task(sel, key, seed, n_jobs):
    c, tr, te, ytr = window(key)
    X = winsorized(tr)[c.feats].values[tr]
    yrs, pa = c.years[tr], c.paaer[tr]
    if sel == 'stab':
        return stability_frequency(X, ytr, yrs, pa, c.gvkey[tr], c.bestL, seed, n_jobs)
    if sel == 'shadow':
        return shadow_frequency(X, ytr, yrs, pa, c.bestL, seed, n_jobs)
    if sel == 'perm':
        return permutation_drop(X, ytr, yrs, pa, c.bestL, seed, n_jobs)
    if sel == 'adv':
        return adversarial_drift(X, yrs, seed, n_jobs)
    imp = oot_importance(X, ytr, yrs, pa, c.bestL, seed, n_jobs)
    return {'imp': imp, 'order': mrmr_order(X, imp, seed)}


def pool_task(key, tau, seed, n_jobs):
    c, tr, te, ytr = window(key)
    h = history_frame()
    X = np.hstack([winsorized(tr)[c.feats].values, h.values])[tr]
    return DynamicRFE(c.bestL, tau=tau, seed=seed, n_jobs=n_jobs).fit(X, ytr, c.years[tr], c.paaer[tr], c.feats + list(h.columns)).path_


def sel_path(sel, key, seed):
    return path('fs', 'select', f'{sel}_{key}_s{seed}.pkl')


def pool_path(key, tau, seed):
    return path('fs', 'select', f'pool_{key}_tau{tau}_s{seed}.pkl')


def needed_selectors(setting):
    """Selector results a setting depends on: (fn, out path, kwargs) builders, by key and seed."""
    fam, *arg = setting.split('_')
    if fam == 'F1':
        return ['stab']
    if fam == 'F2':
        return ['shadow']
    if fam == 'F3':
        return ['perm']
    if fam == 'F4':
        return ['adv']
    if fam in ('F5', 'F7'):
        return ['oot']
    return []


def selector_tasks(setting, key, seed):
    fam, *arg = setting.split('_')
    tasks = [(selector_task, sel_path(s, key, seed), dict(sel=s, key=key, seed=seed)) for s in needed_selectors(setting)]
    if fam in ('F6', 'F8'):
        tau = float(arg[0])
        tasks.append((path_task, p_path(key, tau, seed), dict(key=key, tau=tau, seed=seed)))
    if fam == 'F9' and arg[0] != 'none':
        tau = float(arg[0])
        tasks.append((pool_task, pool_path(key, tau, seed), dict(key=key, tau=tau, seed=seed)))
    return tasks


def subset(setting, key, seed):
    """Feature names a hard-selection setting keeps in this window."""
    c = ctx()
    fam, *arg = setting.split('_')
    names = np.array(c.feats)
    if fam == 'F1':
        return list(names[load(sel_path('stab', key, seed)) >= float(arg[0])])
    if fam == 'F2':
        return list(names[load(sel_path('shadow', key, seed))[float(arg[0])] >= 0.5])
    if fam == 'F3':
        drop, se = load(sel_path('perm', key, seed))
        return list(names[drop > (se if arg[0] == 'se' else 0)])
    if fam == 'F4':
        share = load(sel_path('adv', key, seed))
        return list(names[np.argsort(share)[:len(names) - int(float(arg[0]) * len(names))]])
    if fam == 'F5':
        return list(names[load(sel_path('oot', key, seed))['order'][:int(arg[0])]])
    if fam == 'F6':
        return choose_subset(load(p_path(key, float(arg[0]), seed)), arg[1])
    if fam == 'F8':
        return choose_subset(load(p_path(key, float(arg[0]), seed)), 'one_se')
    if arg[0] == 'none':
        return c.feats + list(history_frame().columns)
    return choose_subset(load(pool_path(key, float(arg[0]), seed)), arg[1])


# ---------------------------------------------------------------- fitting one changed member
def member_task(setting, key, member, seed, n_jobs):
    c, tr, te, ytr = window(key)
    fam = setting.split('_')[0]
    D = winsorized(tr)
    if fam == 'F9':
        D = pd.concat([D, history_frame()], axis=1)
    if fam == 'F7':
        alpha = float(setting.split('_')[1])
        w, X = load(sel_path('oot', key, seed))['imp'], D[c.feats].values
        make, p, s = ((lgbm_factory(c.bestL, n_jobs), c.bestL, seed) if member == 'LGBM'
                      else (xgb_factory(c.bestX, n_jobs), c.bestX, seed + 1))
        return SoftBagger(make, w, alpha, n_bags=N_BAGS, ratio=p['ratio'], seed=s).fit(X[tr], ytr).predict_proba(X[te])
    return fit_member(member, D, subset(setting, key, seed), tr, te, ytr, seed, N_BAGS, n_jobs)


def gated_task(setting, key, seed, n_members, grid, n_jobs):
    c, tr, te, ytr = window(key)
    _, gate, n = setting.split('_')
    feats = c.groups['rank_only'] if n == '68' else c.feats
    X = winsorized(tr)[feats].values
    v, m = Preprocessor().fit(X[tr]).transform(X)
    idx = history_index(c.gvkey, c.years, HIST_LEN)
    kw = {k: x for k, x in TEMPORAL_MEMBER.items() if k not in ('arch', 'arch_kw', 'n_members', 'epochs')}
    clf = DeepClassifier('gated_tt', dict(TEMPORAL_MEMBER['arch_kw'], gate=gate), n_members=n_members, epochs=max(grid), seed=seed, **kw)
    return clf.fit_predict(stack_history(v, m, idx[tr]), ytr, stack_history(v, m, idx[te]), eval_epochs=grid)


CHANGED = {'F6': ('LR',), 'F7': ('LGBM', 'XGB'), 'F8': ('LGBM',)}


def changed_members(setting):
    return CHANGED.get(setting.split('_')[0], MEMBERS)


def member_path(setting, key, member, seed):
    return path('fs', 'member', f'{setting}_{key}_{member}_s{seed}.pkl')


def gated_path(setting, key, seed):
    return path('fs', 'member', f'{setting}_{key}_s{seed}.pkl')


def tt_score(key, seed):
    return load(cv_path('temp_trans', 0, key, seed))[15] if key.startswith('val') and seed == SCREEN_SEED else load(final_path('temp_trans', key, seed))


def reference(key, seed):
    return ensemble_scores([load(ref_path(key, m, seed)) for m in MEMBERS] + [tt_score(key, seed)])


def score(setting, key, seed, epoch=None):
    """Rank-averaged ensemble of a candidate; epoch is the frozen epoch of an F10 candidate."""
    fam = setting.split('_')[0]
    trees = {m: ref_path(key, m, seed) for m in MEMBERS}
    if fam == 'F10':
        return ensemble_scores([load(p) for p in trees.values()] + [load(gated_path(setting, key, seed))[epoch]])
    if fam == 'F8':
        return ensemble_scores([load(p) for p in trees.values()] + [load(member_path(setting, key, 'LGBM', seed)), tt_score(key, seed)])
    trees.update({m: member_path(setting, key, m, seed) for m in changed_members(setting)})
    return ensemble_scores([load(p) for p in trees.values()] + [tt_score(key, seed)])


def run(settings, stage_keys, seeds, members=CV_MEMBERS, grid=GATE_EPOCHS):
    """Selectors first (each once, however many settings share it), then the changed members."""
    sel, fit = {}, []
    for st in settings:
        for k in stage_keys:
            for s in seeds:
                if st.startswith('F10'):
                    fit.append((gated_task, gated_path(st, k, s), dict(setting=st, key=k, seed=s, n_members=members, grid=grid)))
                    continue
                for t in selector_tasks(st, k, s):
                    sel[t[1]] = t
                fit += [(member_task, member_path(st, k, m, s), dict(setting=st, key=k, member=m, seed=s))
                        for m in changed_members(st)]
    run_tasks(list(sel.values()), n_workers=5, threads=3)
    run_tasks(fit, n_workers=5, threads=3)


def report(setting, seeds, epoch=None):
    return cv_report({s: {k: score(setting, k, s, epoch) for k in keys('cv')} for s in seeds})


def n_features(setting, seeds):
    if setting.split('_')[0] in ('F7', 'F10'):
        return None
    return float(np.mean([len(subset(setting, k, s)) for k in keys('cv') for s in seeds]))


def best_epoch(setting):
    c = ctx()
    by = {k: load(gated_path(setting, k, SCREEN_SEED)) for k in keys('cv')}
    aucs = {ep: np.mean([roc_auc_score(c.y[split(k)[1]], ensemble_scores([load(ref_path(k, m, SCREEN_SEED)) for m in MEMBERS] + [by[k][ep]]))
                         for k in keys('cv')]) for ep in GATE_EPOCHS}
    return max(aucs, key=aucs.get)


def stage_screen():
    run(SETTINGS, keys('cv'), (SCREEN_SEED,))
    ref = cv_report({SCREEN_SEED: {k: reference(k, SCREEN_SEED) for k in keys('cv')}})
    rows = []
    for st in SETTINGS:
        ep = best_epoch(st) if st.startswith('F10') else None
        r = report(st, (SCREEN_SEED,), ep)
        rows.append({'setting': st, 'family': st.split('_')[0], 'epoch': ep, 'n_features': n_features(st, (SCREEN_SEED,)),
                     'gain': r['cv_auc'] - ref['cv_auc'], **r})
    json.dump({'reference': ref, 'candidates': rows}, open(os.path.join(OUT, 'cv_fs_screen.json'), 'w'), indent=2)
    print('reference', ref, flush=True)
    for r in sorted(rows, key=lambda r: -r['gain']):
        print(r['setting'], 'gain %+.4f' % r['gain'], 'AUC %.4f' % r['cv_auc'], 'n', r['n_features'], 'ep', r['epoch'], flush=True)


def stage_check():
    scr = json.load(open(os.path.join(OUT, 'cv_fs_screen.json')))['candidates']
    best = {}
    for r in scr:
        if r['family'] not in best or r['gain'] > best[r['family']]['gain']:
            best[r['family']] = r
    settings = [r['setting'] for r in best.values()]
    epochs = {r['setting']: r['epoch'] for r in best.values()}
    grid_only = [s for s in settings if s.startswith('F10')]
    run([s for s in settings if s not in grid_only], keys('cv'), CHECK_SEEDS)
    run(grid_only, keys('cv'), CHECK_SEEDS)
    ref = cv_report({s: {k: reference(k, s) for k in keys('cv')} for s in CHECK_SEEDS})
    rows = []
    for st in settings:
        r = report(st, CHECK_SEEDS, epochs[st])
        rows.append({'setting': st, 'epoch': epochs[st], 'screen_gain': best[st.split('_')[0]]['gain'],
                     'gain': r['cv_auc'] - ref['cv_auc'], 'gate': r['cv_auc'] - ref['cv_auc'] >= CV_GATE, **r})
    rows.sort(key=lambda r: -r['gain'])
    frozen = [r['setting'] for r in rows if r['gate']][:3]
    json.dump({'reference': ref, 'candidates': rows, 'frozen': frozen}, open(os.path.join(OUT, 'cv_fs.json'), 'w'), indent=2)
    print('reference', ref, flush=True)
    for r in rows:
        print(r['setting'], 'screen %+.4f' % r['screen_gain'], 'fresh %+.4f' % r['gain'], 'gate', r['gate'], r['fold_auc'], flush=True)
    print('frozen for test years:', frozen, flush=True)


def stage_test(cands):
    res = json.load(open(os.path.join(OUT, 'cv_fs.json')))
    assert set(cands) <= set(res['frozen']), f'only frozen candidates may see the test years: {res["frozen"]}'
    epochs = {r['setting']: r['epoch'] for r in res['candidates']}
    for st in cands:
        grid = (epochs[st],) if st.startswith('F10') else GATE_EPOCHS
        run([st], keys('test'), SEEDS, members=TEST_MEMBERS, grid=grid)
        for s in SEEDS:
            with open(path('fs', 'candidates', f'{st}_s{s}.pkl'), 'wb') as fh:
                pickle.dump({k: score(st, k, s, epochs[st]) for k in keys('test')}, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['screen', 'check', 'test'], required=True)
    ap.add_argument('--cand', nargs='*', default=[])
    a = ap.parse_args()
    ctx()
    if a.stage == 'screen':
        stage_screen()
    elif a.stage == 'check':
        stage_check()
    else:
        stage_test(a.cand)


if __name__ == '__main__':
    main()
