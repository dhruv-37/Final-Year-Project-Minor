"""Candidates of PROTOCOL_AUC.md against the current ensemble (LGBM + XGB + LR + Temporal Transformer).

    python experiments/run_auc.py --stage screen          # all settings, seed 42, validation years
    python experiments/run_auc.py --stage check           # best of each family, fresh seeds, the gate
    python experiments/run_auc.py --stage check --cand A1_mkt+A5_rf      # a combination of passing settings
    python experiments/run_auc.py --stage test --cand A5_rf ...
"""
import argparse
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from common import N_BAGS, OUT, SEEDS, ctx, keys, load, mask_serial, os, path, pickle, run_tasks, split, winsorized
from auc_features import (ModulatedTT, PairwiseClassifier, WeightedBagger, accrual_features, market_features,
                          peer_features, peer_labels, surprise_features)
from deep_models import TEMPORAL_MEMBER, ARCH, DeepClassifier, Preprocessor, history_index, stack_history
from models import BalancedBagger, LogReg, ensemble_scores, lgbm_factory, sample_lgbm, sample_xgb, xgb_factory
from run_deep import CV_MEMBERS, TEST_MEMBERS, cv_path, final_path
from run_reference import MEMBERS, ref_path
from scoring import cv_report

SCREEN_SEED, CHECK_SEEDS = 42, (43, 44)
CV_GATE = 0.002
TT_GRID = (5, 10, 15, 20, 30)
N_RETUNE, TUNE_BAGS = 40, 3
SETTINGS = ['A1_mkt', 'A2_peer', 'A3_acc', 'A4_3', 'A4_6', 'A5_rf', 'A5_et', 'A5_both', 'A6_six', 'A7_tt2', 'A7_tt3', 'A7_nolr',
            'A7_tt2nolr', 'A7_mamba', 'A8_long', 'A9_mod', 'A10_surp', 'A11_L', 'A11_X', 'A11_both', 'A12_pair', 'A13_5',
            'A13_10']
ARCH.setdefault('mod_tt', ModulatedTT)


def parts(setting):
    return setting.split('+')


def spec(setting):
    """What a setting (or a '+' combination) changes relative to the reference."""
    s = dict(pool=[], weight=None, retune=[], add=[], tt=None, blend=None)
    for p in parts(setting):
        fam, arg = p.split('_')
        if fam in ('A1', 'A2', 'A3', 'A10'):
            s['pool'].append(fam)
        elif fam == 'A4':
            s['weight'] = ('recency', float(arg))
        elif fam == 'A13':
            s['weight'] = ('shift', float(arg))
        elif fam == 'A5':
            s['add'] += ['RF', 'ET'] if arg == 'both' else [arg.upper()]
        elif fam in ('A6', 'A8', 'A9', 'A12'):
            s['tt'] = arg
        elif fam == 'A7':
            s['blend'] = arg
        elif fam == 'A11':
            s['retune'] += ['LGBM', 'XGB'] if arg == 'both' else [{'L': 'LGBM', 'X': 'XGB'}[arg]]
    return s


def window(key):
    c = ctx()
    tr, te = split(key)
    return c, tr, te, mask_serial(c.y, c.paaer, tr, te)


def static_features():
    """Market, peer and accrual features: same-year or earlier rows only, so fit once for the panel."""
    p = path('auc', 'static.pkl')
    if not os.path.exists(p):
        c = ctx()
        labels = peer_labels(c.DW, c.years)
        blob = {'A1': market_features(c.DW, c.gvkey, c.years), 'A2': peer_features(c.DW, c.years, labels),
                'A3': accrual_features(c.DW, c.gvkey, c.years, labels)}
        with open(p, 'wb') as fh:
            pickle.dump(blob, fh)
    return load(p)


def frame(s, key, tr, ytr):
    """Window's winsorised features plus the pools the setting adds, and the feature names to use."""
    c = ctx()
    D = winsorized(tr)
    feats = list(c.feats)
    if s['pool']:
        st = static_features()
        extra = [surprise_features(D, c.gvkey, c.years, tr, ytr, c.groups['rank_only']) if fam == 'A10' else st[fam] for fam in s['pool']]
        for e in extra:
            e.index = D.index
        D = pd.concat([D] + extra, axis=1)
        feats += [col for e in extra for col in e.columns]
    return D, feats


def row_weights(s, D, feats, tr, te, seed, n_jobs):
    kind, val = s['weight']
    c = ctx()
    if kind == 'recency':
        return 0.5 ** ((c.years[tr].max() - c.years[tr]) / val)
    rng = np.random.RandomState(seed)
    both = np.r_[rng.choice(tr, min(30000, len(tr)), replace=False), te]
    label = np.r_[np.zeros(len(both) - len(te)), np.ones(len(te))]
    m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                       random_state=seed, n_jobs=n_jobs, verbose=-1).fit(D[feats].values[both], label)
    q = np.clip(m.predict_proba(D[feats].values[tr])[:, 1], 1e-3, 1 - 1e-3)
    w = q / (1 - q)
    return np.clip(w / w.mean(), 1 / val, val)


def retuned(kind):
    return load(path('auc', 'retune_best.pkl'))[kind]


def member_task(setting, key, member, seed, n_jobs):
    c, tr, te, ytr = window(key)
    s = spec(setting)
    D, feats = frame(s, key, tr, ytr)
    X = D[feats].values
    if member == 'LR':
        return LogReg().fit(X[tr], ytr).predict_proba(X[te])
    if member in ('RF', 'ET'):
        cls = RandomForestClassifier if member == 'RF' else ExtraTreesClassifier
        kw = dict(bootstrap=True) if member == 'ET' else {}
        m = cls(n_estimators=500, min_samples_leaf=5, max_samples=0.3, class_weight='balanced_subsample', n_jobs=n_jobs, random_state=seed, **kw)
        return m.fit(X[tr], ytr).predict_proba(X[te])[:, 1]
    p = retuned(member) if member in s['retune'] else (c.bestL if member == 'LGBM' else c.bestX)
    make, sd = (lgbm_factory(p, n_jobs), seed) if member == 'LGBM' else (xgb_factory(p, n_jobs), seed + 1)
    if s['weight']:
        return WeightedBagger(make, N_BAGS, p['ratio'], sd).fit(X[tr], ytr, row_weights(s, D, feats, tr, te, seed, n_jobs)).predict_proba(X[te])
    return BalancedBagger(make, N_BAGS, p['ratio'], sd).fit(X[tr], ytr).predict_proba(X[te])


def tt_task(kind, key, seed, n_members, grid, n_jobs):
    c, tr, te, ytr = window(key)
    X = winsorized(tr)[c.groups['rank_only']].values
    v, m = Preprocessor().fit(X[tr]).transform(X)
    k = 8 if kind == 'long' else 5
    idx = history_index(c.gvkey, c.years, k)
    kw = {a: b for a, b in TEMPORAL_MEMBER.items() if a not in ('arch', 'arch_kw', 'n_members', 'epochs')}
    ak = dict(TEMPORAL_MEMBER['arch_kw'], seq_len=k)
    arch, cls = {'mod': ('mod_tt', DeepClassifier), 'pair': ('temp_trans', PairwiseClassifier)}.get(kind, ('temp_trans', DeepClassifier))
    inp = [stack_history(v, m, idx[tr]), stack_history(v, m, idx[te])]
    if kind == 'mod':
        steps = np.arange(-k + 1, 1, dtype=np.float32)
        inp = [a + (((c.years[r][:, None] + steps - 2000) / 10).astype(np.float32),) for a, r in zip(inp, (tr, te))]
    return cls(arch, ak, n_members=n_members, epochs=max(grid), seed=seed, **kw).fit_predict(inp[0], ytr, inp[1], eval_epochs=grid)


def retune_task(member, trial, key, n_jobs):
    c, tr, te, ytr = window(key)
    X = winsorized(tr)[c.feats].values
    p = trial_params(member, trial)
    make, sd = (lgbm_factory(p, n_jobs), SCREEN_SEED) if member == 'LGBM' else (xgb_factory(p, n_jobs), SCREEN_SEED + 1)
    return BalancedBagger(make, TUNE_BAGS, p['ratio'], sd).fit(X[tr], ytr).predict_proba(X[te])


def trial_params(member, trial):
    rng = np.random.RandomState({'LGBM': 2000, 'XGB': 2001}[member])
    sampler = sample_lgbm if member == 'LGBM' else sample_xgb
    return [sampler(rng) for _ in range(trial + 1)][trial]


def retune_path(member, trial, key):
    return path('auc', 'retune', f'{member}_t{trial}_{key}.pkl')


def member_path(setting, key, member, seed):
    return path('auc', 'member', f'{setting}_{key}_{member}_s{seed}.pkl')


def tt_path(kind, key, seed):
    return path('auc', 'member', f'tt_{kind}_{key}_s{seed}.pkl')


def tt_score(key, seed):
    return load(cv_path('temp_trans', 0, key, seed))[15] if key.startswith('val') and seed == SCREEN_SEED else load(final_path('temp_trans', key, seed))


def mamba_score(key, seed):
    return load(cv_path('temp_mamba', 1, key, seed))[10] if key.startswith('val') and seed == SCREEN_SEED else load(final_path('temp_mamba', key, seed))


def changed_members(s):
    if s['pool']:
        return list(MEMBERS)
    tree = ['LGBM', 'XGB'] if s['weight'] else list(s['retune'])
    return tree


def reference(key, seed):
    return ensemble_scores([load(ref_path(key, m, seed)) for m in MEMBERS] + [tt_score(key, seed)])


def score(setting, key, seed, epoch=None):
    s = spec(setting)
    changed = changed_members(s)
    scores = {m: load(member_path(setting, key, m, seed) if m in changed else ref_path(key, m, seed)) for m in MEMBERS}
    for m in s['add']:
        scores[m] = load(member_path(setting, key, m, seed))
    scores['TT'] = load(tt_path(s['tt'], key, seed))[epoch] if s['tt'] else tt_score(key, seed)
    blend = s['blend']
    if blend in ('nolr', 'tt2nolr'):
        scores.pop('LR')
    if blend == 'mamba':
        scores['MAMBA'] = mamba_score(key, seed)
    w = [{'tt2': 2, 'tt3': 3, 'tt2nolr': 2}.get(blend, 1) if m == 'TT' else 1 for m in scores]
    return ensemble_scores(list(scores.values()), w)


def tasks_for(settings, stage_keys, seeds, members=CV_MEMBERS, grid=TT_GRID):
    fit = []
    for st in settings:
        s = spec(st)
        for k in stage_keys:
            for sd in seeds:
                fit += [(member_task, member_path(st, k, m, sd), dict(setting=st, key=k, member=m, seed=sd))
                        for m in changed_members(s) + s['add']]
                if s['tt']:
                    n = 6 if s['tt'] == 'six' else members
                    g = (15,) if s['tt'] == 'six' else grid
                    fit.append((tt_task, tt_path(s['tt'], k, sd), dict(kind=s['tt'], key=k, seed=sd, n_members=n, grid=g)))
    return fit


def best_tt_epoch(setting):
    """Epoch of a Temporal Transformer variant, from the ensemble's score on the screening seed."""
    s = spec(setting)
    if not s['tt']:
        return None
    if s['tt'] == 'six':
        return 15
    c = ctx()
    aucs = {ep: np.mean([roc_auc_score(c.y[split(k)[1]], score(setting, k, SCREEN_SEED, ep)) for k in keys('cv')]) for ep in TT_GRID}
    return max(aucs, key=aucs.get)


def pick_retune():
    """Best trial per tree model: the one that most helps the full ensemble on the validation years."""
    c = ctx()
    run_tasks([(retune_task, retune_path(m, t, k), dict(member=m, trial=t, key=k))
               for m in ('LGBM', 'XGB') for t in range(N_RETUNE) for k in keys('cv')], n_workers=5, threads=3)
    best, table = {}, {}
    for m in ('LGBM', 'XGB'):
        aucs = []
        for t in range(N_RETUNE):
            per = []
            for k in keys('cv'):
                others = [load(ref_path(k, o, SCREEN_SEED)) for o in MEMBERS if o != m] + [tt_score(k, SCREEN_SEED)]
                per.append(roc_auc_score(c.y[split(k)[1]], ensemble_scores(others + [load(retune_path(m, t, k))])))
            aucs.append(float(np.mean(per)))
        t = int(np.argmax(aucs))
        best[m] = trial_params(m, t)
        table[m] = {'best_trial': t, 'auc': aucs[t], 'params': best[m]}
    with open(path('auc', 'retune_best.pkl'), 'wb') as fh:
        pickle.dump(best, fh)
    json.dump(table, open(os.path.join(OUT, 'auc_retune.json'), 'w'), indent=2)
    print('retune', table, flush=True)


def stage_screen():
    static_features()
    pick_retune()
    run_tasks(tasks_for(SETTINGS, keys('cv'), (SCREEN_SEED,)), n_workers=5, threads=3)
    ref = cv_report({SCREEN_SEED: {k: reference(k, SCREEN_SEED) for k in keys('cv')}})
    rows = []
    for st in SETTINGS:
        ep = best_tt_epoch(st)
        r = cv_report({SCREEN_SEED: {k: score(st, k, SCREEN_SEED, ep) for k in keys('cv')}})
        rows.append({'setting': st, 'family': st.split('_')[0], 'epoch': ep, 'gain': r['cv_auc'] - ref['cv_auc'], **r})
    json.dump({'reference': ref, 'candidates': rows}, open(os.path.join(OUT, 'cv_auc_screen.json'), 'w'), indent=2)
    print('reference', ref, flush=True)
    for r in sorted(rows, key=lambda r: -r['gain']):
        print(r['setting'], 'gain %+.4f' % r['gain'], 'AUC %.4f' % r['cv_auc'], 'ep', r['epoch'], flush=True)


def check(settings, epochs, screen_gain):
    run_tasks(tasks_for(settings, keys('cv'), CHECK_SEEDS), n_workers=5, threads=3)
    ref = cv_report({s: {k: reference(k, s) for k in keys('cv')} for s in CHECK_SEEDS})
    rows = []
    for st in settings:
        r = cv_report({s: {k: score(st, k, s, epochs[st]) for k in keys('cv')} for s in CHECK_SEEDS})
        rows.append({'setting': st, 'epoch': epochs[st], 'screen_gain': screen_gain[st], 'gain': r['cv_auc'] - ref['cv_auc'],
                     'gate': r['cv_auc'] - ref['cv_auc'] >= CV_GATE, **r})
    print('reference', ref, flush=True)
    for r in sorted(rows, key=lambda r: -r['gain']):
        print(r['setting'], 'screen', r['screen_gain'], 'fresh %+.4f' % r['gain'], 'gate', r['gate'], r['fold_auc'], flush=True)
    return ref, rows


def stage_check(cands):
    res_path = os.path.join(OUT, 'cv_auc.json')
    if cands:           # a combination of passing settings: still on the validation years only
        prior = json.load(open(res_path))
        epochs = {st: max([r['epoch'] or 0 for r in prior['candidates'] if r['setting'] in parts(st)], default=0) or None for st in cands}
        _, rows = check(cands, epochs, {st: None for st in cands})
        prior['candidates'] = [r for r in prior['candidates'] if r['setting'] not in cands] + rows
    else:
        scr = json.load(open(os.path.join(OUT, 'cv_auc_screen.json')))['candidates']
        best = {}
        for r in scr:
            if r['family'] not in best or r['gain'] > best[r['family']]['gain']:
                best[r['family']] = r
        settings = [r['setting'] for r in best.values()]
        ref, rows = check(settings, {r['setting']: r['epoch'] for r in best.values()}, {r['setting']: r['gain'] for r in best.values()})
        prior = {'reference': ref, 'candidates': rows}
    ok = sorted([r for r in prior['candidates'] if r['gate']], key=lambda r: -r['gain'])
    prior['frozen'] = [r['setting'] for r in ok][:3]
    json.dump(prior, open(res_path, 'w'), indent=2)
    print('frozen for test years:', prior['frozen'], flush=True)


def stage_test(cands):
    res = json.load(open(os.path.join(OUT, 'cv_auc.json')))
    assert set(cands) <= set(res['frozen']), f'only frozen candidates may see the test years: {res["frozen"]}'
    epochs = {r['setting']: r['epoch'] for r in res['candidates']}
    for st in cands:
        ep = epochs[st]
        run_tasks(tasks_for([st], keys('test'), SEEDS, members=TEST_MEMBERS, grid=(ep,) if ep else TT_GRID), n_workers=5, threads=3)
        for s in SEEDS:
            with open(path('auc', 'candidates', f'{st}_s{s}.pkl'), 'wb') as fh:
                pickle.dump({k: score(st, k, s, ep) for k in keys('test')}, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['screen', 'check', 'test'], required=True)
    ap.add_argument('--cand', nargs='*', default=[])
    a = ap.parse_args()
    ctx()
    if a.stage == 'screen':
        stage_screen()
    elif a.stage == 'check':
        stage_check(a.cand)
    else:
        stage_test(a.cand)


if __name__ == '__main__':
    main()
