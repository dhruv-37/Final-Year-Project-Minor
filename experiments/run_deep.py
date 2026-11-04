"""Temporal Transformer and Temporal Mamba on the rank_only features (5 x 68 firm histories).

    python experiments/run_deep.py --stage cv [--arch temp_trans]
    python experiments/run_deep.py --stage test --arch temp_trans
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import OUT, SEEDS, ctx, keys, load, mask_serial, os, path, run_tasks, split, winsorized
from deep_models import DeepClassifier, Preprocessor, history_index, stack_history
from models import ensemble_scores
from run_reference import MEMBERS, ref_path
from scoring import cv_report, reference_cv, reference_ensemble, save_candidate

N_TRIALS = 6
EPOCHS = (5, 10, 15, 20)
LONG_EPOCHS = (30, 40, 50, 60)  # only for a trial whose best score is at the last of EPOCHS
CV_MEMBERS, TEST_MEMBERS = 2, 3
HIST_LEN = 5
TUNE_SEED, CHECK_SEEDS = 42, (43, 44)
SPACE = {
    'temp_trans': dict(d=[32, 64], n_layers=[1, 2], dropout=[0.1, 0.2, 0.3]),
    'temp_mamba': dict(d=[32, 64], n_layers=[1, 2], d_state=[8, 16], dropout=[0.1, 0.2, 0.3]),
}


CONFIG_STREAM = {'temp_trans': 3, 'temp_mamba': 4}      # fixed so the frozen configs stay reproducible


def sample_config(arch, trial):
    """Deterministic config number `trial` for this architecture."""
    rng = np.random.RandomState(1000 + 37 * CONFIG_STREAM[arch] + trial)
    kw = {k: v[rng.randint(len(v))] for k, v in SPACE[arch].items()}
    kw = {k: (float(v) if k == 'dropout' else int(v)) for k, v in kw.items()}
    train = dict(lr=float(rng.choice([3e-4, 1e-3, 2e-3])), weight_decay=float(rng.choice([1e-5, 1e-4, 1e-3])),
                 ratio=int(rng.choice([3, 5, 10])))
    return kw, train


def inputs(arch, key):
    """Model inputs for one split: preprocessing is fit on that split's training rows only."""
    c = ctx()
    tr, te = split(key)
    X = winsorized(tr)[c.groups['rank_only']].values
    v, m = Preprocessor().fit(X[tr]).transform(X)
    idx = history_index(c.gvkey, c.years, HIST_LEN)
    return stack_history(v, m, idx[tr]), stack_history(v, m, idx[te]), tr, te


def cv_task(arch, trial, key, seed, n_jobs, grid=EPOCHS):
    c = ctx()
    kw, train = sample_config(arch, trial)
    Itr, Ite, tr, te = inputs(arch, key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    return DeepClassifier(arch, kw, n_members=CV_MEMBERS, epochs=max(grid), seed=seed, **train).fit_predict(Itr, ytr, Ite, eval_epochs=grid)


def final_task(arch, trial, epochs, key, seed, n_members, n_jobs):
    c = ctx()
    kw, train = sample_config(arch, trial)
    Itr, Ite, tr, te = inputs(arch, key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    return DeepClassifier(arch, kw, n_members=n_members, epochs=epochs, seed=seed, **train).fit_predict(Itr, ytr, Ite)


def cv_path(arch, trial, key, seed):
    return path('deep', f'cv_{arch}_t{trial}_{key}_s{seed}.pkl')


def long_path(arch, trial, key, seed):
    return path('deep', f'long_{arch}_t{trial}_{key}_s{seed}.pkl')


def final_path(arch, key, seed):
    return path('deep', f'final_{arch}_{key}_s{seed}.pkl')


def grid_rows(trial, by_key, grid):
    c = ctx()
    return [{'trial': trial, 'epochs': ep,
             'cv_auc': float(np.mean([roc_auc_score(c.y[split(k)[1]], by_key[k][ep]) for k in keys('cv')]))} for ep in grid]


def stage_cv(archs):
    run_tasks([(cv_task, cv_path(a, t, k, TUNE_SEED), dict(arch=a, trial=t, key=k, seed=TUNE_SEED))
               for a in archs for t in range(N_TRIALS) for k in keys('cv')], n_workers=5, threads=3)
    res_path = os.path.join(OUT, 'cv_deep.json')
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    best = {}
    for a in archs:
        rows = [r for t in range(N_TRIALS) for r in grid_rows(t, {k: load(cv_path(a, t, k, TUNE_SEED)) for k in keys('cv')}, EPOCHS)]
        best[a] = max(rows, key=lambda r: r['cv_auc'])
    # a best score at the last epoch of the grid means under-training, not a real optimum
    at_edge = {a: best[a]['trial'] for a in archs if best[a]['epochs'] == EPOCHS[-1]}
    print('extending to', LONG_EPOCHS, at_edge, flush=True)
    run_tasks([(cv_task, long_path(a, t, k, TUNE_SEED), dict(arch=a, trial=t, key=k, seed=TUNE_SEED, grid=LONG_EPOCHS))
               for a, t in at_edge.items() for k in keys('cv')], n_workers=5, threads=3)
    frozen = {}
    for a in archs:
        long_rows = grid_rows(at_edge[a], {k: load(long_path(a, at_edge[a], k, TUNE_SEED)) for k in keys('cv')}, LONG_EPOCHS) if a in at_edge else []
        frozen[a] = max([best[a]] + long_rows, key=lambda r: r['cv_auc'])
        print(a, 'tuning best', frozen[a], sample_config(a, frozen[a]['trial']), long_rows, flush=True)
    # fresh seeds for the gate: the tuning score above is optimistic (best of the grid)
    run_tasks([(final_task, final_path(a, k, s),
                dict(arch=a, trial=frozen[a]['trial'], epochs=frozen[a]['epochs'], key=k, seed=s, n_members=CV_MEMBERS))
               for a in archs for k in keys('cv') for s in CHECK_SEEDS], n_workers=5, threads=3)
    ref = reference_cv(CHECK_SEEDS)
    print('reference', ref, flush=True)
    for a in archs:
        alone = cv_report({s: {k: load(final_path(a, k, s)) for k in keys('cv')} for s in CHECK_SEEDS})
        ens = cv_report({s: {k: ensemble_scores([load(ref_path(k, m, s)) for m in MEMBERS] + [load(final_path(a, k, s))])
                             for k in keys('cv')} for s in CHECK_SEEDS})
        results[a] = {'frozen': frozen[a], 'config': sample_config(a, frozen[a]['trial']), 'alone': alone, 'with_reference': ens}
        print(a, 'alone', alone['cv_auc'], 'with reference', ens['cv_auc'], 'gate', ens['cv_auc'] - ref['cv_auc'] >= 0.002, flush=True)
    results['reference'] = ref
    json.dump(results, open(res_path, 'w'), indent=2)


def stage_test(archs):
    results = json.load(open(os.path.join(OUT, 'cv_deep.json')))
    run_tasks([(final_task, final_path(a, k, s),
                dict(arch=a, trial=results[a]['frozen']['trial'], epochs=results[a]['frozen']['epochs'], key=k, seed=s,
                     n_members=TEST_MEMBERS))
               for a in archs for k in keys('test') for s in SEEDS], n_workers=5, threads=3)
    for a in archs:
        for s in SEEDS:
            save_candidate(f'{a}_alone', s, {k: load(final_path(a, k, s)) for k in keys('test')})
            save_candidate(f'reference+{a}', s, {k: ensemble_scores([load(ref_path(k, m, s)) for m in MEMBERS]
                                                                     + [load(final_path(a, k, s))]) for k in keys('test')})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['cv', 'test'], required=True)
    ap.add_argument('--arch', nargs='+', default=list(SPACE))
    a = ap.parse_args()
    ctx()
    (stage_cv if a.stage == 'cv' else stage_test)(a.arch)


if __name__ == '__main__':
    main()
