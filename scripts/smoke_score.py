"""Smoke test for score_new_data.py: 20 firms of FY2012 (+ their 2011 rows) must come back scored.

    python scripts/smoke_score.py [--model results/final_models.joblib]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.path.join(ROOT, 'results', 'final_models.joblib'))
ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'data_FraudDetection_JAR2020.csv'))
a = ap.parse_args()

d = pd.read_csv(a.data)
firms = d.loc[d.fyear == 2012, 'gvkey'].drop_duplicates().sample(20, random_state=0)
sub = d[d.gvkey.isin(firms) & d.fyear.between(2011, 2012)].drop(columns=['misstate', 'p_aaer'])
with tempfile.TemporaryDirectory() as tmp:
    inp, outp = os.path.join(tmp, 'in.csv'), os.path.join(tmp, 'out.csv')
    sub.to_csv(inp, index=False)
    subprocess.run([sys.executable, os.path.join(ROOT, 'score_new_data.py'), '--input', inp, '--year', '2012',
                    '--reference', a.data, '--model', a.model, '--out', outp], check=True)
    out = pd.read_csv(outp)
assert len(out) == 20, len(out)
need = {'gvkey', 'fyear', 'raw_score', 'risk_percentile', 'fraud_probability', 'market_size'}
assert need <= set(out.columns), need - set(out.columns)
assert out.fyear.eq(2012).all() and set(out.gvkey) == set(firms)
assert out.risk_percentile.between(0, 1).all() and out.fraud_probability.between(0, 1).all()
print('smoke OK')
print(out.head().to_string(index=False))
