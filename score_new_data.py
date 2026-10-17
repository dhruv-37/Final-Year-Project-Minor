"""Score new firm-years with the saved model (results/final_models.joblib).

Input CSV needs: gvkey, fyear and the 28 raw Compustat columns + the 14 paper ratios
(same layout as data_FraudDetection_JAR2020.csv). Include the PRIOR year's rows for each firm
so growth/lag features can be computed.

Why a reference file is needed: many features are percentile ranks WITHIN a fiscal year, and the
final risk score is a rank too. Ranking a firm only against the other firms in the input file gives
a different answer every time the file changes. So the input firms are ranked against the full
market cross-section of the same year from the reference CSV (the training data by default), and
only the input firms for the target year are written out.

Output columns: raw_score (mean of the members' bagged probabilities, uncalibrated; bags are
fraud-enriched), risk_percentile (rank among the market, the evaluated ensemble score) and
fraud_probability (risk_percentile mapped through the isotonic calibrator fit on out-of-sample
test years).

    python score_new_data.py --input my_firms.csv --year 2012 --out scored.csv
    python score_new_data.py --input my_firms.csv --year 2012 --reference data/data_FraudDetection_JAR2020.csv
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from features import engineer  # noqa: E402
from models import ensemble_scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True)
ap.add_argument('--year', type=int, required=True, help='fiscal year to score (prior-year rows are used only for lags)')
ap.add_argument('--reference', default='data/data_FraudDetection_JAR2020.csv',
                help='market cross-section the input firms are ranked against')
ap.add_argument('--model', default='results/final_models.joblib')
ap.add_argument('--out', default='scored.csv')
a = ap.parse_args()

art = joblib.load(a.model)
new = pd.read_csv(a.input)
ref = pd.read_csv(a.reference)
for d in (new, ref):
    if 'p_aaer' not in d:
        d['p_aaer'] = np.nan
    if 'misstate' not in d:
        d['misstate'] = 0

# Reference rows for the target year (the ranking population) and the year before (their lags).
ref = ref[ref['fyear'].between(a.year - 1, a.year)]
# A firm present in both files: the input row wins, so a user can overwrite stale reference data.
ref = ref[~ref.set_index(['gvkey', 'fyear']).index.isin(new.set_index(['gvkey', 'fyear']).index)]
new['_new'] = True
ref['_new'] = False
raw = pd.concat([ref, new], ignore_index=True)
if raw.loc[raw['_new'] & (raw['fyear'] == a.year)].empty:
    sys.exit(f'no input rows for fyear {a.year}')

df, _ = engineer(raw)                      # within-year ranks now span the whole market
df = df[df['fyear'] == a.year].reset_index(drop=True)   # prior-year rows served their purpose
D = art['winsorizer'].transform(df[art['winsor_cols']].astype(np.float32))
X = D[art['features']].values

# Same recipe as evaluation: rank every member within the year's cross-section, then average.
raw_scores = [art['models'][m].predict_proba(X) for m in art['ensemble_members']]
df['raw_score'] = np.mean(raw_scores, axis=0)
df['risk_percentile'] = ensemble_scores(raw_scores)
df['fraud_probability'] = art['calibrator'].predict(df['risk_percentile'])
df['market_size'] = len(df)

out = df.loc[df['_new'], ['gvkey', 'fyear', 'raw_score', 'risk_percentile', 'fraud_probability', 'market_size']]
out = out.sort_values('risk_percentile', ascending=False)
out.to_csv(a.out, index=False)
print(f'scored {len(out)} input firms for fyear {a.year} against {len(df)} market firms -> {a.out}')
print(f'model trained on fiscal years {art["train_years"][0]}-{art["train_years"][1]}; risk_percentile is the '
      'rank among the market, fraud_probability its calibrated counterpart.')
