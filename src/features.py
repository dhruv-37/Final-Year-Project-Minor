"""Feature engineering for accounting-fraud detection.

Starts from the 28 raw Compustat items + 14 ratios released by Bao et al. (2020, JAR)
and adds scale-free, accounting-meaningful features (Beneish M-score components,
Altman Z-score, common-size items, year-on-year changes, "red-flag" growth gaps).

All lags use the *previous fiscal year of the same firm only* (NaN if the firm has a gap),
so no information from the future ever leaks into a row.
"""
import numpy as np
import pandas as pd

RAW28 = ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib',
         'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale',
         'sstk', 'txp', 'txt', 'xint', 'prcc_f']
RATIOS14 = ['dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs', 'ch_cm',
            'ch_roa', 'issue', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf']
PAPER42 = RAW28 + RATIOS14

LAG_COLS = ['at', 'sale', 'cogs', 'rect', 'act', 'ppegt', 'dp', 'lct', 'dltt', 'dlc',
            'invt', 'ib', 'che', 're', 'lt', 'ap']

GLOSSARY = {
    # Beneish M-score components
    'dsri': 'Days-sales-in-receivables index (receivables growing faster than sales)',
    'gmi': 'Gross-margin index (>1 = margins deteriorating)',
    'aqi': 'Asset-quality index (rising share of non-current, non-PPE assets)',
    'sgi': 'Sales-growth index (sales this year / last year)',
    'depi': 'Depreciation index (slowing depreciation rate)',
    'lvgi': 'Leverage index (rising leverage)',
    'tata': 'Total accruals to total assets (earnings not backed by cash)',
    'beneish_m': 'Beneish M-score (partial; SGAI set neutral). Higher = more manipulation-like',
    'altman_z': "Altman Z-score. Lower = more financial distress (incentive to manipulate)",
    # growth gaps
    'rect_vs_sale_g': 'Receivables growth minus sales growth (classic revenue-inflation flag)',
    'inv_vs_cogs_g': 'Inventory growth minus COGS growth',
    'asset_g': 'Total-asset growth',
    'rect_g': 'Receivables growth',
    'inv_g': 'Inventory growth',
    # levels
    'roa': 'Return on assets', 'margin': 'Gross margin', 'asset_turn': 'Asset turnover',
    'cur_ratio': 'Current ratio', 'quick': 'Quick ratio', 'lev': 'Total liabilities / assets',
    'debt_ta': 'Interest-bearing debt / assets', 'cash_ta': 'Cash / assets',
    'rect_sale': 'Receivables / sales', 'inv_cogs': 'Inventory / COGS',
    'ap_cogs': 'Payables / COGS', 'ppe_ta': 'Gross PPE / assets', 'etr': 'Effective tax rate',
    'int_cov': 'Interest coverage', 'mve_ta': 'Market value of equity / assets',
    'log_at': 'Firm size (log assets)', 'log_sale': 'Log sales', 'loss': 'Loss-making flag',
    'fin_need': 'External financing raised / assets (debt + equity issuance)',
    # changes
    'd_lev': 'Change in leverage', 'd_margin': 'Change in gross margin',
    'd_rect_sale': 'Change in receivables / sales', 'd_asset_turn': 'Change in asset turnover',
    'd_roa': 'Change in ROA',
    # paper ratios
    'dch_wc': 'Working-capital accruals', 'ch_rsst': 'RSST accruals',
    'dch_rec': 'Change in receivables', 'dch_inv': 'Change in inventory',
    'soft_assets': '% soft assets', 'ch_cs': 'Change in cash sales', 'ch_cm': 'Change in cash margin',
    'ch_roa': 'Change in ROA (paper)', 'issue': 'Security issuance flag', 'bm': 'Book-to-market',
    'dpi': 'Depreciation index (paper)', 'reoa': 'Retained earnings / assets',
    'EBIT': 'EBIT / assets', 'ch_fcf': 'Change in free cash flow',
}


def sd(a, b):
    """Safe divide: NaN where the denominator is ~0, inf -> NaN."""
    b = b.where(b.abs() > 1e-6)
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _lags(df):
    g = df.groupby('gvkey')
    prev = g[LAG_COLS].shift(1)
    consecutive = g['fyear'].shift(1) == (df['fyear'] - 1)
    prev = prev.where(consecutive)
    prev.columns = [c + '_l' for c in LAG_COLS]
    return prev


def engineer(df: pd.DataFrame):
    """Return (sorted df with all features, dict of feature groups)."""
    df = df.sort_values(['gvkey', 'fyear']).reset_index(drop=True)
    X = pd.concat([df, _lags(df)], axis=1)
    f = pd.DataFrame(index=df.index)

    A, S = X['at'], X['sale']
    mve = X['prcc_f'] * X['csho']

    # ---- size / levels --------------------------------------------------------------
    f['log_at'] = np.log1p(A.clip(lower=0))
    f['log_sale'] = np.log1p(S.clip(lower=0))
    f['roa'] = sd(X['ib'], A)
    f['margin'] = sd(S - X['cogs'], S)
    f['asset_turn'] = sd(S, A)
    f['cur_ratio'] = sd(X['act'], X['lct'])
    f['quick'] = sd(X['act'] - X['invt'], X['lct'])
    f['lev'] = sd(X['lt'], A)
    f['debt_ta'] = sd(X['dlc'] + X['dltt'], A)
    f['cash_ta'] = sd(X['che'], A)
    f['rect_sale'] = sd(X['rect'], S)
    f['inv_cogs'] = sd(X['invt'], X['cogs'])
    f['ap_cogs'] = sd(X['ap'], X['cogs'])
    f['ppe_ta'] = sd(X['ppegt'], A)
    f['etr'] = sd(X['txt'], X['ib'] + X['txt'])
    f['int_cov'] = sd(X['ib'] + X['xint'] + X['txt'], X['xint'])
    f['mve_ta'] = sd(mve, A)
    f['loss'] = (X['ib'] < 0).astype(float)
    f['fin_need'] = sd(X['dltis'].fillna(0) + X['sstk'].fillna(0), A)
    for c in ['act', 'ap', 'che', 'dlc', 'dltt', 'dp', 'ib', 'invt', 'ivao', 'lct', 're', 'rect',
              'txp', 'xint']:
        f[f'{c}_ta'] = sd(X[c], A)

    # ---- growth / change features ---------------------------------------------------
    f['asset_g'] = sd(A, X['at_l']) - 1
    f['rect_g'] = sd(X['rect'], X['rect_l']) - 1
    f['inv_g'] = sd(X['invt'], X['invt_l']) - 1
    sale_g = sd(S, X['sale_l']) - 1
    cogs_g = sd(X['cogs'], X['cogs_l']) - 1
    f['sale_g'] = sale_g
    f['rect_vs_sale_g'] = f['rect_g'] - sale_g
    f['inv_vs_cogs_g'] = f['inv_g'] - cogs_g
    lev_l = sd(X['lt_l'], X['at_l'])
    margin_l = sd(X['sale_l'] - X['cogs_l'], X['sale_l'])
    rect_sale_l = sd(X['rect_l'], X['sale_l'])
    turn_l = sd(X['sale_l'], X['at_l'])
    roa_l = sd(X['ib_l'], X['at_l'])
    f['d_lev'] = f['lev'] - lev_l
    f['d_margin'] = f['margin'] - margin_l
    f['d_rect_sale'] = f['rect_sale'] - rect_sale_l
    f['d_asset_turn'] = f['asset_turn'] - turn_l
    f['d_roa'] = f['roa'] - roa_l

    # ---- Beneish M-score components (SGAI not available -> neutral 1) ---------------
    f['dsri'] = sd(sd(X['rect'], S), rect_sale_l)
    f['gmi'] = sd(margin_l, f['margin'])
    f['aqi'] = sd(1 - sd(X['act'] + X['ppegt'], A), 1 - sd(X['act_l'] + X['ppegt_l'], X['at_l']))
    f['sgi'] = sd(S, X['sale_l'])
    f['depi'] = sd(sd(X['dp_l'], X['dp_l'] + X['ppegt_l']), sd(X['dp'], X['dp'] + X['ppegt']))
    f['lvgi'] = sd(sd(X['lct'] + X['dltt'], A), sd(X['lct_l'] + X['dltt_l'], X['at_l']))
    nwc = X['rect'] + X['invt'] - X['ap']
    nwc_l = X['rect_l'] + X['invt_l'] - X['ap_l']
    f['tata'] = sd(nwc - nwc_l - X['dp'], A)
    f['beneish_m'] = (-4.84 + 0.920 * f['dsri'] + 0.528 * f['gmi'] + 0.404 * f['aqi']
                      + 0.892 * f['sgi'] + 0.115 * f['depi'] - 0.172 * 1.0
                      + 4.679 * f['tata'] - 0.327 * f['lvgi'])

    # ---- Altman Z ---------------------------------------------------------------------
    f['altman_z'] = (1.2 * sd(X['act'] - X['lct'], A) + 1.4 * sd(X['re'], A)
                     + 3.3 * X['EBIT'] + 0.6 * sd(mve, X['lt']) + 1.0 * sd(S, A))

    f = f.replace([np.inf, -np.inf], np.nan)
    # ---- within-year percentile ranks: removes era drift (cross-section of the same FY only) ---
    rank_src = [c for c in RATIOS14 if c in df] + [c for c in f.columns if c not in ('loss',)]
    tmp = pd.concat([df[['fyear']], df[[c for c in RATIOS14]], f], axis=1)
    ranks = tmp.groupby('fyear')[rank_src].rank(pct=True)
    ranks.columns = [c + '_yr' for c in rank_src]
    out = pd.concat([df, f, ranks], axis=1)
    groups = {
        'raw28': RAW28,                      # baseline only: absolute $ levels carry a firm-size bias
        'ratios14': RATIOS14,
        'paper42': PAPER42,
        'engineered': list(f.columns),
        # candidate sets for the proposed model are scale-free only (no raw $ items)
        'enhanced': RATIOS14 + list(f.columns),
        'rank_only': list(ranks.columns) + ['loss', 'log_at'],
        'enhanced_rank': RATIOS14 + list(f.columns) + list(ranks.columns),
    }
    return out, groups


class Winsorizer:
    """Clip each column to train-set percentiles (fit on train only -> no leakage)."""

    def __init__(self, lo=0.01, hi=0.99):
        self.lo, self.hi = lo, hi

    def fit(self, X: pd.DataFrame):
        self.lo_ = X.quantile(self.lo)
        self.hi_ = X.quantile(self.hi)
        return self

    def transform(self, X: pd.DataFrame):
        return X.clip(self.lo_, self.hi_, axis=1)


def describe(feat):
    """Human-readable meaning for any feature name (handles raw items and *_ta / *_yr variants)."""
    RAW = {'act': 'current assets', 'ap': 'accounts payable', 'at': 'total assets', 'ceq': 'common equity',
           'che': 'cash & ST investments', 'cogs': 'cost of goods sold', 'csho': 'shares outstanding',
           'dlc': 'current debt', 'dltis': 'LT debt issued', 'dltt': 'long-term debt',
           'dp': 'depreciation & amortisation', 'ib': 'income before extraordinary items',
           'invt': 'inventory', 'ivao': 'investments & advances (other)', 'ivst': 'ST investments',
           'lct': 'current liabilities', 'lt': 'total liabilities', 'ni': 'net income',
           'ppegt': 'gross PPE', 're': 'retained earnings', 'rect': 'receivables', 'sale': 'sales',
           'sstk': 'stock issued (cash raised)', 'txp': 'taxes payable', 'txt': 'income taxes',
           'xint': 'interest expense', 'pstk': 'preferred stock', 'prcc_f': 'share price (FY close)'}
    if feat in GLOSSARY:
        return GLOSSARY[feat]
    if feat.endswith('_yr'):
        return 'Within-year percentile rank of ' + feat[:-3] + ' (' + describe(feat[:-3]) + ')'
    if feat.endswith('_ta') and feat[:-3] in RAW:
        return RAW[feat[:-3]].capitalize() + ' / total assets'
    return RAW.get(feat, '').capitalize()
