"""Step 3 of the fresh test set: one row per filer and fiscal year in the layout of the paper's dataset.

Reads the first-reported facts saved by xbrl_download.py and maps XBRL tags to the 28 Compustat items,
then adds the 14 paper ratios (ratios.py). Money is in $ millions and shares in millions, as in Compustat.

Differences from Compustat that cannot be removed with public data:
* XBRL tags are chosen by each filer; Compustat standardises. An item with no matching tag is 0 when the
  item is commonly absent (inventory, preferred stock, debt, taxes payable...) and missing otherwise.
* There is no share price in XBRL. prcc_f is the public float on the 10-K cover divided by shares
  outstanding: measured at the end of the second fiscal quarter and without affiliates' shares, so it is
  a noisy stand-in for the fiscal-year-end price.
* Like the paper's sample, a row needs a classified balance sheet (current assets and current
  liabilities), which leaves out most banks and insurers.

    python experiments/fresh/xbrl_build.py   ->  data/external/fresh/xbrl_firm_years.csv
"""
import glob
import gzip
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ratios import ratios14  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'data', 'external', 'fresh', 'xbrl')
RAW28 = ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib', 'invt', 'ivao', 'ivst',
         'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale', 'sstk', 'txp', 'txt', 'xint', 'prcc_f']


def _d(s):
    return date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def firm_rows(cik, blob):
    inst, dur, dei = {}, {}, {}
    for tag, unit, start, end, val, accn, form, filed in blob['facts']:
        if tag.startswith('Entity'):
            dei.setdefault(tag, []).append((accn, end, val))
        elif unit == 'USD' or (unit == 'shares' and tag == 'CommonStockSharesOutstanding'):
            if start:
                days = (_d(end) - _d(start)).days
                if 330 <= days <= 400:                      # a full fiscal year (52/53-week years included)
                    old = dur.get((tag, end))
                    if old is None or abs(days - 365) < old[1]:
                        dur[(tag, end)] = (val, abs(days - 365))
            else:
                inst[(tag, end)] = val
    # one fiscal year per 10-K: the latest balance-sheet date it reports; the earliest filing wins
    years = {}
    for accn, form, filed, fye in blob['filings']:
        if fye not in years or filed < years[fye][1]:
            years[fye] = (accn, filed)
    out = []
    for fye, (accn, filed) in years.items():
        e = _d(fye)
        I = lambda *tags: next((inst[(t, fye)] for t in tags if (t, fye) in inst), None)        # noqa: E731,E741
        D = lambda *tags: next((dur[(t, fye)][0] for t in tags if (t, fye) in dur), None)        # noqa: E731
        z = lambda v: 0.0 if v is None else v                                                     # noqa: E731

        def total(*vals):
            got = [v for v in vals if v is not None]
            return sum(got) if got else None

        r = {'cik': cik, 'fye': fye, 'filed': filed, 'fyear': e.year if e.month >= 6 else e.year - 1}
        r['at'], r['act'], r['lct'] = I('Assets'), I('AssetsCurrent'), I('LiabilitiesCurrent')
        se, se_all = I('StockholdersEquity'), I('StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')
        lt = I('Liabilities')
        if lt is None and I('LiabilitiesAndStockholdersEquity') is not None and (se_all is not None or se is not None):
            eq = se_all if se_all is not None else se + z(I('MinorityInterest'))
            lt = (I('LiabilitiesAndStockholdersEquity') - eq - z(I('TemporaryEquityCarryingAmountAttributableToParent'))
                  - z(I('RedeemableNoncontrollingInterestEquityCarryingAmount')))
        r['lt'] = lt
        r['pstk'] = z(I('PreferredStockValue', 'PreferredStockValueOutstanding'))
        r['ceq'] = None if se is None else se - r['pstk']
        r['re'] = I('RetainedEarningsAccumulatedDeficit')
        cash = I('CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents')
        both = I('CashCashEquivalentsAndShortTermInvestments')
        ivst = I('ShortTermInvestments', 'MarketableSecuritiesCurrent', 'AvailableForSaleSecuritiesCurrent',
                 'AvailableForSaleSecuritiesDebtSecuritiesCurrent', 'OtherShortTermInvestments',
                 'HeldToMaturitySecuritiesCurrent', 'TradingSecuritiesCurrent')
        if ivst is None:
            ivst = both - cash if both is not None and cash is not None else 0.0
        r['ivst'] = ivst
        r['che'] = both if both is not None else (None if cash is None else cash + ivst)
        r['rect'] = z(I('AccountsReceivableNetCurrent', 'ReceivablesNetCurrent', 'AccountsNotesAndLoansReceivableNetCurrent',
                        'AccountsAndOtherReceivablesNetCurrent'))
        r['invt'] = z(I('InventoryNet'))
        net, gross = I('PropertyPlantAndEquipmentNet'), I('PropertyPlantAndEquipmentGross')
        acc = I('AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment')
        r['ppegt'] = gross if gross is not None else (net + acc if net is not None and acc is not None else z(net))
        r['ppent'] = net if net is not None else (gross - acc if gross is not None and acc is not None else z(gross))
        lti = I('LongTermInvestments')
        r['ivao'] = lti if lti is not None else z(total(I('OtherLongTermInvestments'), I('MarketableSecuritiesNoncurrent'),
                                                       I('AvailableForSaleSecuritiesNoncurrent', 'AvailableForSaleSecuritiesDebtSecuritiesNoncurrent')))
        r['ap'] = z(I('AccountsPayableCurrent', 'AccountsPayableTradeCurrent', 'AccountsPayableAndAccruedLiabilitiesCurrent'))
        r['txp'] = z(I('AccruedIncomeTaxesCurrent', 'TaxesPayableCurrent'))
        ltd_cur = I('LongTermDebtCurrent', 'LongTermDebtAndCapitalLeaseObligationsCurrent')
        dlc = I('DebtCurrent')
        if dlc is None:
            short = I('ShortTermBorrowings')
            if short is None:
                short = total(I('NotesPayableCurrent'), I('ConvertibleNotesPayableCurrent'), I('LinesOfCreditCurrent'),
                              I('CommercialPaper'), I('NotesPayableRelatedPartiesClassifiedCurrent'))
            dlc = z(ltd_cur) + z(short)
        r['dlc'] = dlc
        dltt = I('LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations')
        if dltt is None and I('LongTermDebt') is not None:
            dltt = I('LongTermDebt') - z(ltd_cur)
        if dltt is None:
            dltt = z(total(I('ConvertibleLongTermNotesPayable', 'ConvertibleNotesPayable'), I('SeniorLongTermNotes'),
                           I('LongTermNotesPayable'), I('LongTermLineOfCredit'), I('CapitalLeaseObligationsNoncurrent'),
                           I('NotesPayableRelatedPartiesNoncurrent')))
        r['dltt'] = dltt
        sale = D('Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet',
                 'RevenueFromContractWithCustomerIncludingAssessedTax')
        if sale is None:
            sale = total(D('SalesRevenueGoodsNet'), D('SalesRevenueServicesNet'))
        if sale is None:
            sale = D('RevenuesNetOfInterestExpense', 'RegulatedAndUnregulatedOperatingRevenue', 'ElectricUtilityRevenue',
                     'OilAndGasRevenue', 'RealEstateRevenueNet', 'HealthCareOrganizationRevenue')
        if sale is None:
            sale = total(D('InterestAndDividendIncomeOperating'), D('NoninterestIncome'))
        r['sale'] = z(sale)
        cogs = D('CostOfRevenue', 'CostOfGoodsAndServicesSold')
        r['cogs'] = cogs if cogs is not None else total(D('CostOfGoodsSold'), D('CostOfServices'))
        r['dp'] = D('DepreciationDepletionAndAmortization', 'DepreciationAndAmortization',
                    'DepreciationAmortizationAndAccretionNet', 'Depreciation')
        r['xint'] = z(D('InterestExpense', 'InterestExpenseDebt', 'InterestAndDebtExpense', 'InterestExpenseNonoperating'))
        r['txt'] = z(D('IncomeTaxExpenseBenefit'))
        r['ni'] = D('NetIncomeLoss', 'ProfitLoss')
        ib = D('IncomeLossFromContinuingOperations', 'IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest')
        r['ib'] = ib if ib is not None else r['ni']
        stock = total(D('ProceedsFromIssuanceOfCommonStock'), D('ProceedsFromIssuanceOfPreferredStockAndPreferenceStock'),
                      D('ProceedsFromStockOptionsExercised'), D('ProceedsFromIssuanceOfPrivatePlacement'),
                      D('ProceedsFromIssuanceInitialPublicOffering'))
        r['sstk'] = stock if stock is not None else z(D('ProceedsFromIssuanceOrSaleOfEquity'))
        debt = D('ProceedsFromIssuanceOfLongTermDebt', 'ProceedsFromIssuanceOfDebt')
        if debt is None:
            debt = total(D('ProceedsFromIssuanceOfSeniorLongTermDebt'), D('ProceedsFromConvertibleNotesPayable'),
                         D('ProceedsFromNotesPayable'), D('ProceedsFromIssuanceOfSecuredDebt'),
                         D('ProceedsFromIssuanceOfUnsecuredDebt'), D('ProceedsFromLongTermLinesOfCredit'),
                         D('ProceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet'), D('ProceedsFromRelatedPartyDebt'))
        r['dltis'] = z(debt)
        # cover-page facts of the same 10-K: shares outstanding and public float
        cover = {t: next((v for a, _, v in dei.get(t, []) if a == accn), None)
                 for t in ('EntityCommonStockSharesOutstanding', 'EntityPublicFloat')}
        shares = I('CommonStockSharesOutstanding')
        if shares is None or shares <= 0:
            shares = cover['EntityCommonStockSharesOutstanding']
        r['csho'] = shares
        fl, sh = cover['EntityPublicFloat'], cover['EntityCommonStockSharesOutstanding'] or shares
        r['prcc_f'] = fl / sh if fl and sh and sh > 0 and fl > 0 else None
        out.append(r)
    return out


def build():
    rows, names = [], {}
    files = sorted(glob.glob(os.path.join(SRC, 'facts', '*.json.gz')))
    for i, p in enumerate(files):
        cik = int(os.path.basename(p).split('.')[0])
        with gzip.open(p, 'rt', encoding='utf-8') as fh:
            blob = json.load(fh)
        names[cik] = blob.get('name')
        if blob['filings']:
            rows += firm_rows(cik, blob)
        if i % 3000 == 0:
            print(f'{i}/{len(files)} filers read', flush=True)
    df = pd.DataFrame(rows)
    money = [c for c in RAW28 + ['ppent'] if c not in ('csho', 'prcc_f')]
    df[money] = df[money].astype(float) / 1e6
    df['csho'] = df['csho'].astype(float) / 1e6
    df['prcc_f'] = df['prcc_f'].astype(float)
    df['name'] = df['cik'].map(names)
    # a firm that changes its year end can report two periods in one fiscal year: keep the later one
    df = df.sort_values(['cik', 'fyear', 'fye']).drop_duplicates(['cik', 'fyear'], keep='last')
    n0 = len(df)
    need = ['at', 'act', 'lct', 'lt', 'ceq', 'ni', 'che', 're']
    df = df[df[need].notna().all(axis=1) & (df['at'] > 0)].reset_index(drop=True)
    print(f'{n0} filer-years with a 10-K, {len(df)} with a classified balance sheet and the core items')
    df['gvkey'] = df['cik']
    rat = ratios14(df)
    df = df.merge(rat, on=['gvkey', 'fyear'])
    df['p_aaer'], df['misstate'] = np.nan, 0
    cols = ['fyear', 'gvkey', 'cik', 'name', 'fye', 'filed', 'p_aaer', 'misstate'] + RAW28 + ['ppent'] + \
           ['dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa', 'issue', 'bm', 'dpi',
            'reoa', 'EBIT', 'ch_fcf']
    df[cols].to_csv(os.path.join(os.path.dirname(SRC), 'xbrl_firm_years.csv'), index=False)
    print(df.groupby('fyear').size().to_string())
    print((df[RAW28].isna().mean() * 100).round(1).to_string())


if __name__ == '__main__':
    build()
