"""Step 2 of the fresh test set: annual-report facts for every XBRL filer, as first reported.

Source: the SEC's public XBRL API on data.sec.gov (frames for the list of filers, companyfacts for the
numbers). For every filer and every tag in TAGS, the value kept for a period is the one in the
EARLIEST-filed 10-K that reports it. Later 10-Ks repeat prior years as comparatives, often restated, and
a restated number is information nobody had at the time, so it is never used.

    python experiments/fresh/xbrl_download.py            # resumable, about 50 minutes
Run ONE copy only: the SEC allows 10 requests a second per address and blocks for about ten minutes
when that is exceeded.
Output: data/external/fresh/xbrl/universe.csv and facts/<cik>.json.gz (one small file per filer).
"""
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'external', 'fresh', 'xbrl')
UA = 'fraud-detection-research/1.0 (academic project)'
YEARS = range(2010, 2022)

# Compustat item -> XBRL tags that can carry it (how they are combined is in xbrl_build.py)
TAGS = {
    'at': ['Assets'],
    'act': ['AssetsCurrent'],
    'lct': ['LiabilitiesCurrent'],
    'lt': ['Liabilities', 'LiabilitiesAndStockholdersEquity',
           'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 'MinorityInterest',
           'TemporaryEquityCarryingAmountAttributableToParent', 'RedeemableNoncontrollingInterestEquityCarryingAmount'],
    'ceq': ['StockholdersEquity'],
    'pstk': ['PreferredStockValue', 'PreferredStockValueOutstanding'],
    're': ['RetainedEarningsAccumulatedDeficit'],
    'che': ['CashCashEquivalentsAndShortTermInvestments', 'CashAndCashEquivalentsAtCarryingValue', 'Cash',
            'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
    'ivst': ['ShortTermInvestments', 'MarketableSecuritiesCurrent', 'AvailableForSaleSecuritiesCurrent',
             'AvailableForSaleSecuritiesDebtSecuritiesCurrent', 'OtherShortTermInvestments',
             'HeldToMaturitySecuritiesCurrent', 'TradingSecuritiesCurrent'],
    'rect': ['AccountsReceivableNetCurrent', 'ReceivablesNetCurrent', 'AccountsNotesAndLoansReceivableNetCurrent',
             'AccountsAndOtherReceivablesNetCurrent'],
    'invt': ['InventoryNet'],
    'ppegt': ['PropertyPlantAndEquipmentGross', 'PropertyPlantAndEquipmentNet',
              'AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment'],
    'ivao': ['LongTermInvestments', 'OtherLongTermInvestments', 'MarketableSecuritiesNoncurrent',
             'AvailableForSaleSecuritiesNoncurrent', 'AvailableForSaleSecuritiesDebtSecuritiesNoncurrent'],
    'ap': ['AccountsPayableCurrent', 'AccountsPayableTradeCurrent', 'AccountsPayableAndAccruedLiabilitiesCurrent'],
    'txp': ['AccruedIncomeTaxesCurrent', 'TaxesPayableCurrent'],
    'dlc': ['DebtCurrent', 'LongTermDebtCurrent', 'ShortTermBorrowings', 'LongTermDebtAndCapitalLeaseObligationsCurrent',
            'NotesPayableCurrent', 'ConvertibleNotesPayableCurrent', 'LineOfCredit', 'LinesOfCreditCurrent',
            'CommercialPaper', 'CapitalLeaseObligationsCurrent', 'NotesPayableRelatedPartiesClassifiedCurrent'],
    'dltt': ['LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations', 'LongTermDebt',
             'ConvertibleNotesPayable', 'SeniorLongTermNotes', 'LongTermNotesPayable', 'CapitalLeaseObligationsNoncurrent',
             'LongTermLineOfCredit', 'ConvertibleLongTermNotesPayable', 'NotesPayableRelatedPartiesNoncurrent'],
    'sale': ['Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax',
             'RevenueFromContractWithCustomerIncludingAssessedTax', 'SalesRevenueGoodsNet', 'SalesRevenueServicesNet',
             'RevenuesNetOfInterestExpense', 'InterestAndDividendIncomeOperating', 'NoninterestIncome',
             'RealEstateRevenueNet', 'OilAndGasRevenue', 'HealthCareOrganizationRevenue',
             'RegulatedAndUnregulatedOperatingRevenue', 'ElectricUtilityRevenue'],
    'cogs': ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfGoodsSold', 'CostOfServices'],
    'dp': ['DepreciationDepletionAndAmortization', 'DepreciationAndAmortization',
           'DepreciationAmortizationAndAccretionNet', 'Depreciation'],
    'xint': ['InterestExpense', 'InterestExpenseDebt', 'InterestAndDebtExpense', 'InterestExpenseNonoperating'],
    'txt': ['IncomeTaxExpenseBenefit'],
    'ib': ['IncomeLossFromContinuingOperations',
           'IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest'],
    'ni': ['NetIncomeLoss', 'ProfitLoss'],
    'sstk': ['ProceedsFromIssuanceOfCommonStock', 'ProceedsFromIssuanceOrSaleOfEquity',
             'ProceedsFromIssuanceOfPreferredStockAndPreferenceStock', 'ProceedsFromStockOptionsExercised',
             'ProceedsFromIssuanceOfPrivatePlacement', 'ProceedsFromIssuanceInitialPublicOffering'],
    'dltis': ['ProceedsFromIssuanceOfLongTermDebt', 'ProceedsFromIssuanceOfDebt', 'ProceedsFromIssuanceOfSeniorLongTermDebt',
              'ProceedsFromConvertibleNotesPayable', 'ProceedsFromNotesPayable', 'ProceedsFromIssuanceOfSecuredDebt',
              'ProceedsFromIssuanceOfUnsecuredDebt', 'ProceedsFromLongTermLinesOfCredit',
              'ProceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet', 'ProceedsFromRelatedPartyDebt'],
    'csho': ['CommonStockSharesOutstanding'],
}
DEI = ['EntityCommonStockSharesOutstanding', 'EntityPublicFloat']
WANT = {t for ts in TAGS.values() for t in ts}

GAP = 0.2                                  # seconds between requests: 5 a second, half the SEC limit
_lock, _last, _blocked = threading.Lock(), [0.0], [0.0]


def get(url, tries=6):
    """GET with a shared rate limit. A 429 means the address is blocked: wait out the ten minutes."""
    for k in range(tries + 4):
        with _lock:
            wait = max(_last[0] + GAP, _blocked[0]) - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept-Encoding': 'gzip'})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return json.loads(gzip.decompress(raw) if r.headers.get('Content-Encoding') == 'gzip' else raw)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 429):
                if _blocked[0] < time.time():                # one shared wait for every thread
                    print('rate-limited by the SEC; waiting 11 minutes', flush=True)
                    _blocked[0] = time.time() + 660
                continue
            time.sleep(2 ** k)
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(f'failed: {url}')


def universe():
    path = os.path.join(OUT, 'universe.csv')
    if os.path.exists(path):
        return [int(line.split(',')[0]) for line in open(path, encoding='utf-8').read().splitlines()[1:]]
    names = {}
    for y in YEARS:
        for q in (1, 2, 3, 4):
            d = get(f'https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY{y}Q{q}I.json')
            for x in (d or {}).get('data', []):
                names[x['cik']] = x['entityName']
            print(f'frame CY{y}Q{q}I: {len(names)} filers so far', flush=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('cik,name\n')
        for c in sorted(names):
            fh.write(f'{c},"{(names[c] or "").replace(chr(34), "")}"\n')
    return sorted(names)


def reduce(d):
    """Earliest-filed 10-K value for every (tag, unit, start, end); plus each 10-K's own year end."""
    rows, filings = {}, {}
    for tax, want in (('us-gaap', WANT), ('dei', set(DEI))):
        for tag, body in d.get('facts', {}).get(tax, {}).items():
            if tag not in want:
                continue
            for unit, facts in body.get('units', {}).items():
                for f in facts:
                    form = f.get('form', '')
                    if not form.startswith('10-K'):
                        continue
                    key = (tag, unit, f.get('start', ''), f['end'])
                    cand = (f['filed'], form != '10-K', f['accn'], form, f['val'])
                    if key not in rows or cand < rows[key]:
                        rows[key] = cand
                    if tag == 'Assets':
                        a = filings.setdefault(f['accn'], [form, f['filed'], f['end']])
                        a[2] = max(a[2], f['end'])
    return {'name': d.get('entityName'),
            'filings': [[a] + v for a, v in filings.items()],
            'facts': [list(k) + [v[4], v[2], v[3], v[0]] for k, v in rows.items()]}


def one(cik):
    path = os.path.join(OUT, 'facts', f'{cik}.json.gz')
    if os.path.exists(path):
        return 0
    d = get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json')
    out = reduce(d) if d else {'name': None, 'filings': [], 'facts': []}
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8') as fh:
        json.dump(out, fh)
    os.replace(tmp, path)
    return 1


if __name__ == '__main__':
    os.makedirs(os.path.join(OUT, 'facts'), exist_ok=True)
    ciks = universe()
    print(f'{len(ciks)} filers', flush=True)
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(4) as ex:
        for i, n in enumerate(ex.map(one, ciks)):
            done += n
            if i % 500 == 0:
                print(f'{i}/{len(ciks)}  fetched {done}  {time.time() - t0:.0f}s', flush=True)
    print('done', flush=True)
