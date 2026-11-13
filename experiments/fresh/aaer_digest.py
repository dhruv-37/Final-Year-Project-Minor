"""Step 1 of the fresh-period labels: a reading digest of AAERs released 2019-2024.

Reads the AAER texts mirrored at huggingface.co/datasets/codewithdark/SEC-RPT-Panel (SEC public-domain
documents) and prints, for every release that talks about misstated financial statements in fiscal
years 2015 or later, the opening of its summary and the sentences that name the period. The labels in
aaer_labels_fresh.csv were then written by hand from this digest. Nothing here assigns a label.
"""
import os
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'data', 'external', 'fresh', 'hf_aaer')
KEY = re.compile(r'misstat|overstat|understat|restat|inflat|improperly rec|prematurely rec|false(ly)? (and misleading )?'
                 r'(financial|revenue|record)|fraudulent(ly)? (financial|record|recogni|account)|materially false|'
                 r'accounting fraud|not in (conformity|accordance) with GAAP|violat\w+ (of )?GAAP|earnings management|'
                 r'manipulat', re.I)
YEAR = re.compile(r'\b(20(?:1[5-9]|2[0-3]))\b')


def digest(lo=2019, hi=2024, width=650):
    m = pd.read_csv(os.path.join(SRC, 'aaer_manifest.csv'))
    m = m[m.year.between(lo, hi)].drop_duplicates('aaer_no').sort_values('aaer_no')
    kept = 0
    for r in m.itertuples():
        p = os.path.join(SRC, 'txt', r.aaer_no + '.txt')
        if not os.path.exists(p):
            continue
        t = re.sub(r'\s+', ' ', open(p, encoding='utf-8', errors='replace').read())
        body = t[re.search(r'release no\.? ?\d{4}', t, re.I).end():] if re.search(r'release no\.? ?\d{4}', t, re.I) else t
        sents = re.split(r'(?<=[.;])\s+(?=[A-Z0-9(])', body)
        hits = [s for s in sents if KEY.search(s) and YEAR.search(s) and len(s) < 600]
        if not hits:
            continue
        kept += 1
        s0 = re.search(r'\b(SUMMARY|Summary|finds\d? that:?|alleges?:?|alleged that)\b', body)
        start = body[s0.end():s0.end() + width] if s0 else ''
        print(f'## {r.aaer_no[5:]} | {r.release_date} | {str(r.respondents)[:80]} | {len(t)}c')
        print('S:', start.strip())
        for h in hits[:3]:
            if h[:80] not in start:
                print('H:', h.strip()[:330])
    print(f'# {kept} of {len(m)} releases kept', file=sys.stderr)


if __name__ == '__main__':
    digest(*(int(a) for a in sys.argv[1:3]))
