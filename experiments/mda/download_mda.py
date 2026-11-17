"""Step 1 of PROTOCOL_MDA.md: download EDGAR-CORPUS and keep only the MD&A (Item 7) of every 10-K.

Source: huggingface.co/datasets/eloukas/edgar-corpus (Loukas et al. 2021, Apache-2.0): 10-K filings
1993-2020 split into items, one per CIK and filing year. Each yearly file (up to about 1.6 GB) is
downloaded to a temporary file with curl, checked against the size the Hugging Face API lists, parsed and
deleted; only cik, filing year and the Item 7 text are kept (about 41 GB pass through in all).

    python experiments/mda/download_mda.py            # resumable, one output file per filing year
Output: data/external/mda/mda_<year>.jsonl.gz, data/external/mda/skipped.json

Rules so a download cannot hang:
  1. Stall abort: a transfer slower than 50 KB/s for 60 s is aborted and resumed from where it stopped.
  2. Hard timer per attempt: one curl call may run at most 20 minutes (curl's own limit, and the process
     is killed 60 s after that if it still has not returned); the next attempt resumes the partial file.
  3. Budget per file: if a source file is not complete after 60 minutes of attempts (waits between
     attempts grow from 10 s to 2 minutes), the file and its filing year are skipped for this round.
  4. Skipped years are retried in two later rounds with fresh budgets. Years still missing after that
     are written to skipped.json and left out; their firm-years count as having no MD&A.
  A line that is not valid JSON is skipped and counted (a file whose size matches is never truncated).
"""
import gzip
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'external', 'mda')
BASE = 'https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main'
API = 'https://huggingface.co/api/datasets/eloukas/edgar-corpus/tree/main'
STALL_BYTES, STALL_SECONDS = 50_000, 60        # rule 1
ATTEMPT_SECONDS = 20 * 60                       # rule 2
FILE_BUDGET_SECONDS = 60 * 60                   # rule 3
ROUNDS = 3                                      # rule 4: the first pass and two retries


class Skip(Exception):
    pass


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def curl(args, timeout):
    """curl over IPv4 (the IPv6 route to the download CDN failed); killed if it outlives its own timer."""
    try:
        return subprocess.run(['curl', '-4', '-sL', '--connect-timeout', '30'] + args,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f'curl killed after {timeout}s')
        return None


def listing(y):
    for k in range(5):
        r = curl(['--max-time', '60', f'{API}/{y}'], timeout=120)
        try:
            return [(f['path'], f['size']) for f in json.loads(r.stdout) if f['path'].endswith('.jsonl')]
        except Exception:
            time.sleep(10 * (k + 1))
    raise Skip(f'no file listing for {y}')


def fetch(f, size):
    """Download one source file within its budget; returns the local path of the complete file."""
    raw = os.path.join(OUT, '_raw_' + f.replace('/', '_'))
    start, k = time.time(), 0
    while True:
        have = os.path.getsize(raw) if os.path.exists(raw) else 0
        if have == size:
            return raw
        if have > size:                                     # cannot be resumed: start this file again
            os.remove(raw)
            have = 0
        if time.time() - start > FILE_BUDGET_SECONDS:
            raise Skip(f'{f}: {have} of {size} bytes after {FILE_BUDGET_SECONDS // 60} minutes')
        curl(['-C', '-', '--speed-limit', str(STALL_BYTES), '--speed-time', str(STALL_SECONDS),
              '--max-time', str(ATTEMPT_SECONDS), '-o', raw, f'{BASE}/{f}'], timeout=ATTEMPT_SECONDS + 60)
        now = os.path.getsize(raw) if os.path.exists(raw) else 0
        if now != size:
            k += 1
            wait = min(120, 10 * 2 ** (k - 1))
            log(f'{f}: {now / 1e6:.0f} of {size / 1e6:.0f} MB, attempt {k} ended, resuming in {wait}s')
            time.sleep(wait)


def one_year(y):
    path = os.path.join(OUT, f'mda_{y}.jsonl.gz')
    if os.path.exists(path):
        return
    tmp, n, kept, bad, t0 = path + '.tmp', 0, 0, 0, time.time()
    with gzip.open(tmp, 'wt', encoding='utf-8') as out:
        for f, size in listing(y):
            raw = fetch(f, size)
            with open(raw, encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        bad += 1
                        continue
                    s7 = d.get('section_7') or ''
                    n += 1
                    kept += bool(s7.strip())
                    out.write(json.dumps({'cik': int(d['cik']), 'year': int(d['year']), 'file': d['filename'],
                                          'mda': s7}) + '\n')
            os.remove(raw)
    os.replace(tmp, path)
    log(f'{y}: {n} filings, {kept} with an MD&A, {bad} bad lines, {time.time() - t0:.0f}s')


def main():
    os.makedirs(OUT, exist_ok=True)
    for p in os.listdir(OUT):
        if p.endswith('.tmp'):
            os.remove(os.path.join(OUT, p))
    years = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 1993, 2021))
    todo = [y for y in years if not os.path.exists(os.path.join(OUT, f'mda_{y}.jsonl.gz'))]
    for rnd in range(1, ROUNDS + 1):
        if not todo:
            break
        log(f'round {rnd}: {len(todo)} years to do {todo}')
        failed = []
        for y in todo:
            try:
                one_year(y)
            except Skip as e:
                log(f'{y} skipped this round: {e}')
                failed.append(y)
            except Exception as e:
                log(f'{y} skipped this round: {e!r}')
                failed.append(y)
            finally:
                if os.path.exists(os.path.join(OUT, f'mda_{y}.jsonl.gz.tmp')):
                    os.remove(os.path.join(OUT, f'mda_{y}.jsonl.gz.tmp'))
        todo = failed
    json.dump({'skipped_filing_years': todo}, open(os.path.join(OUT, 'skipped.json'), 'w'))
    log('done' if not todo else f'done; skipped for good: {todo}')


if __name__ == '__main__':
    main()
