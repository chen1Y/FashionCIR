"""Generate the fixed 040C subset with the 041 Qwen3.7-Plus request."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from run_fiq_context_generation_041 import BASE_URL, SETTINGS, call, checked_endpoint, write


MODEL = 'qwen3.7-plus'
TOTAL = 576
WORKERS = 8


def main(args):
    prompt = args.prompt.read_text(encoding='utf-8')
    endpoint = checked_endpoint(BASE_URL)
    with ZipFile(args.bundle) as bundle:
        raw_cases = bundle.read('cases.json')
        cases = json.loads(raw_cases)
        if len(cases) != TOTAL or len({(c['category'], c['split'], c['source_index']) for c in cases}) != TOTAL:
            raise ValueError('040C subset changed')
        prepared = []
        for case in cases:
            if set(case) != {'category', 'split', 'offset', 'source_index', 'candidate', 'reference_sha256', 'fragments'}:
                raise ValueError('Unexpected query fields')
            blob = bundle.read(f"references/{case['candidate']}.jpg")
            if hashlib.sha256(blob).hexdigest() != case['reference_sha256'] or not blob.startswith(b'\xff\xd8'):
                raise ValueError('040C reference JPEG mismatch')
            if len(case['fragments']) != 2 or sum(len(s.encode()) for s in case['fragments']) >= 1024:
                raise ValueError('040C edit fragments changed')
            prepared.append('data:image/jpeg;base64,' + base64.b64encode(blob).decode('ascii'))
    if args.dry_run:
        print(f'040C verified {len(cases)} requests with one reference image each.')
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    key = getpass.getpass('DashScope API key (hidden input): ').strip()
    if not key:
        raise ValueError('No API key entered')
    state = dict(status='running', model=MODEL, settings=SETTINGS, submitted=0, max_calls=TOTAL,
                 target_images_sent=0, cases_sha256=hashlib.sha256(raw_cases).hexdigest(),
                 prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), answers=[])
    write(args.output, state)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, TOTAL, WORKERS):
            stop = min(TOTAL, start + WORKERS)
            state['submitted'] = stop
            write(args.output, state)
            futures = [pool.submit(call, endpoint, key, MODEL, prepared[i], prompt, cases[i]['fragments'])
                       for i in range(start, stop)]
            answers = [job.result() for job in futures]
            state['answers'].extend(answers)
            if any('error' in answer for answer in answers):
                state['status'] = 'failed'
                write(args.output, state)
                print(f'040C stopped after a provider error. Saved {stop} responses/statuses.', flush=True)
                return 1
            write(args.output, state)
            print(f'040C generated {stop}/{TOTAL}', flush=True)
    state['status'] = 'complete'
    write(args.output, state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'prompt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    raise SystemExit(main(parser.parse_args()))
