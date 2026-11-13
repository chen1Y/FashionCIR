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
REJECTED = ('HTTP 403', 'AccessDenied.Unpurchased')


def resume_indices(state, case_sha, prompt_sha):
    """Only replay explicit provider denials whose outcome is known."""
    answers = state['answers']
    if (state['status'] != 'failed' or state.get('in_flight') or state['model'] != MODEL
            or state['settings'] != SETTINGS or state['max_calls'] != TOTAL
            or state['target_images_sent'] != 0 or len(answers) != state['submitted']
            or len(answers) > TOTAL or state['cases_sha256'] != case_sha
            or state['prompt_sha256'] != prompt_sha):
        raise ValueError('040C checkpoint identity or settled responses differ from plan')
    rejected = []
    for i, answer in enumerate(answers):
        if answer['model'] != MODEL:
            raise ValueError('040C checkpoint response model differs')
        if 'error' in answer:
            if (answer['error'], answer.get('provider_code')) != REJECTED:
                raise ValueError('040C checkpoint includes a request with an uncertain outcome')
            rejected.append(i)
    if not rejected:
        raise ValueError('040C has no explicit rejection to resume')
    return rejected + list(range(len(answers), TOTAL))


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
    case_sha = hashlib.sha256(raw_cases).hexdigest()
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    if args.resume:
        state = json.loads(args.output.read_text(encoding='utf-8'))
        pending = resume_indices(state, case_sha, prompt_sha)
    else:
        if args.output.exists() and not args.dry_run:
            raise FileExistsError(args.output)
        state = dict(status='running', model=MODEL, settings=SETTINGS, submitted=0, max_calls=TOTAL,
                     target_images_sent=0, cases_sha256=case_sha, prompt_sha256=prompt_sha, answers=[])
        pending = list(range(TOTAL))
    if args.dry_run:
        print(f'040C verified {len(cases)} reference-only rows; {len(pending)} remaining requests.')
        return 0
    key = getpass.getpass('DashScope API key (hidden input): ').strip()
    if not key:
        raise ValueError('No API key entered')
    state['status'] = 'running'
    state.setdefault('attempts_total', state['submitted'])
    rejected = state.setdefault('explicit_rejections', [])
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        cursor = 0
        while cursor < len(pending):
            # Confirm restored access on one pending row before sending a batch.
            width = 1 if args.resume and cursor == 0 else WORKERS
            batch = pending[cursor:cursor + width]
            state['submitted'] = max(state['submitted'], max(batch) + 1)
            state['attempts_total'] += len(batch)
            state['in_flight'] = batch
            write(args.output, state)
            futures = [pool.submit(call, endpoint, key, MODEL, prepared[i], prompt, cases[i]['fragments'])
                       for i in batch]
            answers = [job.result() for job in futures]
            for i, answer in zip(batch, answers):
                if i == len(state['answers']):
                    state['answers'].append(answer)
                else:
                    old = state['answers'][i]
                    rejected.append(dict(index=i, error=old['error'], provider_code=old['provider_code']))
                    state['answers'][i] = answer
            state.pop('in_flight')
            cursor += len(batch)
            if any('error' in answer for answer in answers):
                state['status'] = 'failed'
                write(args.output, state)
                print('040C stopped after a provider error; results checkpointed.', flush=True)
                return 1
            write(args.output, state)
            print(f"040C returned {sum('error' not in x for x in state['answers'])}/{TOTAL}", flush=True)
    state['status'] = 'complete'
    write(args.output, state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'prompt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    raise SystemExit(main(parser.parse_args()))
