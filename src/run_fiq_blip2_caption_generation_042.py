"""Reference-caption-only Qwen Thinking pilot on the fixed 041/040C examples."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import json
from pathlib import Path

from run_fiq_context_generation_041 import BASE_URL, PICKS, SETTINGS, call_content, checked_endpoint, write


MODEL = 'qwen3.7-plus'
CASES = (*PICKS, ('shirt', 913))


def main(args):
    raw = args.cases.read_bytes()
    cases = json.loads(raw)
    if tuple((c['category'], c['source_index']) for c in cases) != CASES:
        raise ValueError('042 case order differs from plan')
    prompt = args.prompt.read_text(encoding='utf-8')
    inputs = []
    for case in cases:
        caption, fragments = case['reference_caption'], case['fragments']
        if not isinstance(caption, str) or not caption.strip() or len(fragments) != 2:
            raise ValueError('042 reference caption or edit fragments missing')
        data = json.dumps(dict(reference_caption=caption, original_edit_fragments=fragments), ensure_ascii=False)
        inputs.append([{'type': 'text', 'text': prompt + '\nInput data: ' + data}])
    if args.dry_run:
        print(f'{len(cases)} caption-only requests, {MODEL}, Thinking enabled.')
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    key = getpass.getpass('DashScope API key (hidden input): ').strip()
    if not key:
        raise ValueError('No API key entered')
    endpoint = checked_endpoint(BASE_URL)
    state = dict(status='running', model=MODEL, settings=SETTINGS, max_calls=len(cases), submitted=0,
                 reference_images_sent=0, target_images_sent=0, rows=[],
                 input_sha256=hashlib.sha256(raw).hexdigest(), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    write(args.output, state)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for start in range(0, len(cases), 3):
            stop = min(start + 3, len(cases))
            state['submitted'] = stop
            write(args.output, state)
            futures = [pool.submit(call_content, endpoint, key, MODEL, inputs[i]) for i in range(start, stop)]
            results = [future.result() for future in futures]
            state['rows'].extend(dict(case=case, output=result) for case, result in zip(cases[start:stop], results))
            if any('error' in result for result in results):
                state['status'] = 'failed'
                write(args.output, state)
                print('042 stopped after a provider error.', flush=True)
                return 1
            write(args.output, state)
            print(f'042 completed {stop}/{len(cases)} responses.', flush=True)
    state['status'] = 'complete'
    write(args.output, state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'prompt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    raise SystemExit(main(parser.parse_args()))
