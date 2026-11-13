"""Bounded paired FashionIQ context-generation pilot with two Thinking models."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import json
from pathlib import Path
import re
import time
from urllib import error, request

from shoes_online_pilot import BASE_URL, NoRedirect, checked_endpoint


MODELS = ('qwen3.7-plus', 'qwen3.8-flash')
FIELDS = ('garment', 'color', 'shape', 'detail', 'texture')
PICKS = (('toptee', 2), ('dress', 0), ('dress', 26), ('shirt', 503), ('toptee', 929))
SETTINGS = dict(stream=True, stream_options={'include_usage': True}, enable_thinking=True,
                thinking_budget=1536, max_tokens=2560, temperature=0)
NEG = re.compile(r"\b(?:no|not|without|never|lack\s+of)\b|\bnon[-–]", re.I)
POINTER = re.compile(r'\b(?:than|as|to|like)\s+(?:in\s+)?(?:the\s+)?(?:reference|original|source)\s+(?:image|picture|photo)\b', re.I)


def write(path, state):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def parse(answer):
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', answer.strip(), flags=re.DOTALL)
    try:
        obj = json.loads(fenced[1] if fenced else answer)
    except ValueError:
        return None, ['invalid JSON']
    keys = {'ref_attrs', 'drop_attrs', 'Tkep', 'Tmod', 'final_text', 'uncertain'}
    if not isinstance(obj, dict) or set(obj) != keys:
        return obj, ['wrong top-level fields']
    refs = obj['ref_attrs']
    if not isinstance(refs, dict) or set(refs) != set(FIELDS):
        return obj, ['wrong reference fields']
    arrays = list(refs.values()) + [obj['drop_attrs'], obj['uncertain']]
    if not all(isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v) for v in arrays):
        return obj, ['invalid attribute lists']
    kept, mod, final = (obj[key] for key in ('Tkep', 'Tmod', 'final_text'))
    if not all(isinstance(value, str) and value.strip() for value in (kept, mod, final)):
        return obj, ['missing caption clauses']
    flags = []
    if final != f'{kept}, and {mod}.' or kept[-1] in '.?!;,' or mod[-1] in '.?!;,':
        flags.append('final_text differs from exact join')
    known = {x.casefold().strip() for v in refs.values() for x in v}
    if any(x.casefold().strip() not in known for x in obj['drop_attrs']):
        flags.append('drop attribute absent from ref_attrs')
    if any(NEG.search(text) for text in (kept, mod)):
        flags.append('explicit negation in output clause')
    if POINTER.search(mod):
        flags.append('reference-picture pointer in Tmod')
    if len(final.split()) > 42:
        flags.append('caption longer than 42 words')
    return obj, flags


def call(endpoint, key, model, url, prompt, fragments):
    return call_content(endpoint, key, model, [
        {'type': 'image_url', 'image_url': {'url': url}},
        {'type': 'text', 'text': prompt + '\nOriginal edit fragments: ' + json.dumps(fragments)},
    ])


def call_content(endpoint, key, model, content):
    started = time.monotonic()
    body = dict(model=model, **SETTINGS, messages=[{'role': 'user', 'content': content}])
    req = request.Request(endpoint, data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    answer, thought, done, finish, usage, actual = [], False, False, None, None, None
    try:
        with opener.open(req, timeout=90) as stream:
            for line in stream:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                chunk = json.loads(data)
                if 'error' in chunk:
                    raise RuntimeError('provider stream error')
                actual = chunk.get('model') or actual
                usage = chunk.get('usage') or usage
                for choice in chunk.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    delta = choice.get('delta', {})
                    thought = thought or bool(delta.get('reasoning_content'))
                    if delta.get('content'):
                        answer.append(delta['content'])
        if not done or finish is None:
            raise RuntimeError('incomplete stream')
    except error.HTTPError as exc:
        try:
            code = json.loads(exc.read(8192)).get('error', {}).get('code', '')
        except (ValueError, AttributeError):
            code = ''
        safe_code = code if isinstance(code, str) and re.fullmatch(r'[A-Za-z][\w.-]{0,63}', code) and key not in code else ''
        return dict(model=model, error=f'HTTP {exc.code}', provider_code=safe_code)
    except (error.URLError, OSError, ValueError, RuntimeError) as exc:
        return dict(model=model, error=type(exc).__name__)
    final = re.sub(r'<think>.*?</think>', '', ''.join(answer), flags=re.DOTALL).strip()
    if '<think>' in final or '</think>' in final:
        final = ''
    final = re.sub(r'sk-[A-Za-z0-9_-]{20,}', '[REDACTED]', final.replace(key, '[REDACTED]'))
    parsed, flags = parse(final)
    return dict(model=model, provider_model=actual, thinking_present=thought, finish=finish, usage=usage,
                seconds=round(time.monotonic() - started, 3), answer=parsed, flags=flags, raw_final=final)


def main(args):
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    if tuple((row['category'], row['source_index']) for row in cases) != PICKS:
        raise ValueError('041 case order differs from the fixed plan')
    prompt = args.prompt.read_text(encoding='utf-8')
    endpoint = checked_endpoint(BASE_URL)
    prepared = []
    for row in cases:
        image = args.images / f"{row['candidate']}.jpg"
        data = image.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['reference_sha256'] or not data.startswith(b'\xff\xd8'):
            raise ValueError('041 reference JPEG mismatch')
        if len(row['fragments']) != 2 or not all(isinstance(x, str) for x in row['fragments']):
            raise ValueError('041 requires two original fragments')
        prepared.append((row, 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')))
    if args.dry_run:
        print(f'{len(cases)} paired examples, {len(cases)*len(MODELS)} planned calls')
        return 0
    if args.output.exists():
        raise FileExistsError('041 results already exist')
    key = getpass.getpass('DashScope API key (hidden input): ').strip()
    if not key:
        raise ValueError('No API key entered')
    state = dict(status='running', models=MODELS, settings=SETTINGS, max_calls=10, submitted=0, target_images_sent=0,
                 prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), pairs=[])
    write(args.output, state)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for row, url in prepared:
            state['submitted'] += len(MODELS)
            state['active'] = [row['category'], row['source_index']]
            write(args.output, state)
            futures = [executor.submit(call, endpoint, key, model, url, prompt, row['fragments']) for model in MODELS]
            outputs = [future.result() for future in futures]
            state['pairs'].append(dict(case=row, models=outputs))
            state.pop('active')
            if any('error' in output for output in outputs):
                state['status'] = 'failed'
                write(args.output, state)
                print('041 stopped after a provider error. See paired results.')
                return 1
            write(args.output, state)
            print(f"Saved {row['category']} {row['source_index']}", flush=True)
    state['status'] = 'complete'
    write(args.output, state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'images', 'prompt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    raise SystemExit(main(parser.parse_args()))
