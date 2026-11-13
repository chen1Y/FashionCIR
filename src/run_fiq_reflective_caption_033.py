"""Run a five-triplet, reference-only DashScope Thinking pilot."""

import argparse
import base64
import getpass
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib import error, request
from zipfile import ZipFile

from shoes_online_pilot import BASE_URL, NoRedirect, checked_endpoint


MODEL = 'qwen3.7-flash'
CASES = (('dress', 0), ('toptee', 17), ('shirt', 172), ('dress', 20), ('shirt', 64))
MAX_TOKENS = 1024
THINKING_BUDGET = 768
DIRECT = ('You receive a reference garment photo and two FashionIQ edit fragments for one '
          'intended target. Apply the edits and preserve relevant unchanged garment attributes. '
          'Return only one natural English sentence describing the target garment in up to 35 words. '
          'Do not describe the person, pose or background.')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    tmp.replace(path)


def prepare(archive):
    from PIL import Image

    with ZipFile(archive) as source:
        cases = json.loads(source.read('manifest.json'))
        if tuple((x['category'], x['source_index']) for x in cases) != CASES:
            raise ValueError('The pilot requires the five preregistered samples')
        for case in cases:
            ref = case['reference']
            blob = source.read(ref['file'])
            if hashlib.sha256(blob).hexdigest() != ref['sha256']:
                raise ValueError('Reference JPEG SHA mismatch')
            with Image.open(io.BytesIO(blob)) as original:
                photo = original.convert('RGB')
            photo.thumbnail((512, 512))
            encoded = io.BytesIO()
            photo.save(encoded, format='JPEG', quality=90)
            url = 'data:image/jpeg;base64,' + base64.b64encode(encoded.getvalue()).decode('ascii')
            yield case, url


def review(answer):
    match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', answer.strip(), flags=re.DOTALL)
    try:
        obj = json.loads(match[1] if match else answer)
    except ValueError:
        return None
    keys = {'retain', 'erase', 'add', 'relative', 'unclear'}
    if not isinstance(obj, dict) or set(obj) != {'inspection', 'target_caption'}:
        return None
    if not isinstance(obj['target_caption'], str) or not obj['target_caption'].strip():
        return None
    inspection = obj['inspection']
    if not isinstance(inspection, dict) or set(inspection) != keys:
        return None
    for key in keys - {'relative'}:
        if not isinstance(inspection[key], list) or any(not isinstance(x, str) for x in inspection[key]):
            return None
    if not isinstance(inspection['relative'], list):
        return None
    for relation in inspection['relative']:
        if (not isinstance(relation, dict) or set(relation) != {'cue', 'part', 'anchor'}
                or not all(isinstance(value, str) for value in relation.values())):
            return None
    return obj


def request_body(url, prompt, fragments):
    return dict(model=MODEL, messages=[{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': url}},
        {'type': 'text', 'text': prompt + '\nFashionIQ edit fragments: ' + json.dumps(fragments)}
    ]}], stream=True, stream_options={'include_usage': True}, enable_thinking=True,
        thinking_budget=THINKING_BUDGET, max_tokens=MAX_TOKENS, temperature=0)


def read_stream(lines, key):
    final, thinking_present, done, usage, finish, model, request_id = [], False, False, None, None, None, None
    for line in lines:
        if not line.startswith(b'data:'):
            continue
        item = line[5:].strip()
        if item == b'[DONE]':
            done = True
            break
        chunk = json.loads(item)
        if 'error' in chunk:
            raise RuntimeError('DashScope returned an error inside the stream')
        model = chunk.get('model') or model
        request_id = chunk.get('id') or request_id
        usage = chunk.get('usage') or usage
        for choice in chunk.get('choices', []):
            finish = choice.get('finish_reason') or finish
            delta = choice.get('delta') or {}
            thinking_present = thinking_present or bool(delta.get('reasoning_content'))
            if delta.get('content'):
                final.append(delta['content'])
    if not done or finish is None:
        raise RuntimeError('Incomplete provider stream')
    answer = ''.join(final).strip()
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    if '<think>' in answer or '</think>' in answer:
        answer = ''
    answer = re.sub(r'sk-[A-Za-z0-9_-]{20,}', '[REDACTED]', answer.replace(key, '[REDACTED]'))
    return dict(final_answer=answer, thinking_present=thinking_present, usage=usage,
                finish=finish, provider_model=model, request_id=request_id)


def run(args):
    cases = list(prepare(args.samples))
    reflective = args.prompt.read_text(encoding='utf8')
    if args.dry_run:
        for case, url in cases:
            payload = request_body(url, reflective, case['fragments'])
            assert case['target']['id'] not in json.dumps(payload['messages'][0]['content'][1])
            print(case['category'], case['source_index'], case['reference']['id'], case['fragments'])
        return
    key = getpass.getpass('DashScope API key (hidden input): ').strip()
    if not key:
        raise ValueError('No API key entered')
    endpoint = checked_endpoint(args.base_url)
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    arms = (('reflective', reflective),)
    if not args.reflective_only:
        arms += (('direct', DIRECT),)
    args.output.mkdir(parents=True)
    state = dict(status='running', model=MODEL, thinking=True, thinking_budget=THINKING_BUDGET,
                 max_tokens=MAX_TOKENS, max_calls=len(arms) * len(cases), sent_calls=0, target_images_sent=0,
                 prompt_sha256=hashlib.sha256(reflective.encode()).hexdigest(),
                 direct_prompt=DIRECT, cases=[case for case, _ in cases], answers=[])
    status = args.output / 'results.json'
    write(status, state)
    for arm, prompt in arms:
        for case, url in cases:
            state['sent_calls'] += 1
            state['active'] = dict(arm=arm, category=case['category'], source_index=case['source_index'])
            write(status, state)
            req = request.Request(endpoint, data=json.dumps(request_body(url, prompt, case['fragments'])).encode(),
                                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
            start = time.monotonic()
            try:
                with opener.open(req, timeout=90) as response:
                    result = read_stream(response, key)
            except (error.HTTPError, error.URLError, RuntimeError, ValueError, TimeoutError) as exc:
                detail = f'HTTP {exc.code}' if isinstance(exc, error.HTTPError) else type(exc).__name__
                state.update(status='failed', error=detail)
                write(status, state)
                print(f'Pilot stopped at {state["active"]}: {detail}; no automatic retry.')
                return 1
            result.update(state.pop('active'), seconds=round(time.monotonic() - start, 2))
            if arm == 'reflective':
                answer = result['final_answer']
                result['parsed'] = review(answer) if result['finish'] == 'stop' else None
                if result['parsed'] is None:
                    result['unparsed_answer_sha256'] = hashlib.sha256(answer.encode()).hexdigest()
                    result['unparsed_characters'] = len(answer)
                    result['final_answer'] = ''
            state['answers'].append(result)
            write(status, state)
            print(arm, case['category'], case['source_index'], result['finish'], flush=True)
    state['status'] = 'complete'
    write(status, state)
    print(f'Results saved to {status}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('samples', 'prompt', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--base-url', default=BASE_URL)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--reflective-only', action='store_true')
    raise SystemExit(run(parser.parse_args()))
