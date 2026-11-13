"""Generate edited attribute phrases on the fixed 033 reference-only pilot."""

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import re
from urllib import error, request

from run_fiq_reflective_caption_033 import (
    BASE_URL, MAX_TOKENS, THINKING_BUDGET, NoRedirect, checked_endpoint,
    prepare, read_stream, request_body, write,
)
from run_fiq_reflective_caption_033c import VALUES, parse


MODEL = 'qwen3.8-flash'
POINTER = re.compile(r'\b(?:original|reference|source|pictured)\b', re.I)
COMPARATIVE = re.compile(r'\b(?:lighter|darker|brighter|longer|shorter|fuller|more casual|more formal)\b', re.I)


def validate(plan, fragments):
    if (not isinstance(plan, dict) or set(plan) != {'edits', 'unresolved'}
            or not all(isinstance(v, dict) for v in plan.values())):
        return {}, {'plan': 'invalid envelope'}
    accepted, issues = {}, {}
    for field, reason in plan['unresolved'].items():
        if field not in (*VALUES, 'unspecified') or not isinstance(reason, str) or not reason.strip():
            issues[f'unresolved.{field}'] = 'invalid unresolved entry'
    for field, edit in plan['edits'].items():
        if field not in VALUES or not isinstance(edit, dict) or set(edit) != {'op', 'phrase', 'erase'}:
            issues[field] = 'invalid edit fields'
            continue
        phrase, op, erase = edit['phrase'], edit['op'], edit['erase']
        if (not isinstance(phrase, str) or not 0 < len(phrase.split()) <= 10
                or POINTER.search(phrase) or re.search(r'[.;!?:\n]', phrase)):
            issues[field] = 'not a standalone phrase'
        elif not isinstance(erase, list) or any(not isinstance(v, str) or not v.strip() for v in erase):
            issues[field] = 'invalid erase list'
        elif op == 'relative' and not (COMPARATIVE.search(phrase) and re.search(r'\bthan\s+\S', phrase, re.I)):
            issues[field] = 'relative phrase lacks a concrete benchmark'
        elif op in ('set', 'remove') and COMPARATIVE.search(phrase):
            issues[field] = 'comparative placed in a concrete edit'
        elif op not in ('set', 'remove', 'relative'):
            issues[field] = 'invalid edit operation'
        else:
            accepted[field] = edit
    text = ' '.join(fragments).lower()
    if 'sleeves' in accepted:
        sleeve = accepted['sleeves']
        for value in ('long', 'short'):
            if (sleeve['op'] == 'set' and re.search(rf'\b{value}(?:[- ]sleeved| sleeves)\b', sleeve['phrase'], re.I)
                    and re.search(rf'\b{value}er sleeves\b', text)
                    and not re.search(rf'\b{value}(?:[- ]sleeved| sleeves)\b', text)):
                issues['sleeves'] = 'relative instruction became an absolute sleeve length'
    straps = accepted.get('straps', {}).get('phrase', '')
    bodice = accepted.get('bodice', {}).get('phrase', '')
    if re.search(r'\bstrapless\b', bodice, re.I) and re.search(r'\bstraps\b', straps, re.I):
        if not re.search(r'\b(?:no|without)\s+(?:shoulder\s+)?straps\b', straps, re.I):
            issues['bodice'] = issues['straps'] = 'strapless bodice conflicts with straps'
    return {key: val for key, val in accepted.items() if key not in issues}, issues


def run(args):
    cases = list(prepare(args.samples))
    prompt = args.prompt.read_text(encoding='utf8')
    key = getpass.getpass('DashScope API key (hidden input): ').strip()
    if not key:
        raise ValueError('No API key entered')
    endpoint = checked_endpoint(BASE_URL)
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    args.output.mkdir(parents=True)
    path = args.output / 'results.json'
    state = dict(status='running', model=MODEL, thinking=True, thinking_budget=THINKING_BUDGET,
                 max_tokens=MAX_TOKENS, max_calls=len(cases), sent_calls=0, target_images_sent=0,
                 prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                 cases=[c for c, _ in cases], answers=[])
    write(path, state)
    try:
        for case, url in cases:
            state['sent_calls'] += 1
            state['active'] = [case['category'], case['source_index']]
            write(path, state)
            body = request_body(url, prompt, case['fragments'])
            body['model'] = MODEL
            req = request.Request(endpoint, data=json.dumps(body).encode(),
                                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
            with opener.open(req, timeout=90) as stream:
                result = read_stream(stream, key)
            answer = result.pop('final_answer')
            plan = None
            if result['finish'] == 'stop':
                try:
                    plan = parse(answer)
                except ValueError:
                    pass
            if plan is None:
                result['unparsed_answer_sha256'] = hashlib.sha256(answer.encode()).hexdigest()
            accepted, issues = validate(plan, case['fragments'])
            state['answers'].append(dict(category=case['category'], source_index=case['source_index'],
                                         generation=result, plan=plan, accepted_edits=accepted, issues=issues))
            write(path, state)
    except (error.HTTPError, error.URLError, RuntimeError, ValueError, TimeoutError) as exc:
        state.update(status='failed', error=f'HTTP {exc.code}' if isinstance(exc, error.HTTPError) else type(exc).__name__)
        write(path, state)
        print(f'Pilot stopped: {state["error"]}; no automatic retry.')
        return 1
    state.pop('active', None)
    state['status'] = 'complete'
    write(path, state)
    print(f'Results saved to {path}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('samples', 'prompt', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
