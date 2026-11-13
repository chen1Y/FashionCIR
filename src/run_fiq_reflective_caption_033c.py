"""Evaluate typed target decisions, deterministic captions, and a separate visual review."""

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import re
from urllib import error, request

from run_fiq_reflective_caption_033 import (
    BASE_URL, MODEL, MAX_TOKENS, THINKING_BUDGET, NoRedirect, checked_endpoint,
    prepare, read_stream, request_body, write,
)


VALUES = {
    'color': 'black|white|navy|blue|red|pink|green|gray|brown|beige|cream|yellow|purple|orange|light blue|dark blue|near-black|solid black',
    'sleeves': 'sleeveless|cap|short|elbow-length|three-quarter|long|present',
    'straps': 'present|absent', 'neckline': 'crew|scoop|V-neck|collared',
    'closure': 'button-front', 'pattern': 'solid|faint print|graphic print|striped|floral print',
    'length': 'above-knee|knee-length|midi|maxi', 'skirt': 'full|straight',
    'bodice': 'tank-style|strapless', 'style': 'casual|formal', 'weight': '',
}
RELATIVE = {
    'color': ('lighter|darker|brighter', VALUES['color'].removesuffix('|solid black')),
    'sleeves': ('longer|shorter', 'cap|short|elbow-length|three-quarter|long'),
    'length': ('longer|shorter', 'a knee-length hem|a calf-length hem|an ankle-length hem'),
    'style': ('more casual|more formal', 'a dress shirt|a formal blouse|a casual T-shirt'),
}
REMOVALS = {'sleeves': 'sleeveless', 'straps': 'absent', 'pattern': 'solid'}
CRITIC = (
    'Review the supplied target attribute decisions and compiled caption against only the reference garment '
    'and both edit fragments. Report concrete issues under missing, conflict, or unsupported. A comparison '
    'with a visible concrete benchmark is valid. Do not require guessing unanchored edits already noted '
    'as unresolved. Still report an omitted clear change if only its exact degree is uncertain. '
    'Keep each issue under 16 words. Return JSON only: {"missing":[],"conflict":[],"unsupported":[]}. '
    'Do not rewrite the caption. Candidate to review: '
)


def parse(text):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result

    match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text.strip(), flags=re.DOTALL)
    return json.loads(match[1] if match else text, object_pairs_hook=unique_object)


def protect_relative_sleeves(plan, fragments):
    """Reject a concrete long/short value inferred solely from a comparative."""
    if not isinstance(plan, dict) or not isinstance(plan.get('fields'), dict):
        return plan
    decision = plan['fields'].get('sleeves')
    if not isinstance(decision, dict) or set(decision) != {'op', 'value'} or decision['op'] != 'set':
        return plan
    value = decision.get('value')
    if value not in ('long', 'short'):
        return plan
    edits = ' '.join(fragments).lower()
    if not re.search(rf'\b{value}er sleeves\b', edits):
        return plan
    if re.search(rf'\b{value}(?: sleeves\b|[- ]sleeved\b)', edits):
        return plan
    if not isinstance(plan.get('unresolved', []), list):
        return plan
    safe = {'op': 'set', 'value': 'present'} if re.search(r'\b(?:has|have|with) sleeves\b', edits) else {
        'op': 'unknown', 'value': 'sleeve length has no verified benchmark',
    }
    unresolved = [*plan.get('unresolved', []), 'comparative sleeve length lacks a verified benchmark']
    return dict(plan, fields=dict(plan['fields'], sleeves=safe), unresolved=unresolved)


def compile_caption(plan):
    if (not isinstance(plan, dict) or set(plan) - {'item', 'fields', 'unresolved'}
            or plan.get('item') not in ('dress', 'top', 'shirt', 'T-shirt')
            or not isinstance(plan.get('fields'), dict)):
        raise ValueError('invalid plan envelope')
    unresolved = plan.get('unresolved', [])
    if not isinstance(unresolved, list) or any(not isinstance(x, str) or not x.strip() for x in unresolved):
        raise ValueError('invalid unresolved list')
    properties, relations, omitted = {}, {}, list(unresolved)
    for slot, decision in plan['fields'].items():
        if slot not in VALUES or not isinstance(decision, dict):
            raise ValueError(f'invalid field: {slot}')
        op, value = decision.get('op'), decision.get('value')
        if op in ('set', 'keep') and set(decision) == {'op', 'value'} and value in VALUES[slot].split('|') and value:
            properties[slot] = value
        elif op == 'remove' and set(decision) == {'op'} and slot in REMOVALS:
            properties[slot] = REMOVALS[slot]
        elif op == 'unknown' and set(decision) == {'op', 'value'} and isinstance(value, str) and value.strip():
            omitted.append(f'{slot}: {value}')
        elif op == 'relative' and set(decision) == {'op', 'value', 'benchmark'} and slot in RELATIVE:
            cues, anchors = RELATIVE[slot]
            if value not in cues.split('|') or decision['benchmark'] not in anchors.split('|'):
                raise ValueError(f'invalid relative value: {slot}')
            relations[slot] = (value, decision['benchmark'])
        else:
            raise ValueError(f'invalid decision: {slot}')

    if properties.get('bodice') == 'strapless' and properties.get('straps') == 'present':
        raise ValueError('strapless bodice conflicts with straps')
    modifiers, phrases = [], []
    for slot in ('style', 'color', 'sleeves', 'length', 'bodice'):
        if slot not in properties:
            continue
        value = properties[slot]
        if slot == 'sleeves' and value == 'present':
            phrases.append('sleeves')
            continue
        if slot == 'sleeves' and value != 'sleeveless':
            value = f'{value}-sleeved'
        if slot == 'length' and value == 'above-knee':
            value = 'above-the-knee'
        modifiers.append(value)
    if properties.get('pattern') in ('solid', 'striped'):
        if properties['pattern'] != 'solid' or properties.get('color') != 'solid black':
            modifiers.append(properties['pattern'])
    if 'closure' in properties:
        modifiers.append(properties['closure'])
    if 'neckline' in properties:
        value = properties['neckline']
        phrases.append('a collar' if value == 'collared' else f'a {value} neckline')
    if properties.get('straps') == 'present':
        phrases.append('shoulder straps')
    elif properties.get('straps') == 'absent' and 'strapless' not in modifiers:
        modifiers.append('strapless')
    if properties.get('pattern') in ('faint print', 'graphic print', 'floral print'):
        phrases.append(f'a {properties["pattern"]}')
    if 'skirt' in properties:
        phrases.append(f'a {properties["skirt"]} skirt')
    for slot, (cue, anchor) in relations.items():
        phrases.append({
            'color': f'a color {cue} than {anchor}',
            'sleeves': f'sleeves {cue} than {anchor} sleeves',
            'length': f'a hem {cue} than {anchor}',
            'style': f'a style {cue} than {anchor}',
        }[slot])
    subject = ' '.join([*modifiers, plan['item']])
    tail = (' with ' + (' and '.join(phrases) if len(phrases) < 3
                      else ', '.join(phrases[:-1]) + ', and ' + phrases[-1])) if phrases else ''
    article = 'An' if subject[0].lower() in 'aeiou' else 'A'
    return f'{article} {subject}{tail}.', omitted


def main(args):
    cases = list(prepare(args.samples))
    prompt = args.prompt.read_text(encoding='utf8')
    key = getpass.getpass('DashScope API key (hidden input): ').strip()
    if not key:
        raise ValueError('No API key entered')
    endpoint = checked_endpoint(BASE_URL)
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    args.output.mkdir(parents=True)
    path = args.output / 'results.json'
    state = dict(status='running', model=args.model, thinking=True, thinking_budget=THINKING_BUDGET,
                 max_tokens=MAX_TOKENS, max_calls=2 * len(cases), sent_calls=0, target_images_sent=0,
                 sleeve_guard=args.sleeve_guard,
                 prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), critic_prompt=CRITIC,
                 cases=[c for c, _ in cases], answers=[])
    write(path, state)

    def ask(stage, case, url, instruction):
        state['sent_calls'] += 1
        state['active'] = [stage, case['category'], case['source_index']]
        write(path, state)
        payload = request_body(url, instruction, case['fragments'])
        payload['model'] = args.model
        req = request.Request(endpoint, data=json.dumps(payload).encode(),
                              headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        with opener.open(req, timeout=90) as stream:
            result = read_stream(stream, key)
        answer = result.pop('final_answer')
        if result['finish'] == 'stop':
            try:
                return result, parse(answer)
            except ValueError:
                pass
        result['unparsed_answer_sha256'] = hashlib.sha256(answer.encode()).hexdigest()
        return result, None

    try:
        for case, url in cases:
            generation, plan = ask('plan', case, url, prompt)
            row = dict(category=case['category'], source_index=case['source_index'], generation=generation,
                       generated_plan=plan)
            state['answers'].append(row)
            if args.sleeve_guard:
                plan = protect_relative_sleeves(plan, case['fragments'])
            try:
                caption, unresolved = compile_caption(plan)
            except ValueError as exc:
                row['compile_error'] = str(exc)
                write(path, state)
                continue
            row.update(plan=plan, caption=caption, unresolved=unresolved)
            write(path, state)
        for (case, url), row in zip(cases, state['answers'], strict=True):
            if 'caption' not in row:
                continue
            review, issues = ask('critic', case, url, CRITIC + json.dumps({
                'plan': row['plan'], 'caption': row['caption'],
            }))
            keys = {'missing', 'conflict', 'unsupported'}
            valid = isinstance(issues, dict) and set(issues) == keys and all(
                isinstance(v, list) and all(isinstance(x, str) for x in v) for v in issues.values())
            row.update(review=review, issues=issues if valid else None)
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
    parser.add_argument('--model', choices=(MODEL, 'qwen3.8-flash'), default=MODEL)
    parser.add_argument('--no-sleeve-guard', dest='sleeve_guard', action='store_false')
    raise SystemExit(main(parser.parse_args()))
