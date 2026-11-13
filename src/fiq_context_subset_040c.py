"""Bind 040C generated contexts to the exact 040A paired subset."""

from collections import Counter

from run_fiq_native_caption_031 import read, sha
from run_fiq_pixel_free_038 import CATEGORIES


def load_pilot(root, plan):
    pilot = plan['pilot']
    cases_file, generated_file = root / 'cases.json', root / 'generation.json'
    if sha(cases_file) != pilot['cases_sha256'] or sha(generated_file) != pilot['generation_sha256']:
        raise ValueError('040C input manifest changed')
    cases, generated = read(cases_file), read(generated_file)
    limit = pilot['max_requests']
    if (generated['status'] != 'complete' or generated['model'] != pilot['model']
            or generated['submitted'] != limit or generated['max_calls'] != limit
            or generated['target_images_sent'] != 0 or len(generated['answers']) != limit
            or len(cases) != limit or generated['cases_sha256'] != pilot['cases_sha256']
            or generated['prompt_sha256'] != pilot['prompt_sha256']
            or generated['settings'] != pilot['generation_settings']):
        raise ValueError('040C generation protocol differs from plan')
    expected = {(cat, split, offset)
                for cat in CATEGORIES for split in ('train', 'val')
                for offset in pilot['offsets'][cat][split]}
    result = {}
    for case, answer in zip(cases, generated['answers']):
        key = (case['category'], case['split'], case['offset'])
        if key in result or 'error' in answer or answer['model'] != pilot['model']:
            raise ValueError('040C response missing, duplicated or from another model')
        result[key] = (case, answer)
    if set(result) != expected:
        raise ValueError('040C query keys differ from sampled offsets')
    return result


def paired_rows(rows, offsets, category, split, image_hashes, responses, tokenizer):
    original, generated, audit = [], [], Counter()
    for offset in offsets:
        row = rows[offset]
        case, output = responses[(category, split, offset)]
        if (row['source_index'] != case['source_index'] or row['candidate'] != case['candidate']
                or image_hashes[row['candidate']] != case['reference_sha256']):
            raise ValueError('040C source row or reference-image identity changed')
        original.append(row)
        answer = output['answer']
        if (output['finish'] != 'stop' or not output['thinking_present'] or output['flags']
                or not isinstance(answer, dict)):
            audit['generation_fallback'] += 1
            generated.append(row)
            continue
        main, kept, mod = (answer[name] for name in ('final_text', 'Tkep', 'Tmod'))
        drop = ', '.join(answer['drop_attrs'])
        if main != f'{kept}, and {mod}.':
            raise ValueError('040C final sentence is not the prescribed composition')
        if len(tokenizer.encode(main)) > 75 or len(tokenizer.encode(drop)) > 75:
            audit['token_budget_fallback'] += 1
            generated.append(row)
            continue
        audit['qwen_used'] += 1
        audit['qwen_empty_drop'] += not drop
        generated.append(dict(row, main_context_text=main, kept_context=kept, drop_attrs=drop))
    return original, generated, dict(audit)
