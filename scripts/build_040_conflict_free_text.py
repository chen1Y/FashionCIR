"""Prepare offline FashionIQ query text for 040 without exposing target images."""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


SLOTS = ('color', 'shape', 'detail', 'texture')
NEGATION = re.compile(r"\b(?:no|not|without|never|lack\s+of)\b|\bnon(?:[-\u2010-\u2015]|\s+)|"
                      r"\b(?:isn|aren|wasn|weren|doesn|don|didn|hasn|haven|hadn|can|couldn|"
                      r"won|wouldn|shouldn|mustn|ain|mightn)(?:['’]|\s+)t\b", re.I)
WORDS = re.compile(r"[a-z0-9]+")
JOINERS = frozenset('a an and are as but for from has have is it of or the than to with'.split())
GENERIC = frozenset('color colour colored shade tone style cut fit look length shape silhouette pattern design detail sleeve shirt top tee dress woman man image picture reference'.split())
WORD_ALIASES = {'sleeves': 'sleeve', 'sleeved': 'sleeve', 'stripes': 'stripe', 'striped': 'stripe',
                'buttons': 'button', 'buttoned': 'button', 'prints': 'print', 'printed': 'print',
                'colours': 'color', 'colour': 'color', 'grey': 'gray', 'grayer': 'gray'}
POSITIVE = (
    (r'\b(?:has\s+|with\s+)?(?:no|without)\s+(?:any\s+)?sleeves?\b', 'sleeveless'),
    (r'\b(?:is\s+)?not\s+sleeved\b', 'sleeveless'),
    (r'\b(?:has\s+|with\s+)?(?:no|without)\s+(?:any\s+)?(?:stripes?|patterns?|prints?)\b', 'solid plain'),
    (r'\b(?:is\s+)?not\s+(?:striped|patterned|printed)\b', 'solid plain'),
    (r'\bnon(?:[-\u2010-\u2015]|\s+)striped\b', 'solid plain'),
    (r'\bless\s+casual\b', 'more formal'),
    (r'\bless\s+formal\b', 'more casual'),
    (r'\bnot\s+as\s+shiny\b', 'more matte'),
)
RELATIVE = {
    'color': re.compile(r'\b(?:lighter|darker|brighter|paler)\b(?:\s+(?:in\s+)?(?:color|colour|tone|shade))?', re.I),
    'shape': re.compile(r'\b(?:longer|shorter)\s+(?:sleeves?|hem(?:line)?|skirt|length|cut)\b|'
                        r'\b(?:more|less)\s+(?:formal|casual|fitted|loose)\b', re.I),
}
COLORS = re.compile(r'\b(?:(?:light|dark|bright|pale)\s+)?'
                    r'(?:black|white|gr[ae]y|red|pink|orange|yellow|green|blue|purple|brown|beige|navy|burgundy|cream|teal|gold|silver)\b', re.I)
GARMENT_COLOR = re.compile(COLORS.pattern + r'(?=\s+(?:t\s*-?\s*)?(?:dress|shirt|tee|top)\b)', re.I)
PART_COLOR = re.compile(r'\b(?:trim|collar|sleeves?|cuffs?|straps?|buttons?|pockets?|logos?|graphic|'
                        r'print|writing|letters?|words?|stripes?|lining)\b', re.I)
REFERENCE_POINTER = re.compile(r'\b(?:same (?:as|like)|unchanged|reference|original (?:image|picture))\b', re.I)
PUNCTUATION = str.maketrans({c: ' ' for c in string.punctuation})


def words(text):
    return set(WORDS.findall(text.lower()))


def concepts(text):
    tokens = {WORD_ALIASES.get(w, w) for w in words(text)}
    return tokens - JOINERS


def positive_clauses(text):
    text = ' '.join(text.strip().split())
    for pattern, replacement in POSITIVE:
        text = re.sub(pattern, replacement, text, flags=re.I)
    clauses = re.split(r'\s+(?:and|but)\s+|[;,]', text, flags=re.I)
    clauses = [re.sub(r'\s+(?:instead of|rather than)\s+.+$', '', s, flags=re.I) for s in clauses]
    kept = [s.strip(' .') for s in clauses if s.strip(' .') and not NEGATION.search(s)]
    return kept, len(clauses) - len(kept)


def strip_context(caption, conflicts):
    # Remove whole phrases before individual attribute words. Grammatical
    # connector words are not attribute conflicts by themselves.
    context = caption.lower()
    for phrase in sorted(conflicts, key=len, reverse=True):
        parts = WORDS.findall(phrase.lower())
        if parts:
            pattern = r'\b' + r'[\s-]+'.join(re.escape(w) for w in parts) + r'\b'
            context = re.sub(pattern, ' ', context)
    blocked = set().union(*(concepts(x) for x in conflicts))
    clauses, _ = positive_clauses(context)
    kept = []
    for clause in clauses:
        tokens = [word for word in WORDS.findall(clause) if WORD_ALIASES.get(word, word) not in blocked]
        while tokens and tokens[-1] in JOINERS:
            tokens.pop()
        if tokens:
            kept.append(' '.join(tokens))
    return ' and '.join(kept)


def explicit_colors(mod):
    return [match.group() for match in COLORS.finditer(mod)
            if not re.search(r'\b(?:than|instead of)\s+$', mod[:match.start()])]


def garment_mod_clauses(clauses):
    heads = [re.split(r'\s+with\s+', clause, maxsplit=1)[0] for clause in clauses]
    return [head for head in heads if not PART_COLOR.search(head)]


def caption_color_conflict(caption, clauses):
    old = {match.group().lower() for match in GARMENT_COLOR.finditer(caption)}
    if len(old) != 1:
        return ''
    eligible = garment_mod_clauses(clauses)
    desired = [concepts(v) for text in eligible for v in explicit_colors(text)]
    relative = any(re.search(r'\b(?:lighter|darker|brighter|paler)\b', text)
                   and not re.search(r'\b(?:weight|fabric|material)\b', text) for text in eligible)
    value = old.pop()
    return value if (desired or relative) and not any(concepts(value) <= new for new in desired) else ''


def slot_phrases(raw, slot, mod, blocked):
    phrases, _ = positive_clauses(raw)
    # Preserve explicit direction for comparative queries. Legacy absolute
    # guesses for that axis have no support in the modification instruction.
    comparisons = []
    for match in RELATIVE.get(slot, re.compile(r'(?!)')).finditer(mod):
        value = match.group()
        if slot == 'color' and re.match(r'\s+(?:weight|fabric|material)\b', mod[match.end():]):
            continue
        comparisons.append(value)
    relative_words = set().union(*(concepts(x) for x in comparisons))
    anchors = concepts(mod)
    candidates = list(comparisons)
    if slot == 'color':
        candidates.extend(v for head in garment_mod_clauses(mod.split(' and ')) for v in explicit_colors(head))
    if slot == 'shape' and re.search(r'\bsleeveless\b', mod):
        candidates.append('sleeveless')
    if slot == 'texture' and re.search(r'\bsolid plain\b', mod):
        candidates.append('solid plain')
    for phrase in phrases:
        p = concepts(phrase)
        specific = p - GENERIC
        grounded = bool(specific) and specific <= (anchors - GENERIC)
        if not grounded:
            continue
        if comparisons and (slot == 'color' or (p & relative_words) or
                            ('sleeve' in relative_words and 'sleeve' in p)):
            continue
        candidates.append(phrase)
    cleaned, seen = [], set()
    for value in candidates:
        # A positive sleeve state can avoid repeating a shared garment noun
        # when the old conflict used the literal noun "sleeves".
        if words(value) & blocked:
            value = re.sub(r'\b(shorter|longer|short|long)\s+sleeves?\b', r'\1-sleeved', value)
        key = frozenset(concepts(value) - GENERIC)
        if (key and key not in seen and not NEGATION.search(value) and not words(value) & blocked
                and not REFERENCE_POINTER.search(value) and not re.search(r'\bsame\b', value)):
            cleaned.append(value)
            seen.add(key)
    return cleaned


def prepare(record, ref_caption, corrected, tokenizer):
    drop = record['drop_attrs']
    conflicts = [p.strip() for p in re.split('[,;]', drop) if p.strip()]
    clauses, removed = positive_clauses(corrected)
    flags = []
    extra = caption_color_conflict(ref_caption, clauses)
    if extra and not concepts(extra) <= set().union(*(concepts(p) for p in conflicts)):
        conflicts.append(extra)
        drop = ', '.join(conflicts)
        flags.append('caption_grounded_color_conflict')
    blocked = set().union(*(words(x) for x in conflicts))
    if removed:
        flags.append('removed_unsupported_negation_or_empty_clause')
    mod = ' and '.join(clauses)
    if len(tokenizer.encode(mod)) > 75:
        flags.append('modification_token_budget')
        while len(tokenizer.encode(mod)) > 75:
            mod = ' '.join(mod.split()[:-1])
    if not mod:
        mod = {'dress': 'dress', 'shirt': 'shirt', 'toptee': 'top'}[record['category']]
        flags.append('empty_modification_category_fallback')
    kept = strip_context(ref_caption, conflicts)
    main = f'{kept}, but {mod}' if kept else mod
    if len(tokenizer.encode(main)) > 75:
        flags.append('reference_context_token_budget')
        while kept and len(tokenizer.encode(main)) > 75:
            kept = ' '.join(kept.split()[:-1])
            main = f'{kept}, but {mod}' if kept else mod
    if (words(mod) - JOINERS) & (blocked - JOINERS):
        flags.append('modification_repeats_drop_word')
    attrs = {}
    used = set()
    slot_budget = 77 - 2 - 4 - sum(len(tokenizer.encode(s.title() + ':')) for s in SLOTS)
    for k, slot in enumerate(SLOTS):
        values = slot_phrases(record['target_attrs'].get(slot, ''), slot, mod, blocked)
        cap = slot_budget // 4 + (k < slot_budget % 4)
        picked = []
        for value in values:
            key = frozenset(concepts(value))
            if key in used:
                continue
            if len(tokenizer.encode(', '.join(picked + [value]))) <= cap:
                picked.append(value)
                used.add(key)
            else:
                flags.append(f'{slot}_token_budget')
        attrs[slot] = ', '.join(picked)
    if any(NEGATION.search(s) for s in (mod, kept, main, *attrs.values())):
        raise ValueError('040 negation survived sanitization')
    if (words(kept) - JOINERS) & (blocked - JOINERS) or any(words(s) & blocked for s in attrs.values()):
        raise ValueError('040 conflict survived in reference context or slots')
    return dict(source_index=record['source_index'], candidate=record['candidate'],
                mod_clean=mod, drop_attrs=drop, kept_context=kept, main_context_text=main,
                target_attrs=attrs, slot_valid={k: bool(v) for k, v in attrs.items()}, audit_flags=flags)


def val_rows(category):
    from attr_transfer_data import official_inputs
    from run_029d_gate_refresh import inputs_029

    if category == 'dress':
        return official_inputs(category)[0]
    return inputs_029(category, 'val')[0]['records']


def main(root):
    from open_clip import get_tokenizer
    from run_fiq_native_caption_031 import DATA, read, sha, write
    from run_fiq_pixel_free_038 import CATEGORIES, COUNTS, bind_drops
    from fiq_text_branch_math_027 import corrected_modification

    plan = read(root / 'plan.json')
    tokenizer = get_tokenizer('ViT-H-14')
    output = root / 'texts'
    output.mkdir()
    audit, examples = {}, []
    picks = (('dress', 'train', 0), ('shirt', 'train', 0), ('toptee', 'train', 0),
             ('dress', 'val', 0), ('toptee', 'val', 0))
    for category in CATEGORIES:
        spec = plan['inputs'][category]
        metadata = root / 'metadata' / f'{category}.json'
        if sha(metadata) != spec['metadata_sha256']:
            raise ValueError('040 metadata changed')
        for split, rows in (('train', read(metadata)['records']), ('val', val_rows(category))):
            bound = bind_drops(category, split, rows, spec[f'{split}_archive_sha256'])
            names = (f'cap.{category}.{split}.json', f'image_captions_{category}_{split}.json',
                     f'correction_dict_{category}.json')
            paths = [DATA / 'captions' / name for name in names]
            sources = {str(path): sha(path) for path in paths}
            originals, captions, correction = [read(path) for path in paths]
            count = COUNTS[category][0 if split == 'train' else 1]
            if len(bound) != count:
                raise ValueError('040 denominator changed')
            dest = output / f'{category}_{split}.jsonl'
            counters = Counter()
            with dest.open('w', encoding='utf-8') as file:
                for offset, row in enumerate(bound):
                    original = originals[row['source_index']]
                    raw = corrected_modification(original['captions'], {})
                    raw_fragments = [' '.join(fragment.lower().translate(PUNCTUATION).split())
                                     for fragment in original['captions']]
                    comparable = ' '.join(row['modification_text'].lower().translate(PUNCTUATION).split())
                    permitted = {raw.strip(), ' and '.join(filter(None, raw_fragments))}
                    if any(row[k] != original[k] for k in ('candidate', 'target')) or comparable not in permitted:
                        raise ValueError(f"040 original query identity changed: {category} {split} {row['source_index']}")
                    ref_caption = captions.get(row['candidate'], '')
                    if not isinstance(ref_caption, str):
                        raise ValueError('040 caption must be a string')
                    corrected = corrected_modification(original['captions'], correction)
                    if not raw_fragments[0]:
                        corrected = corrected.removeprefix(' and ')
                    if not raw_fragments[1]:
                        corrected = corrected.removesuffix(' and ')
                    record = prepare(row, ref_caption, corrected, tokenizer)
                    file.write(json.dumps(record, ensure_ascii=False) + '\n')
                    counters.update(record['audit_flags'])
                    counters.update(k for k, v in record['slot_valid'].items() if v)
                    counters.update(dict(reference_caption=bool(ref_caption), kept_context=bool(record['kept_context']),
                                         any_slot=any(record['slot_valid'].values()), drop_active=bool(record['drop_attrs']),
                                         spelling_changed=corrected != raw, positive_cleanup=record['mod_clean'] != corrected))
                    if (category, split, offset) in picks:
                        examples.append(dict(category=category, split=split, original_mod=raw,
                                             reference_caption=ref_caption, **record))
            audit[f'{category}_{split}'] = dict(rows=count, counts=dict(counters), source_sha256=sources,
                                               path=dest.name, sha256=sha(dest))
    write(output / 'inventory.json', dict(splits=audit, examples=examples, slot_source='039A archive; 033e pilot only'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
