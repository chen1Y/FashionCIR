"""031 fixed text-input transplant protocol; no model, API, or fitting."""
from fiq_text_branch_math_027 import corrected_modification, textual_query

METHODS = (
    'native_dqu_caption_corrected',
    'native_dqu_raw_mod',
    '029d_raw_original',
    '029d_native_input_beta0',
    '029d_native_input',
    'native_anchor_029d_attributes',
)
COMPARISONS = (
    ('029d_native_input', '029d_raw_original'),
    ('029d_native_input', 'native_dqu_caption_corrected'),
    ('029d_native_input', '029d_native_input_beta0'),
    ('native_anchor_029d_attributes', 'native_dqu_caption_corrected'),
    ('native_anchor_029d_attributes', '029d_native_input'),
)


def native_queries(records, original, reference_captions, correction):
    result = []
    for record in records:
        index = record['source_index']
        if type(index) is not int or index < 0 or index >= len(original):
            raise ValueError('031 original query index mismatch')
        old = original[index]
        if any(record[name] != old[name] for name in ('candidate', 'target')):
            raise ValueError('031 original query reference/target mismatch')
        caption = reference_captions[record['candidate']]
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError('031 missing reference caption; never invent replacement')
        modification = corrected_modification(old['captions'], correction)
        result.append(dict(source_index=index, candidate=record['candidate'],
                           reference_caption=caption, corrected_modification=modification,
                           native_text=textual_query(caption, modification)))
    return result


def metrics(ranks):
    if not ranks or any(type(rank) is not int or rank < 1 for rank in ranks):
        raise ValueError('031 invalid ranks')
    return dict(queries=len(ranks), **{'r' + str(k): 100 * sum(rank <= k for rank in ranks) / len(ranks)
                                     for k in (1, 10, 50)})


def transitions(after, before, k):
    if len(after) != len(before) or not after:
        raise ValueError('031 unmatched comparison denominator')
    counts = dict(gains=0, losses=0, both=0, neither=0)
    for a, b in zip(after, before):
        name = 'both' if a <= k and b <= k else ('gains' if a <= k else ('losses' if b <= k else 'neither'))
        counts[name] += 1
    return dict(counts, queries=len(after), net_pp=100 * (counts['gains'] - counts['losses']) / len(after))
