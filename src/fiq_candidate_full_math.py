"""Predeclared failure controls and byte-bound historical request reuse."""
import hashlib
import json
import random


def pair_key(reference_sha, candidate_sha, instruction):
    return hashlib.sha256(json.dumps([reference_sha,candidate_sha,instruction],ensure_ascii=False).encode()).hexdigest()


def historical_action(entries, prompt_sha, model):
    # A returned invalid JSON is still paid: reuse its raw response, never repair it here.
    for entry in entries:
        response = entry.get('response')
        if (entry['state']=='known' and isinstance(response,dict)
                and entry['prompt_sha256']==prompt_sha and entry['model']==model):
            return dict(action='reuse',origin=entry['origin'],response=response)
    if entries:
        return dict(action='blocked',reason='Prior unknown or non-reusable paid request; no resending',
            origins=[entry['origin'] for entry in entries])
    return dict(action='new')


def rank_after(query, judgments, shuffled=False, neutral=False):
    if len(judgments)!=20:
        raise ValueError('Require all20 candidate positions, including explicit failures')
    baseline = query['baseline_target_rank']
    if not neutral and any(j is None for j in judgments):
        return baseline
    valid = [i for i,j in enumerate(judgments) if j is not None]
    correction = [0.0]*20
    values = [judgments[i]['confidence']*(judgments[i]['overall_score']/4-.5) for i in valid]
    if shuffled:
        random.Random(42+query['query_index']).shuffle(values)
    # Shuffle only valid positions: failures retain zero correction in both controls.
    for i,value in zip(valid,values):
        correction[i] = value
    order = sorted(range(20),key=lambda i:-(query['candidates'][i]['similarity']+.03*correction[i]))
    for rank,index in enumerate(order,1):
        if query['candidates'][index]['candidate_id']==query['gold_target_id']:
            return rank
    return baseline


def schema_guard(samples):
    returned = [s for s in samples if s.get('response') is not None]
    return len(returned)>=16 and sum(s['status']=='ok' for s in returned)/len(returned)<.9
