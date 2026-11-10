"""015: pure-JSON support strata and saved-rank transitions; no model imports."""
import json


def selected_fields(text, wanted):
    """Decode only requested top-level values; skip unrequested containers as text."""
    decoder = json.JSONDecoder()
    def space(i):
        while i < len(text) and text[i].isspace(): i += 1
        return i
    def end(i):
        if text[i] == '"': return decoder.raw_decode(text, i)[1]
        if text[i] not in '[{':
            j = i
            while j < len(text) and text[j] not in ',}': j += 1
            return j
        stack = [text[i]]; j = i + 1
        while stack:
            if j >= len(text): raise ValueError('Unclosed JSON container')
            c = text[j]
            if c == '"': j = decoder.raw_decode(text, j)[1]; continue
            if c in '[{': stack.append(c)
            elif c in ']}':
                if stack.pop() != {']':'[', '}':'{'}[c]: raise ValueError('Invalid JSON nesting')
            j += 1
        return j
    i = space(0)
    if text[i] != '{': raise ValueError('Expected top-level JSON object')
    i += 1; result = {}; keys = set()
    while text[space(i)] != '}':
        i = space(i); key, i = decoder.raw_decode(text, i)
        if not isinstance(key, str) or key in keys: raise ValueError('Invalid/duplicate key')
        keys.add(key); i = space(i)
        if text[i] != ':': raise ValueError('Missing JSON colon')
        i = space(i + 1); j = end(i)
        if key in wanted: result[key] = json.loads(text[i:j])
        i = space(j)
        if text[i] == ',': i += 1
        elif text[i] != '}': raise ValueError('Invalid JSON separator')
    if wanted - result.keys(): raise ValueError('Missing requested fields')
    return result


def support_groups(selected_global, byte_mask, reasons, fit_count=1024):
    n = len(byte_mask)
    if any(type(v) is not bool for v in byte_mask): raise ValueError('Nonboolean eligibility')
    if len(selected_global) != len(set(selected_global)): raise ValueError('Duplicate selected row')
    selected = {v-fit_count for v in selected_global}
    if any(type(v) is not int or not 0 <= v < n for v in selected): raise ValueError('Global/local mapping invalid')
    eligible = {i for i,v in enumerate(byte_mask) if v}
    if not selected <= eligible: raise ValueError('Supported row violates009 byte mask')
    expected = {i for i in eligible if not reasons[str(i+fit_count)]}
    if selected != expected: raise ValueError('Support and saved reasons disagree')
    return dict(all=list(range(n)), supported=sorted(selected),
        byte_eligible_unsupported=sorted(eligible-selected), byte_ineligible=sorted(set(range(n))-eligible))


def recall(ranks):
    return {f'r{k}': 100*sum(r <= k for r in ranks)/len(ranks) if ranks else None for k in (1,10,50)}


def transitions(reference, candidate, indices, k):
    gains=[]; losses=[]; both=[]; neither=[]
    for i in indices:
        old,new = reference[i] <= k, candidate[i] <= k
        (both if old and new else losses if old else gains if new else neither).append(i)
    return dict(queries=len(indices), gains=len(gains), losses=len(losses), both_hit=len(both), neither_hit=len(neither),
        net_queries=len(gains)-len(losses), delta_pp=100*(len(gains)-len(losses))/len(indices) if indices else None,
        gained_query_indices=gains, lost_query_indices=losses)


COMPARISONS = [('head011','fixed05'), ('armA','head011'), ('armB','head011'),
    ('armB','armA'), ('armA','shuffleA'), ('armB','shuffleB')]


def audit_ranks(ranks, groups, gallery_size):
    n = len(ranks['fixed05'])
    if not n or any(len(v)!=n or any(type(r) is not int or not 1 <= r < gallery_size for r in v) for v in ranks.values()):
        raise ValueError('Incomplete or invalid rank vectors')
    pieces = [groups[k] for k in ('supported','byte_eligible_unsupported','byte_ineligible')]
    flat = [i for piece in pieces for i in piece]
    if sorted(flat)!=list(range(n)) or len(flat)!=len(set(flat)) or groups['all']!=list(range(n)):
        raise ValueError('Strata do not partition all queries')
    base=ranks['fixed05']
    for name,row in ranks.items():
        if any((a>20 and b!=a) or ((a<=20)!=(b<=20)) or ((a<=50)!=(b<=50)) for a,b in zip(base,row)):
            raise ValueError('Saved head changed tail/top20 membership')
    def section(indices):
        return dict(queries=len(indices), query_indices=indices,
            metrics={name:recall([row[i] for i in indices]) for name,row in ranks.items()},
            transitions={new+'_minus_'+ref:{f'r{k}':transitions(ranks[ref],ranks[new],indices,k) for k in (1,10,50)}
                for new,ref in COMPARISONS})
    output={}
    for name,indices in groups.items():
        output[name]=section(indices)
        output[name]['fixed05_rank_buckets']={
            bucket:section([i for i in indices if predicate(base[i])])
            for bucket,predicate in [('top1',lambda r:r==1),('rank2_20',lambda r:2<=r<=20),('outside20',lambda r:r>20)]}
    return output
