"""Whole-train component split: no arbitrary row cap, exact-byte isolation."""
import hashlib,json
from collections import Counter


def validation_gate_state(state):
    if state.get('status') in ('failed','cancelled'):return 'blocked'
    if state.get('status')=='complete' and state.get('downstream_allowed') is True:return 'audit_required'
    return 'waiting'


def partition(ids,hashes,rows,heldout):
    if len(ids)!=len(set(ids)) or any(i not in hashes for i in ids):raise ValueError('Incomplete/duplicate gallery index')
    allowed=set(ids);parent={h:h for h in set(hashes[i] for i in ids)}
    def find(x):
        if x not in parent:parent[x]=x
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for row in rows:
        if row['candidate'] not in allowed or row['target'] not in allowed:raise ValueError('Official train query outside original gallery')
        union(hashes[row['candidate']],hashes[row['target']])
    components={h:find(h) for h in list(parent)}
    split={h:('dev' if int(hashlib.sha256(('42|full_component|'+components[h]).encode()).hexdigest()[:8],16)%4==0 else 'fit') for h in components}
    held=set(heldout);banks={s:[i for i in ids if hashes[i] not in held and split[hashes[i]]==s] for s in ('fit','dev')};canonical={}
    for s in banks:
        seen=set();canonical[s]=[]
        for i in banks[s]:
            if hashes[i] not in seen:canonical[s].append(i);seen.add(hashes[i])
    selected={'fit':[],'dev':[]};audit=[];reasons=Counter()
    for row in rows:
        a,b=hashes[row['candidate']],hashes[row['target']];why=[]
        if a in held:why.append('reference_official_heldout_byte')
        if b in held:why.append('target_official_heldout_byte')
        if a==b:why.append('reference_target_same_jpeg_ambiguity')
        if split[a]!=split[b]:raise ValueError('A query crosses component split')
        record=dict(source_index=row['source_index'],split=split[a],eligible=not why,reasons=why,reference_sha256=a,target_sha256=b)
        audit.append(record);reasons.update(why)
        if not why:selected[split[a]].append(dict(row,split020=split[a],original_row=row['source_index']))
    bytesets={s:{hashes[i] for i in banks[s]} for s in banks}
    intersections=dict(fit_dev=len(bytesets['fit']&bytesets['dev']),fit_heldout=len(bytesets['fit']&held),dev_heldout=len(bytesets['dev']&held))
    if any(intersections.values()):raise ValueError('Actual learning bank byte collision')
    return dict(banks=banks,canonical=canonical,selected=selected,audit=audit,reason_counts=dict(reasons),intersections=intersections,components=len(set(components.values())))
