"""Read-only reconciliation: known reuse, unknown fallback, unsent stays eligible."""
import hashlib,json,sqlite3
from pathlib import Path
from expanded_attr_schema_021 import PROMPT,request_key
from expanded_attr_metadata import generation_rows
from generate_structured_edits import atomic_write_json

EMPTY={k:'' for k in ('color','shape','detail','texture')}


def transport_fallback_allowed(message,count):
    return message.startswith('Official API transport error') and 1<=count<=3


def fallback(reason,origin):
    return dict(status='fallback',target_attrs=dict(EMPTY),errors=[reason],origin=origin,response=None)


def reconcile(parent):
    answers={};proof=[];unsent=[]
    for d in sorted((parent/'generation').iterdir()):
        if not d.is_dir():continue
        p=json.loads((d/'protocol.json').read_text())
        if p['model']!='qwen3.7-flash' or p['prompt_sha256']!=hashlib.sha256(PROMPT.encode()).hexdigest():raise ValueError('Different parent prompt/model')
        rows={r['request_key']:r for r in p['rows']};samples=json.loads((d/'responses.json').read_text())
        db=sqlite3.connect('file:'+str(d/'requests.sqlite3')+'?mode=ro',uri=True)
        ledger={i:(state,response) for i,state,response in db.execute('SELECT idx,state,response FROM requests')};db.close();seen=set()
        if any(state not in ('known','unknown') for state,response in ledger.values()):raise ValueError('Unsettled parent request')
        for s in samples:
            k=s['request_key'];r=rows[k];i=s['request_index']
            if i!=r['request_index'] or k!=request_key(r['reference_sha256'],r['instruction']):raise ValueError('Parent request identity conflict')
            state,response=ledger.get(i,(None,None));seen.add(i)
            if s['response'] is not None:
                if state!='known' or json.loads(response)!=s['response']:raise ValueError('Known response differs from durable ledger')
                value=dict(s,origin='reused021:'+d.name)
            elif state=='unknown':value=fallback('unknown021_no_resend','unknown021:'+d.name)
            elif state is None:
                unsent.append(k);continue
            else:raise ValueError('Known request lost its response')
            if k in answers and answers[k]!=value:raise ValueError('Conflicting duplicate request')
            answers[k]=value
        if set(ledger)-seen:raise ValueError('Durable request absent from parent responses')
        proof.append(dict(batch=d.name,known=sum(s=='known' for s,r in ledger.values()),unknown=sum(s=='unknown' for s,r in ledger.values())))
    return answers,dict(batches=proof,unsent_keys=unsent)


def resolve(root,phase,parent=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-021')):
    rows=generation_rows(root,phase);answers,proof=reconcile(parent)
    old=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-020/generation/train_000');p=json.loads((old/'protocol.json').read_text());lookup={r['request_key']:r for r in p['rows']}
    blocked={(lookup[s['request_key']]['reference_sha256'],lookup[s['request_key']]['instruction']) for s in json.loads((old/'responses.json').read_text())}
    tasks={}
    for row in rows:
        k=request_key(row['reference_sha256'],row['instruction']);row['request_key']=k
        if row['blocked_old_unknown'] or (row['reference_sha256'],row['instruction']) in blocked:answers[k]=fallback('historical_pair_blocked_no_resend','historical_block')
        if k not in answers:tasks.setdefault(k,dict(row,request_index=len(tasks)))
    counts={cat:dict(rows=sum(r['category']==cat for r in rows),missing=sum(r['category']==cat and r['request_key'] not in answers for r in rows),fallback=sum(r['category']==cat and r['request_key'] in answers and answers[r['request_key']]['status']!='ok' for r in rows)) for cat in ('shirt','toptee')}
    return rows,answers,list(tasks.values()),dict(proof=proof,categories=counts,unique_missing=len(tasks))


def publish_ready(root,phase,rows,answers):
    directory=root/'texts';directory.mkdir(exist_ok=True)
    for cat in ('shirt','toptee'):
        selected=[r for r in rows if r['category']==cat]
        if any(r['request_key'] not in answers for r in selected):continue
        samples=[]
        for row in selected:
            s=answers[row['request_key']];samples.append(dict(row,status=s['status'],target_attrs=s['target_attrs'],errors=s['errors'],origin=s.get('origin','new022')))
        payload=dict(version='022_continuation_same021_prompt',phase=phase,model='qwen3.7-flash',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),samples=samples,valid=sum(s['status']=='ok' for s in samples),all_rows_retained=True,targets_sent=False)
        p=directory/(cat+'_'+phase+'.json')
        if p.exists():
            if json.loads(p.read_text())!=payload:raise ValueError('Existing ready text changed')
        else:atomic_write_json(p,payload)
