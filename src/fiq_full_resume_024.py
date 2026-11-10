"""Same021 online prompt; complete original query manifests and durable response reuse."""
import hashlib,json,sqlite3
from pathlib import Path
from expanded_attr_resume_022 import reconcile,fallback
from expanded_attr_schema_021 import PROMPT,request_key
from fiq_full_metadata_024 import CATS,DATA,PRIMARY
from generate_structured_edits import atomic_write_json


def resolve(root,phase):
    answers={};proof=[]
    for parent in map(Path,('/data/8T1/lcy/CIRData/runs/fiq-expanded-021','/data/8T1/lcy/CIRData/runs/fiq-expanded-022','/data/8T1/lcy/CIRData/runs/fiq-expanded-023')):
        found,p=reconcile(parent)
        if set(found)&set(answers):raise ValueError('Duplicate prior paid request across stages')
        for k,s in found.items():
            v=dict(s);v['origin']='reused:'+parent.name+':'+s.get('origin','');answers[k]=v
        proof.append(dict(parent=str(parent),ledger_proof=p))
    old=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-020/generation/train_000');p=json.loads((old/'protocol.json').read_text());lookup={r['request_key']:r for r in p['rows']};blocked={(lookup[s['request_key']]['reference_sha256'],lookup[s['request_key']]['instruction']) for s in json.loads((old/'responses.json').read_text())}
    rows=[];counts={};tasks={}
    for cat in CATS:
        m=json.loads((root/'metadata'/(cat+'_'+phase+'.json')).read_text());legacy=set();states={}
        if cat=='dress':
            p=DATA/'captions'/('structured_edits_dress_'+phase+'_qwen37flash.json')
            if not p.exists():raise ValueError('Legacy Dress response provenance unavailable')
            for s in json.loads(p.read_text())['samples']:
                if s.get('status')!='ok':legacy.add((s['candidate'],s['modification_text']))
        else:
            p=PRIMARY/cat/'requests.sqlite3';db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
            for i,state in db.execute('select idx,state from requests where split=?',('train' if phase=='train' else 'val',)):
                if state in ('unknown','reserved','metered'):states[i]=state
            db.close()
        for i,r in enumerate(m['records']):
            sha=m['image_sha256'][r['candidate']];k=request_key(sha,r['modification_text']);row=dict(category=cat,index=i,phase=phase,source_index=r['source_index'],reference=r['candidate'],instruction=r['modification_text'],reference_path=str(DATA/'resized_image'/(r['candidate']+'.jpg')),reference_sha256=sha,request_key=k)
            if (sha,r['modification_text']) in blocked or r['source_index'] in states or (r['candidate'],r['modification_text']) in legacy:answers[k]=fallback('historical_unknown_or_blocked_no_resend','historical_block')
            rows.append(row)
            if k not in answers:tasks.setdefault(k,dict(row,request_index=len(tasks)))
        selected=[r for r in rows if r['category']==cat];counts[cat]=dict(rows=len(selected),missing=sum(r['request_key'] not in answers for r in selected),fallback=sum(r['request_key'] in answers and answers[r['request_key']]['status']!='ok' for r in selected))
    return rows,answers,list(tasks.values()),dict(prior=proof,categories=counts,unique_missing=len(tasks))


def publish_ready(root,phase,rows,answers):
    directory=root/'texts';directory.mkdir(exist_ok=True)
    for cat in CATS:
        selected=[r for r in rows if r['category']==cat]
        if not selected or any(r['request_key'] not in answers for r in selected):continue
        samples=[]
        for r in selected:
            s=answers[r['request_key']];samples.append(dict(r,status=s['status'],target_attrs=s['target_attrs'],errors=s['errors'],origin=s.get('origin','new024')))
        payload=dict(version='024_full_same021_prompt',phase=phase,model='qwen3.7-flash',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),samples=samples,valid=sum(s['status']=='ok' for s in samples),all_rows_retained=True,targets_sent=False)
        p=directory/(cat+'_'+phase+'.json')
        if p.exists():
            if json.loads(p.read_text())!=payload:raise ValueError('Full text file already exists with different data')
        else:atomic_write_json(p,payload)
        if phase=='train':
            m=json.loads((root/'metadata'/(cat+'_train.json')).read_text());by_index={s['source_index']:s for s in samples};allrows=[]
            for r,audit in zip(m['all_original_records'],m['supervision_audit']):
                s=by_index.get(r['source_index']);allrows.append(dict(r,eligible=audit['eligible'],reasons=audit['reasons'],generation=s if s else dict(status='not_requested_excluded_from_supervision',target_attrs={},errors=audit['reasons'])))
            destination=directory/(cat+'_train_all_original_records.json');allpayload=dict(original_queries=m['original_train_rows'],fit_dev_queries=len(samples),all_original_records=allrows)
            if destination.exists():
                if json.loads(destination.read_text())!=allpayload:raise ValueError('Full original query audit changed')
            else:atomic_write_json(destination,allpayload)
