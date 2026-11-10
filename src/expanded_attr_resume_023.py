"""023 merge immutable021/022 ledgers, never resend known or unknown requests."""
import hashlib,json,re,sqlite3,urllib.error,urllib.request,time
from pathlib import Path
from expanded_attr_resume_022 import reconcile,resolve as resolve021,publish_ready as publish022
from expanded_attr_schema_021 import PROMPT,request_key
from shoes_online_pilot import Client,request_body
from generate_structured_edits import atomic_write_json


def provider_code(payload):
    if not isinstance(payload,dict):return 'unavailable'
    error=payload.get('error',payload)
    value=error.get('code') if isinstance(error,dict) else None
    if not isinstance(value,str) or 'sk-' in value.lower() or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,80}',value):return 'unavailable'
    return value


class DiagnosticClient(Client):
    def ask(self,image_url,text,max_tokens):
        if len(text.encode('utf-8'))+4096>16384:raise ValueError('Prompt exceeds input bound')
        self.budget.reserve(max_tokens);body=json.dumps(request_body(image_url,text,max_tokens)).encode()
        req=urllib.request.Request(self.endpoint,data=body,method='POST',headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key});start=time.monotonic()
        try:
            with self.opener.open(req,timeout=60) as response:payload=json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:code=self.safe_text(provider_code(json.loads(exc.read(65536))))
            except Exception:code='unavailable'
            raise RuntimeError('Official API HTTP '+str(exc.code)+' code='+code+'; stopped without retry') from None
        except (urllib.error.URLError,TimeoutError) as exc:
            raise RuntimeError('Official API transport error ('+type(exc).__name__+'); stopped without retry') from None
        self.budget.account(payload.get('usage'));choice=payload['choices'][0]
        return dict(text=self.safe_text(choice['message'].get('content') or ''),usage=payload.get('usage'),model=payload.get('model'),request_id=payload.get('id'),finish_reason=choice.get('finish_reason'),seconds=time.monotonic()-start)


def resolve(root,phase):
    rows,answers,_,proof=resolve021(root,phase)
    more,second=reconcile(Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-022'))
    if set(answers)&set(more):raise ValueError('021/022 paid request collision')
    for key,value in more.items():
        value=dict(value);value['origin']=value.get('origin','').replace('021','022');value['errors']=[e.replace('unknown021_','unknown022_') for e in value['errors']];answers[key]=value
    todo={}
    for row in rows:
        k=row['request_key']
        if k not in answers:todo.setdefault(k,dict(row,request_index=len(todo)))
    counts={cat:dict(rows=sum(r['category']==cat for r in rows),missing=sum(r['category']==cat and r['request_key'] not in answers for r in rows),fallback=sum(r['category']==cat and r['request_key'] in answers and answers[r['request_key']]['status']!='ok' for r in rows)) for cat in ('shirt','toptee')}
    return rows,answers,list(todo.values()),dict(source021=proof,source022=second,categories=counts,unique_missing=len(todo))


def publish_ready(root,phase,rows,answers):
    # Existing Shirt text is copied verbatim: its saved teacher/fit hashes stay valid.
    (root/'texts').mkdir(exist_ok=True)
    for cat in ('shirt','toptee'):
        selected=[r for r in rows if r['category']==cat];p=root/'texts'/(cat+'_'+phase+'.json')
        if not selected or any(r['request_key'] not in answers for r in selected):continue
        if p.exists():
            old=json.loads(p.read_text());samples=old['samples']
            if len(samples)!=len(selected):raise ValueError('Existing text row count changed')
            for row,s in zip(selected,samples):
                a=answers[row['request_key']]
                if any(s[k]!=row[k] for k in ('source_index','reference','instruction','request_key')) or s['status']!=a['status'] or s['target_attrs']!=a['target_attrs'] or s['errors']!=a['errors']:raise ValueError('Existing immutable text conflicts with reuse')
        else:
            samples=[]
            for row in selected:
                a=answers[row['request_key']];samples.append(dict(row,status=a['status'],target_attrs=a['target_attrs'],errors=a['errors'],origin=a.get('origin','new023')))
            atomic_write_json(p,dict(version='023_same021_prompt',phase=phase,model='qwen3.7-flash',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),samples=samples,valid=sum(s['status']=='ok' for s in samples),all_rows_retained=True,targets_sent=False))
