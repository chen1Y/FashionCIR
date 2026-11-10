"""019 paid reference-only clean-slot generation, durable ledgers, no paid retries."""
import argparse,concurrent.futures,hashlib,json,os,resource,sqlite3,threading,time
from pathlib import Path
from attr_clean_schema import prompt,validate,request_key,PROMPT
from cir_cumulative_budget import CumulativeBudget
from generate_shoes_online import Ledger,RateLimiter
from shoes_online_pilot import Client,BASE_URL,reference_data_url
from generate_structured_edits import atomic_write_json,extract_json_object
from experiment_storage import require_space
from fashioniq_noise import file_sha
import open_clip

BATCH='clean_attr_slots_019_20261028'

def main():
 p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);p.add_argument('--key-fd',type=int,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 r=a.research;control=r/'attr_clean_019_control';plan=json.loads((control/'generation_plan.json').read_text());root=Path(__file__).resolve().parents[1]
 for name,h in plan['source_hashes'].items():
  if file_sha(root/name)!=h:raise ValueError('019 sealed source changed')
 output=r/'clean_attr_generation_019'
 if output.exists():raise FileExistsError('Inspect019 existing outputs, no blind restart')
 require_space(r);output.mkdir();atomic_write_json(output/'protocol.json',plan)
 # No imported history, no reset: atomic reservation inside the SAME300CNY ledger.
 budget=CumulativeBudget('/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3')
 if any(x['state']!='settled' for x in budget.summary()['batches']):raise ValueError('Prior paid batch still in flight')
 budget.reserve(BATCH,1.5,file_sha(output/'protocol.json'));atomic_write_json(output/'global_before.json',budget.summary());budget.db.close()
 with os.fdopen(a.key_fd) as f:key=f.read(4096).strip()
 if not key:raise ValueError('Empty hidden credential')
 manifest=json.loads((control/'manifest.json').read_text())
 if file_sha(control/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Manifest changed')
 config=dict(id=BATCH,model='qwen3.7-flash',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),max_tokens=512,api_retries=0,manifest_sha256=plan['manifest_sha256'])
 ledger=Ledger(output/'requests.sqlite3',config,max_cny=1.5,max_calls=320);stop=threading.Event();rate=RateLimiter(60,stop);tokenizer=open_clip.get_tokenizer('ViT-H-14');local=threading.local();samples={};lock=threading.Lock()
 tasks=manifest['unique_requests'];start=time.monotonic()
 def process(task):
  if stop.is_set():return None
  require_space(r)
  if (control/'cancel.request').exists():stop.set();return None
  sample=dict(request_index=task['request_index'],request_key=task['request_key'],status='fallback',errors=[],response=None,target_attrs={k:'' for k in ('color','shape','detail','texture')})
  try:
   if not hasattr(local,'client'):local.client=Client(key,BASE_URL,ledger)
   path=Path(task['reference_path'])
   if file_sha(path)!=task['reference_sha256'] or request_key(task['reference_sha256'],task['instruction'])!=task['request_key']:raise ValueError('Reference/query binding changed')
   ledger.begin('clean_slots',task['request_index'],'reference_only_clean_slots_v1');rate.wait()
   answer=local.client.ask(reference_data_url(path),prompt(task['instruction']),512);ledger.finish(answer);sample['response']=answer
   if answer['finish_reason']!='stop':raise ValueError('Incomplete model output')
   payload=extract_json_object(answer['text']);errors=validate(payload,tokenizer.encode)
   sample['generated_payload']=payload;sample['errors']=errors
   if not errors:sample.update(status='ok',target_attrs=payload['target_attrs'])
  except Exception as error:
   ledger.abandon();sample['errors']=[local.client.safe_text(str(error)) if hasattr(local,'client') else type(error).__name__]
   if any('HTTP '+str(x) in str(error) for x in (401,403,404,429)) or sample['response'] is None:stop.set()
  with lock:samples[task['request_key']]=sample
  ledger.save_sample('clean_slots',dict(sample,query_index=task['request_index']))
  return sample
 error=None
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   iterator=iter(tasks);pending=set()
   for _ in range(min(4,len(tasks))):pending.add(pool.submit(process,next(iterator)))
   while pending:
    done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
    for f in done:f.result()
    returned=[s for s in samples.values() if s['response'] is not None]
    if len(returned)>=16 and sum(x['status']=='ok' for x in returned)/len(returned)<.9:stop.set()
    atomic_write_json(output/'state.json',dict(status='stopping' if stop.is_set() else 'generating',returned=len(returned),valid=sum(x['status']=='ok' for x in returned),planned=len(tasks),cost=ledger.summary()))
    if time.monotonic()-start>1200:stop.set()
    if not stop.is_set():
     for _ in done:
      task=next(iterator,None)
      if task is not None:pending.add(pool.submit(process,task))
  if len(samples)!=len(tasks):raise RuntimeError('019 generation incomplete; do not retry unknown requests')
  returned=[s for s in samples.values() if s['response'] is not None]
  if sum(s['status']=='ok' for s in returned)/max(1,len(returned))<.9:raise RuntimeError('019 valid coverage below90%; no training')
  categories={}
  for cat,rows in manifest['categories'].items():
   out=[]
   for row in rows:
    s=samples[row['request_key']];out.append(dict(original_row=row['original_row'],source_index=row['source_index'],split=row['split'],reference=row['reference'],target=row['target'],modification_text=row['instruction'],request_key=row['request_key'],status=s['status'],target_attrs=s['target_attrs'],errors=s['errors']))
   categories[cat]=out
  atomic_write_json(output/'generated_slots.json',dict(categories=categories,samples=list(samples.values()),api_calls=ledger.summary()['calls'],targets_sent_to_model=False))
 except BaseException as exc:error=exc
 finally:
  key='';cost=ledger.summary();inflight=ledger.db.execute("select count(*) from requests where state in ('reserved','metered')").fetchone()[0]
  budget=CumulativeBudget('/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3');budget.settle(BATCH,cost['accounted_upper_cny'],no_inflight=inflight==0);atomic_write_json(output/'global_after.json',budget.summary());budget.db.close()
  atomic_write_json(output/'completion.json',dict(status='failed' if error else 'complete',error_type=type(error).__name__ if error else None,cost=cost,seconds=time.monotonic()-start,targets_sent_to_model=False))
 if error:raise error

if __name__=='__main__':main()
