"""024 whole FashionIQ; strict operational stop; <=1000 calls/5CNY within original300."""
import concurrent.futures,hashlib,json,threading,time
from pathlib import Path
from expanded_attr_schema_021 import PROMPT,validate,request_key
from fiq_full_resume_024 import resolve,publish_ready
from expanded_attr_resume_023 import DiagnosticClient
from cir_cumulative_budget import CumulativeBudget
from generate_shoes_online import Ledger,RateLimiter
from shoes_online_pilot import Client,BASE_URL,reference_data_url
from generate_structured_edits import atomic_write_json,extract_json_object
from fashioniq_noise import file_sha
import open_clip

GLOBAL=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3')

def generate(root,phase,key,guard):
 textdir=root/'texts';textdir.mkdir(exist_ok=True);generation=root/'generation';generation.mkdir(exist_ok=True)
 rows,answers,todo,reuse=resolve(root,phase)
 atomic_write_json(root/('reuse_'+phase+'.json'),reuse)
 publish_ready(root,phase,rows,answers)
 tokenizer=open_clip.get_tokenizer('ViT-H-14');unknown_count=0
 chunks=[todo[j:j+1000] for j in range(0,len(todo),1000)]
 for batch_index,batch in enumerate(chunks):
  guard();directory=generation/(phase+'_'+str(batch_index).zfill(3))
  if directory.exists():raise FileExistsError('Inspect existing paid021 batch, no automatic restart')
  directory.mkdir();batch_id='fiq_full024_'+phase+'_'+str(batch_index).zfill(3)
  protocol=dict(id=batch_id,cap_cny=5.0,max_calls=len(batch),max_tokens=512,model='qwen3.7-flash',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),rows=batch,automatic_retries=0,reference_only=True,temperature=0,price_checked='2026-10-29',input_cny_per_million=.2,output_cny_per_million=.8)
  atomic_write_json(directory/'protocol.json',protocol);budget=CumulativeBudget(GLOBAL)
  if any(b['state']!='settled' for b in budget.summary()['batches']):raise ValueError('Other paid work in flight')
  budget.reserve(batch_id,5.0,file_sha(directory/'protocol.json'));atomic_write_json(directory/'global_before.json',budget.summary());budget.db.close()
  ledger=Ledger(directory/'requests.sqlite3',protocol,max_cny=5.0,max_calls=len(batch));stop=threading.Event();rate=RateLimiter(60,stop);local=threading.local();done={};lock=threading.Lock();error=None;started=time.monotonic()
  def task(row):
   nonlocal unknown_count
   if stop.is_set():return
   sample=dict(request_key=row['request_key'],request_index=row['request_index'],status='fallback',target_attrs={k:'' for k in ('color','shape','detail','texture')},errors=[],response=None)
   try:
    guard()
    if not hasattr(local,'client'):local.client=DiagnosticClient(key,BASE_URL,ledger)
    if file_sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Reference bytes changed')
    ledger.begin(phase,row['request_index'],'native_relative_slots024');rate.wait()
    response=local.client.ask(reference_data_url(row['reference_path']),PROMPT+row['instruction'],512);ledger.finish(response);sample['response']=response
    if response['finish_reason']!='stop':raise ValueError('Incomplete output')
    payload=extract_json_object(response['text']);sample['generated_payload']=payload;sample['errors']=validate(payload,row['instruction'],tokenizer.encode)
    if not sample['errors']:sample.update(status='ok',target_attrs=payload['target_attrs'])
   except BaseException as exc:
    ledger.abandon();sample['errors']=[local.client.safe_text(str(exc)) if hasattr(local,'client') else type(exc).__name__]
    if sample['response'] is None:stop.set()
   with lock:done[row['request_key']]=sample
   ledger.save_sample(phase,dict(sample,query_index=row['request_index']))
  try:
   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    iterator=iter(batch);pending={pool.submit(task,next(iterator)) for _ in range(min(4,len(batch)))}
    while pending:
     completed,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
     for f in completed:f.result()
     returned=[s for s in done.values() if s['response'] is not None]
     if len(returned)>=16 and sum(s['status']=='ok' for s in returned)/len(returned)<.8:stop.set()
     if time.monotonic()-started>1800:stop.set()
     atomic_write_json(directory/'state.json',dict(status='stopping' if stop.is_set() else 'generating',returned=len(returned),valid=sum(s['status']=='ok' for s in returned),planned=len(batch),cost=ledger.summary()))
     if not stop.is_set():
      for _ in completed:
       row=next(iterator,None)
       if row is not None:pending.add(pool.submit(task,row))
   if stop.is_set() or any(s['response'] is None for s in done.values()) or len(done)!=len(batch):raise RuntimeError('024 batch stopped; preserve known/unknown and do not retry')
   if sum(s['status']=='ok' for s in done.values())/len(batch)<.8:raise RuntimeError('021 strict comparative/schema coverage below80%; no automatic relaxation')
  except BaseException as exc:error=exc
  finally:
   cost=ledger.summary();inflight=ledger.db.execute("select count(*) from requests where state in ('reserved','metered')").fetchone()[0]
   budget=CumulativeBudget(GLOBAL);budget.settle(batch_id,cost['accounted_upper_cny'],inflight==0);atomic_write_json(directory/'global_after.json',budget.summary());budget.db.close()
   atomic_write_json(directory/'responses.json',list(done.values()));atomic_write_json(directory/'completion.json',dict(status='failed' if error else 'complete',cost=cost,seconds=time.monotonic()-started,error=str(error).replace(key,'[REDACTED]') if error else None))
  atomic_write_json(directory/'state.json',dict(status='failed' if error else 'complete',returned=sum(s['response'] is not None for s in done.values()),valid=sum(s['status']=='ok' for s in done.values()),planned=len(batch),cost=cost))
  if error:raise error
  answers.update(done)
  publish_ready(root,phase,rows,answers)
  atomic_write_json(root/('generation_'+phase+'_progress.json'),dict(completed_batches=batch_index+1,completed_unique=sum(t['request_key'] in answers for t in todo),planned_unique=len(todo),total_rows=len(rows)))
 publish_ready(root,phase,rows,answers)
