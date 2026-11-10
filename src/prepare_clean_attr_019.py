"""Prepare sealed019 from immutable018 rows; publishes plan only after all gates."""
import argparse,json,sqlite3,hashlib
from pathlib import Path
from attr_clean_schema import request_key,PROMPT
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from experiment_storage import require_space

FILES=('src/attr_clean_schema.py','src/generate_clean_attr_slots.py','src/train_clean_attr_alignment.py','src/prepare_clean_attr_019.py',
 'scripts/hold_attr_clean_secret.py','scripts/run_clean_attr_pipeline.py','scripts/run_attr_pair_batch.py',
 'src/attr_token_pair_alignment.py','src/train_attr_pair_alignment.py','src/train_attr_slot_pilot.py','src/attr_slot_fusion.py','src/attr_slot_data.py','src/attr_slot_rank_math.py',
 'src/experiment_storage.py','src/fashioniq_noise.py','src/generate_structured_edits.py','src/generate_shoes_online.py','src/shoes_online_pilot.py','src/cir_cumulative_budget.py',
 'tests/test_attr_clean_schema.py','tests/test_clean_attr_training.py')

def main():
 p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);a=p.parse_args();r=a.research;control=r/'attr_clean_019_control';root=Path(__file__).resolve().parents[1]
 if (control/'generation_plan.json').exists():raise FileExistsError('Never reseal019')
 require_space(r);require_space('/home/lcy',2)
 if json.loads((control/'native_fallback_preflight.json').read_text())['status']!='native_fallback_passed':raise ValueError('Native fallback preflight missing')
 for name in ('schema_tests.log','training_tests.log'):
  if not (control/name).read_text().rstrip().endswith('OK'):raise ValueError('Tests did not pass')
 history={}
 for name in ('cirr_shoes_20261015/online_qwen37_v3/requests.sqlite3','fiq_augmentation_ablation_20261019/fiq_online_requests.sqlite3','fiq_shirt_toptee_20261022/shirt/requests.sqlite3','fiq_shirt_toptee_20261022/toptee/requests.sqlite3'):
  path=r.parent/name;db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
  if db.execute('pragma quick_check').fetchone()[0]!='ok':raise ValueError('Historical ledger integrity')
  states=dict(db.execute('select state,count(*) from requests group by state'))
  if states.get('reserved',0) or states.get('metered',0):raise ValueError('Historical paid calls still in flight')
  history[name]=states;db.close()
 atomic_write_json(control/'historical_inventory.json',history)
 db=sqlite3.connect('file:/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3?mode=ro',uri=True)
 if db.execute('select cap from config where id=1').fetchone()!=(300000000,) or any(row[0]!='settled' for row in db.execute('select state from batches')):raise ValueError('Budget cap/inflight mismatch')
 before=db.execute('select coalesce(sum(upper),0) from history').fetchone()[0]+db.execute('select coalesce(sum(coalesce(charged,cap)),0) from batches').fetchone()[0];db.close()
 primary=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_shirt_toptee_20261022');categories={};tasks={};bindings={}
 for cat in ('shirt','toptee'):
  folder=r/'attr_pair_alignment_018'/cat;m=json.loads((folder/'manifest.json').read_text());result=json.loads((folder/'results.json').read_text());bindings[str(folder/'manifest.json')]=file_sha(folder/'manifest.json');bindings[str(folder/'results.json')]=file_sha(folder/'results.json')
  if result['status']!='complete' or len(m['records'])!=160:raise ValueError('018 not fully complete')
  expected=result['evidence']['structured_train_sha256'];source=primary/cat/('structured_'+cat+'_train.json')
  if file_sha(source)!=expected:raise ValueError('Canonical Qwen source changed')
  ledger=sqlite3.connect('file:'+str(primary/cat/'requests.sqlite3')+'?mode=ro',uri=True)
  if ledger.execute('pragma quick_check').fetchone()[0]!='ok' or ledger.execute("select count(*) from requests where state in ('reserved','metered')").fetchone()[0]:raise ValueError('Old ledger integrity or inflight')
  rows=[]
  for row in m['records']:
   states=ledger.execute("select state from requests where split='train' and idx=?",(row['source_index'],)).fetchall()
   if not states or any(st[0]!='known' for st in states):raise ValueError('Do not resend any old unknown query')
   path=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/resized_image')/(row['reference']+'.jpg');sha=file_sha(path)
   if sha!=m['image_sha256'][row['reference']]:raise ValueError('Reference bytes changed')
   text=row['view']['mod_text'];key=request_key(sha,text)
   if key not in tasks:tasks[key]=dict(request_index=len(tasks),request_key=key,reference_path=str(path),reference_sha256=sha,instruction=text)
   rows.append(dict(original_row=row['original_row'],source_index=row['source_index'],split=row['split'],reference=row['reference'],target=row['target'],instruction=text,request_key=key))
  ledger.close();categories[cat]=rows
 if len(tasks)>320:raise ValueError('Calls exceed initial cap')
 atomic_write_json(control/'manifest.json',dict(categories=categories,unique_requests=list(tasks.values()),targets_in_local_metadata_only=True))
 plan=dict(id='019_clean_attr_slots',cap_cny=1.5,max_calls=320,model='qwen3.7-flash',source_hashes={n:file_sha(root/n) for n in FILES},manifest_sha256=file_sha(control/'manifest.json'),input_bindings=bindings,
  prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),request_count=len(tasks),original_rows=320,api_retries=0,price_source='https://help.aliyun.com/zh/model-studio/qwen3-7-flash',price_checked='2026-10-28',input_cny_per_million=.2,output_cny_per_million=.8,cumulative_accounted_before=before/1e6,old_queue_resumed=False)
 # Publishing this exact file releases the already-authorized in-memory broker.
 atomic_write_json(control/'generation_plan.json',plan)
 print(json.dumps({'status':'sealed','unique_calls':len(tasks),'original_rows':320,'cumulative_before':before/1e6,'batch_cap':1.5,'old_unknown_queries_blocked':True}))

if __name__=='__main__':main()
