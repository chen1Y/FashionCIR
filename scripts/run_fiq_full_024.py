"""Full024 gated on independently verified023; credentials stay memory-only."""
import argparse,getpass,hashlib,json,os,resource,signal,subprocess,sys,time,shutil,threading,concurrent.futures,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from expanded_attr_metadata import OLD,PRIMARY,ARMS
from fiq_full_metadata_024 import prepare_train,prepare_val,CATS,BASES
from fiq_full_partition_024 import validation_gate_state
from fiq_full_provider_024 import generate
from fiq_full_resume_024 import resolve,publish_ready
from fiq_full_assets_024 import base_path
from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from run_attr_pair_batch import identity

FILES=('src/expanded_attr_schema.py','src/expanded_attr_metadata.py','src/expanded_attr_model.py','src/expanded_attr_assets.py','src/expanded_attr_provider.py','src/train_expanded_attr.py','src/native_expanded_attr_020.py','scripts/run_expanded_attr_020.py',
 'src/attr_token_pair_alignment.py','src/attr_slot_fusion.py','src/attr_slot_data.py','src/train_attr_slot_pilot.py','src/train_attr_pair_alignment.py','src/train_clean_attr_alignment.py','src/attr_clean_schema.py','src/fiq_byte_clean_pools.py','src/fiq_candidate_partition.py',
 'src/generate_shoes_online.py','src/shoes_online_pilot.py','src/generate_structured_edits.py','src/cir_cumulative_budget.py','src/fashioniq_noise.py','src/experiment_storage.py','scripts/run_attr_pair_batch.py','tests/test_expanded_attr_020.py')

FILES=FILES+('src/expanded_attr_schema_021.py','src/expanded_attr_provider_021.py','scripts/run_expanded_attr_021.py','tests/test_expanded_attr_021.py',
 'src/expanded_attr_resume_022.py','src/expanded_attr_provider_022.py','scripts/run_expanded_attr_022.py','tests/test_expanded_attr_022.py','src/native_expanded_attr_022.py')


FILES=FILES+('src/expanded_attr_resume_023.py','src/expanded_attr_provider_023.py','scripts/run_expanded_attr_023.py','tests/test_expanded_attr_023.py')


FILES=FILES+('src/fiq_full_partition_024.py','src/fiq_full_metadata_024.py','src/fiq_full_assets_024.py','src/fiq_full_resume_024.py','src/fiq_full_provider_024.py','src/train_fiq_full_024.py','scripts/run_fiq_full_024.py','tests/test_fiq_full_024.py','scripts/audit_expanded_attr_020.py','scripts/audit_fiq_full_024.py')


def prepare_plan(root,tests):
 if root.exists():raise FileExistsError('Full-study root already exists; inspect first')
 require_space(OLD);require_space('/home/lcy',2);require_space('/dev/shm',14)
 if not tests.read_text().rstrip().endswith('OK'):raise ValueError('Full-study tests failed')
 validation=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023');v=json.loads((validation/'state.json').read_text())
 if v['status'] in ('failed','cancelled'):raise ValueError('User requires validation error to stop downstream full study')
 if Path('/dev/shm/fiq_full_024').exists() or Path('/dev/shm/fiq_full_024_weights').exists():raise FileExistsError('An existing full cache needs inspection')
 root.mkdir();prepare_train(root);inventory=json.loads((root/'full_train_inventory.json').read_text());metas={cat:json.loads((root/'metadata'/(cat+'_train.json')).read_text()) for cat in CATS}
 estimate=sum(len(set(r['target'] for r in m['records']))*256*1024*2+sum(len(x) for x in m['banks'].values())*1024*4+len(m['records'])*4*1024*4 for m in metas.values())+36*100795311+512*2**20
 require_space('/dev/shm',estimate/2**30+2)
 protected=[validation/'plan.json',OLD/'train_only_candidate_head/official_heldout_byte_index.json.gz']
 for cat in CATS:
  for split in ('train','val'):
   protected.extend([Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/captions')/('cap.'+cat+'.'+split+'.json'),Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/image_splits')/('split.'+cat+'.'+split+'.json')])
  if cat=='dress':protected.extend(Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/captions')/('structured_edits_dress_'+split+'_qwen37flash.json') for split in ('train','val'))
 plan=dict(id='fiq_full024',source_hashes={name:file_sha(ROOT/name) for name in FILES},protected_texts={str(f):dict(sha256=file_sha(f),size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in protected},protected_checkpoints={str(path):dict(sha256=metas[cat]['base_sha256'],size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns) for cat,path in BASES.items()},
  metadata_hashes={str(root/'metadata'/(cat+'_train.json')):file_sha(root/'metadata'/(cat+'_train.json')) for cat in CATS},fit_counts={cat:len(m['offsets']['fit']) for cat,m in metas.items()},dev_counts={cat:len(m['offsets']['dev']) for cat,m in metas.items()},arms=list(ARMS),epochs=3,batch=8,seed=42,selection='max internaldev mean(R10,R50); ties min devNCE, earlier epoch',official_queries=dict(dress=2017,shirt=2038,toptee=1961),official_gallery={cat:len(json.loads((Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/image_splits')/('split.'+cat+'.val.json')).read_text())) for cat in CATS},
  validation_root=str(validation),validation_gate_required=True,cumulative_cap_cny=300,per_paid_batch_cap_cny=5,max_calls_per_batch=1000,max_master_seconds=43200,max_small_job_seconds=3600,max_validation_wait_seconds=18000,
  weight_directory='/dev/shm/fiq_full_024_weights',cache_directory='/dev/shm/fiq_full_024',estimated_new_tmpfs_gib=estimate/2**30,original_arm_uses_only_raw_caption_sha=True,full_original_rows=18000,old_queue_resumed=False,models_and_training_hyperparameters_unchanged=True)
 atomic_write_json(root/'plan.json',plan);shutil.copyfile(tests,root/'tests.log');atomic_write_json(root/'native_preflight.json',dict(status='passed',scope='Synthetic and metadata preflight only; each full arm requires batch8 native optimizer smoke before fitting'))
 print(json.dumps({'status':'prepared024_waiting_validation','fit':plan['fit_counts'],'dev':plan['dev_counts'],'estimated_new_tmpfs_gib':plan['estimated_new_tmpfs_gib']}),flush=True)


def worker(root,key):
 import fcntl
 lock=None;plan=json.loads((root/'plan.json').read_text());started=time.monotonic();state=dict(status='starting',identity=identity(os.getpid()),jobs=[],old_queue_resumed=False);child=None;ci=None;asset=None;asset_identity=None;asset_started=None;generation_future=None;generation_pool=None;abort=threading.Event()
 env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ};env.update(HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
 def save():atomic_write_json(root/'state.json',state)
 def stop_owned(p,ident):
  if p and p.poll() is None:
   actual=identity(p.pid)
   if actual and ident and all(actual[k]==ident[k] for k in ('pid','ppid','start_ticks','cwd')):
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 def guard():
  require_space(OLD);require_space(root,2);require_space('/dev/shm',2)
  if abort.is_set():raise RuntimeError('Continuation aborted')
  if generation_future is not None and generation_future.done() and generation_future.exception() is not None:raise generation_future.exception()
  if (root/'cancel.request').exists():raise KeyboardInterrupt('User stopped020')
  if time.monotonic()-started>plan['max_master_seconds']:raise TimeoutError('Full master exceeds12h')
  for n,h in plan['source_hashes'].items():
   if file_sha(ROOT/n)!=h:raise ValueError('Sealed020 source changed')
  for path,h in plan['metadata_hashes'].items():
   if file_sha(path)!=h:raise ValueError('Sealed full train metadata changed')
  for path,info in {**plan['protected_texts'],**plan['protected_checkpoints']}.items():
   st=Path(path).stat()
   if st.st_size!=info['size'] or st.st_mtime_ns!=info['mtime_ns']:raise ValueError('Protected original text/weight changed')
  if asset is not None:
   if asset.poll() not in (None,0):raise RuntimeError('020 asset job failed; stop new API calls')
   if asset.poll() is None and time.monotonic()-asset_started>3600:raise TimeoutError('Asset job exceeds1h')
 def launch_job(name,script,args,background=False):
  nonlocal child,ci,asset,asset_identity,asset_started
  guard()
  if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied by an existing process; do not interfere')
  cmd=[sys.executable,script,'--root',str(root),*args];row=dict(id=name,status='running',command=cmd);state['jobs'].append(row);state['status']=name;begin=time.monotonic()
  with (root/(name+'.log')).open('x') as log:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  ident=identity(p.pid);row['identity']=ident;save()
  if background:asset=p;asset_identity=ident;asset_started=begin;return row
  child=p;ci=ident
  while p.poll() is None:
   guard()
   if time.monotonic()-begin>3600:raise TimeoutError('Small020 job exceeds1h')
   time.sleep(2)
  row.update(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,seconds=time.monotonic()-begin);save();child=None
  if p.returncode:raise RuntimeError('020 '+name+' failed; no automatic retry')
 def finish_asset(row):
  nonlocal asset
  while asset.poll() is None:guard();time.sleep(2)
  row.update(status='complete' if asset.returncode==0 else 'failed',returncode=asset.returncode,seconds=time.monotonic()-asset_started);save()
  if asset.returncode:raise RuntimeError('020 visual preparation failed')
  asset=None
 try:
  guard();save()
  if json.loads((root/'native_preflight.json').read_text())['status']!='passed':raise ValueError('Native preflight missing')
  state['status']='waiting_for_successful_023_validation';state['downstream_allowed']=False;save();validation=Path(plan['validation_root']);wait_started=time.monotonic()
  while True:
   guard();previous=json.loads((validation/'state.json').read_text())
   if validation_gate_state(previous)=='blocked':
    state.update(status='not_started_validation_failed',reason='User requires reporting Toptee validation failure before any full experiment',api_calls=0,fit_updates=0);save();key='';return
   if validation_gate_state(previous)=='audit_required':
    actual=identity(previous['identity']['pid'])
    if actual and actual['start_ticks']==previous['identity']['start_ticks']:time.sleep(2);continue
    break
   if time.monotonic()-wait_started>plan['max_validation_wait_seconds']:raise TimeoutError('Validation gate wait exceeds5h')
   time.sleep(5)
  with (root/'validation_gate_audit.log').open('x') as log:
   result=subprocess.run([sys.executable,'scripts/audit_expanded_attr_020.py','--root',str(validation)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
  if result.returncode:raise ValueError('023 independent audit failed; full study must not start')
  certificate=json.loads((validation/'audit/verification.json').read_text())
  if certificate['status']!='verified':raise ValueError('No verified validation certificate')
  atomic_write_json(root/'validation_gate_certificate.json',dict(status='passed',validation_state_sha256=file_sha(validation/'state.json'),validation_audit_sha256=file_sha(validation/'audit/verification.json'),full_api_started_after_validation=True))
  lock=(OLD/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();state['validation_gate_passed']=True;save()
  generation_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);generation_future=generation_pool.submit(generate,root,'train',key,guard)
  for cat in CATS:
   launch_job('train_images_'+cat,'src/fiq_full_assets_024.py',['--phase','train','--category',cat])
   launch_job('smoke_'+cat+'_original','src/train_fiq_full_024.py',['--category',cat,'--arm','original','--smoke'])
   launch_job('fit_'+cat+'_original','src/train_fiq_full_024.py',['--category',cat,'--arm','original'])
  for cat in CATS:
   while not (root/'texts'/(cat+'_train.json')).exists():guard();time.sleep(2)
   launch_job('teacher_'+cat,'src/fiq_full_assets_024.py',['--phase','teacher','--category',cat])
   for arm in ('slots','dual'):
    launch_job('smoke_'+cat+'_'+arm,'src/train_fiq_full_024.py',['--category',cat,'--arm',arm,'--smoke'])
    launch_job('fit_'+cat+'_'+arm,'src/train_fiq_full_024.py',['--category',cat,'--arm',arm])
  while not generation_future.done():guard();time.sleep(2)
  generation_future.result();generation_pool.shutdown();generation_future=None;generation_pool=None
  prepare_val(root);state['official_labels_loaded_after_all_fits']=True;save()
  visuals=launch_job('val_visuals','src/fiq_full_assets_024.py',['--phase','val','--category','both'],True)
  state['status']='online_full_official_texts';save();generate(root,'val',key,guard);finish_asset(visuals);key=''
  for cat in CATS:launch_job('official_'+cat,'src/train_fiq_full_024.py',['--category',cat,'--official'])
  state['status']='retrieval_complete_pending_audit';save();launch_job('independent_full_audit','scripts/audit_fiq_full_024.py',[])
  for path,info in plan['protected_texts'].items():
   if file_sha(path)!=info['sha256']:raise ValueError('Old text SHA changed')
  state.update(status='complete',seconds=time.monotonic()-started,api_credential_retained=False,downstream_allowed=True);save()
 except BaseException as exc:
  abort.set();stop_owned(child,ci);stop_owned(asset,asset_identity)
  if generation_pool is not None:generation_pool.shutdown(wait=True,cancel_futures=True)
  message=str(exc).replace(key,'[REDACTED]') if key else str(exc);key='';state.update(status='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',error_type=type(exc).__name__,error=message);save();raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--tests',type=Path);p.add_argument('--launch',action='store_true');a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 if a.prepare:prepare_plan(a.root,a.tests);return
 if not a.launch or not sys.stdin.isatty():raise ValueError('Explicit hidden interactive launch required')
 if (a.root/'state.json').exists():raise FileExistsError('Inspect existing020 state before any continuation')
 if json.loads((a.root/'native_preflight.json').read_text())['status']!='passed':raise ValueError('Native test required')
 key=getpass.getpass('DashScope API key (hidden, memory only): ').strip()
 if not key or len(key)>4000:raise ValueError('Invalid credential length')
 pid=os.fork()
 if pid:print(json.dumps({'024_master_pid':pid,'root':str(a.root),'secret_on_disk':False}),flush=True);return
 os.setsid()
 with open(os.devnull,'rb') as null,(a.root/'master.log').open('xb',buffering=0) as log:os.dup2(null.fileno(),0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
 worker(a.root,key)

if __name__=='__main__':main()
