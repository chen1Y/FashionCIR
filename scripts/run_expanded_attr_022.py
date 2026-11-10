"""One explicit larger020 batch; credentials remain memory-only; no old queue replay."""
import argparse,getpass,hashlib,json,os,resource,signal,subprocess,sys,time,shutil,threading,concurrent.futures,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from expanded_attr_metadata import prepare,OLD,PRIMARY,ARMS
from expanded_attr_provider_022 import generate
from expanded_attr_resume_022 import resolve,publish_ready
from expanded_attr_assets import base_path
from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from run_attr_pair_batch import identity

FILES=('src/expanded_attr_schema.py','src/expanded_attr_metadata.py','src/expanded_attr_model.py','src/expanded_attr_assets.py','src/expanded_attr_provider.py','src/train_expanded_attr.py','src/native_expanded_attr_020.py','scripts/run_expanded_attr_020.py',
 'src/attr_token_pair_alignment.py','src/attr_slot_fusion.py','src/attr_slot_data.py','src/train_attr_slot_pilot.py','src/train_attr_pair_alignment.py','src/train_clean_attr_alignment.py','src/attr_clean_schema.py','src/fiq_byte_clean_pools.py','src/fiq_candidate_partition.py',
 'src/generate_shoes_online.py','src/shoes_online_pilot.py','src/generate_structured_edits.py','src/cir_cumulative_budget.py','src/fashioniq_noise.py','src/experiment_storage.py','scripts/run_attr_pair_batch.py','tests/test_expanded_attr_020.py')

FILES=FILES+('src/expanded_attr_schema_021.py','src/expanded_attr_provider_021.py','scripts/run_expanded_attr_021.py','tests/test_expanded_attr_021.py',
 'src/expanded_attr_resume_022.py','src/expanded_attr_provider_022.py','scripts/run_expanded_attr_022.py','tests/test_expanded_attr_022.py','src/native_expanded_attr_022.py')


def prepare_plan(root,tests):
 if root.exists():raise FileExistsError('Do not overwrite or restart a continuation root')
 require_space(OLD);require_space('/home/lcy',2.6)
 if not tests.read_text().rstrip().endswith('OK'):raise ValueError('Continuation tests failed')
 parent=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-021');audit=json.loads((parent/'failure_audit_20261029.json').read_text());oldplan=json.loads((parent/'plan.json').read_text())
 if audit['status']!='failed_before_fit' or audit['known']!=1215 or audit['unknown']!=1 or audit['inflight'] or audit['fit_results'] or audit['official_results']:raise ValueError('Unexpected parent failure; inspect instead')
 if json.loads((parent/'state.json').read_text())['status']!='failed':raise ValueError('Parent is not stopped')
 if Path('/proc/9561').exists() or subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('An existing process requires inspection')
 for n,h in oldplan['source_hashes'].items():
  if file_sha(ROOT/n)!=h:raise ValueError('Parent source changed')
 for n,d in oldplan['protected_texts'].items():
  if file_sha(n)!=d['sha256']:raise ValueError('Old text changed')
 budget=sqlite3.connect('file:/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3?mode=ro',uri=True)
 if budget.execute("select count(*) from batches where state!='settled'").fetchone()[0]:raise ValueError('Paid work still in flight')
 budget.close();root.mkdir();(root/'metadata').mkdir()
 for cat in ('shirt','toptee'):
  source=parent/'metadata'/(cat+'_train.json');shutil.copyfile(source,root/'metadata'/source.name)
  report=parent/('assets_'+cat+'_train.json');d=json.loads(report.read_text())
  if file_sha(d['cache_path'])!=d['cache_sha256'] or file_sha(source)!=d['metadata_sha256']:raise ValueError('Existing feature bank binding changed')
  shutil.copyfile(report,root/report.name)
 plan=oldplan;plan.update(id='expanded_attr022',parent_root=str(parent),same_prompt021=True,generation_version='022_continuation',source_hashes={n:file_sha(ROOT/n) for n in FILES},new_transport_unknown_per_phase_limit=3,automatic_retries=0,cache_reencoded=False)
 for f in [parent/'state.json',parent/'plan.json',parent/'failure_audit_20261029.json',*parent.glob('metadata/*.json'),*parent.glob('generation/*/protocol.json'),*parent.glob('generation/*/responses.json'),*parent.glob('generation/*/completion.json'),*parent.glob('generation/*/requests.sqlite3'),*parent.glob('assets_*_train.json')]:
  st=f.stat();plan['protected_texts'][str(f)]=dict(sha256=file_sha(f),size=st.st_size,mtime_ns=st.st_mtime_ns)
 rows,answers,todo,proof=resolve(root,'train');publish_ready(root,'train',rows,answers)
 if proof['categories']['shirt']['missing']!=0 or not (root/'texts/shirt_train.json').exists():raise ValueError('Shirt not fully ready')
 atomic_write_json(root/'preflight.json',dict(status='passed',reuse=proof,metadata_sha256={cat:file_sha(root/'metadata'/(cat+'_train.json')) for cat in ('shirt','toptee')},cache_reencoded=False,old_unknown_resend=False))
 atomic_write_json(root/'plan.json',plan);shutil.copyfile(tests,root/'tests.log');shutil.copyfile(parent/'native_preflight.json',root/'native_preflight.json')
 print(json.dumps({'status':'prepared022','reuse':proof['categories'],'unique_new_requests':len(todo)}),flush=True)


def worker(root,key):
 import fcntl
 lock=(OLD/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=json.loads((root/'plan.json').read_text());started=time.monotonic();state=dict(status='starting',identity=identity(os.getpid()),jobs=[],old_queue_resumed=False);child=None;ci=None;asset=None;asset_identity=None;asset_started=None;generation_future=None;generation_pool=None;abort=threading.Event()
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
  require_space(OLD);require_space(root,2)
  if abort.is_set():raise RuntimeError('Continuation aborted')
  if generation_future is not None and generation_future.done() and generation_future.exception() is not None:raise generation_future.exception()
  if (root/'cancel.request').exists():raise KeyboardInterrupt('User stopped020')
  if time.monotonic()-started>14400:raise TimeoutError('020 master exceeds4h')
  for n,h in plan['source_hashes'].items():
   if file_sha(ROOT/n)!=h:raise ValueError('Sealed020 source changed')
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
  generation_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
  state['status']='online_train_texts_and_ready_category_training';save()
  generation_future=generation_pool.submit(generate,root,'train',key,guard)
  for cat in ('shirt','toptee'):
   while not (root/'texts'/(cat+'_train.json')).exists():guard();time.sleep(2)
   launch_job('teacher_'+cat,'src/expanded_attr_assets.py',['--phase','teacher','--category',cat])
   for arm in ARMS:
    launch_job('smoke_'+cat+'_'+arm,'src/native_expanded_attr_022.py',['--category',cat,'--arm',arm,'--smoke'])
    launch_job('fit_'+cat+'_'+arm,'src/train_expanded_attr.py',['--category',cat,'--arm',arm])
  while not generation_future.done():guard();time.sleep(2)
  generation_future.result();generation_pool.shutdown();generation_future=None;generation_pool=None
  # First parse official query labels only after all six immutable fit flags.
  prepare(root,'val');state['official_labels_loaded_after_all_fits']=True;save()
  visuals=launch_job('val_visuals','src/expanded_attr_assets.py',['--phase','val','--category','both'],True)
  state['status']='online_official_texts';save();generate(root,'val',key,guard);finish_asset(visuals);key=''
  for cat in ('shirt','toptee'):launch_job('official_'+cat,'src/train_expanded_attr.py',['--category',cat,'--official'])
  for path,info in plan['protected_texts'].items():
   if file_sha(path)!=info['sha256']:raise ValueError('Old text SHA changed')
  state.update(status='complete',seconds=time.monotonic()-started,api_credential_retained=False);save()
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
 if pid:print(json.dumps({'022_master_pid':pid,'root':str(a.root),'secret_on_disk':False}),flush=True);return
 os.setsid()
 with open(os.devnull,'rb') as null,(a.root/'master.log').open('xb',buffering=0) as log:os.dup2(null.fileno(),0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
 worker(a.root,key)

if __name__=='__main__':main()
