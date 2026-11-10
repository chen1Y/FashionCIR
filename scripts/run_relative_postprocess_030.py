"""030-P finite, API-free continuation; never retries the sealed030 online phase."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(CODE/'src'),str(CODE/'scripts')]
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from experiment_storage import require_space
from run_attr_pair_batch import identity
from fiq_relative_postprocess_030 import derive
PARENT=Path('/data/8T1/lcy/CIRData/runs/fiq-fusion-029');ONLINE=Path('/data/8T1/lcy/CIRData/runs/fiq-relative-030');DATA=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/captions')
ARMS=('anchor_all','anchor_source','polarity_axis','target_minus_source')
FILES=('src/fiq_relative_postprocess_030.py','src/fiq_relative_adapter_030.py','src/run_fiq_relative_postprocess_030.py','scripts/run_relative_postprocess_030.py','scripts/audit_relative_postprocess_030.py','tests/test_fiq_relative_postprocess_030.py')

def prepare(root,tests):
    import open_clip
    if root.exists():raise FileExistsError('030-P root exists; inspect first')
    if not tests.read_text().rstrip().endswith('OK'):raise ValueError('030-P tests must pass')
    if json.loads((PARENT/'state.json').read_text())['status']!='complete' or json.loads((PARENT/'verification.json').read_text())['status']!='verified':raise ValueError('030-P verified029 required')
    old=json.loads((ONLINE/'generation_plan.json').read_text());sources=dict(old['sources'] if 'sources' in old else old['source_hashes'])
    for n,h in sources.items():
        if file_sha(CODE/n)!=h:raise ValueError('Old sealed source changed '+n)
    sources.update({n:file_sha(CODE/n) for n in FILES});protected={}
    def bind(p):
        p=Path(p);s=p.stat();protected[str(p)]=dict(sha256=file_sha(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
    for name in ('generation_plan.json','generation_state_train.json','generation/train_000/responses.json','generation/train_000/completion.json','generation/train_000/protocol.json'):bind(ONLINE/name)
    for name in ('plan.json','verification.json','state.json'):bind(PARENT/name)
    tok=open_clip.get_tokenizer('ViT-H-14');counts={}
    for cat in ('shirt','toptee'):
        report=PARENT/'fits'/cat/'full_learned_written/results.json';bind(report);d=json.loads(report.read_text());bind(d['selected_weight'])
        counts[cat]={}
        for phase in ('train','val'):
            for p in (PARENT/'metadata'/(cat+'_'+phase+'.json'),PARENT/'texts'/(cat+'_'+phase+'.json'),PARENT/('assets_'+cat+'_'+phase+'.json'),DATA/('image_captions_'+cat+'_'+phase+'.json')):bind(p)
        # Metadata-only real preflight: no model, image forward, target/rank filtering.
        meta=json.loads((PARENT/'metadata'/(cat+'_train.json')).read_text());samples=json.loads((PARENT/'texts'/(cat+'_train.json')).read_text())['samples'];captions=json.loads((DATA/('image_captions_'+cat+'_train.json')).read_text());sem=[]
        for r,s in zip(meta['records'],samples):
            if r['source_index']!=s['source_index'] or r['candidate']!=s['reference'] or r['modification_text']!=s['instruction']:raise ValueError('030-P preflight record binding')
            sem.append(derive(r['modification_text'],s['target_attrs'],captions.get(r['candidate'],''),s['status'],tok.encode))
        for split,ix in meta['offsets'].items():counts[cat][split]=dict(queries=len(ix),rule_queries=sum(bool(sem[i]['relations']) for i in ix),source_queries=sum(any(sem[i]['source_phrases'].values()) for i in ix),enumerated_queries=sum(bool(sem[i]['hypotheses']) for i in ix))
    require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',3)
    root.mkdir();plan=dict(id='030-P_deterministic_comparative_postprocessing',sources=sources,protected=protected,preflight_counts=counts,arms=list(ARMS),parent=str(PARENT),online_phase='failed sealed first8, never consumed as accepted data',new_api_calls=0,image_forwards=0,epochs=3,batch=8,seed=42,optimizer='AdamW lr.005 wd0 clip1; projected b[0,.25], gamma[0,.3], zero start',loss='full legal-bank NCE/.07 +.01 residual parameter squared norm',selection='mean internaldev R10/R50 max, NCE min, earlier',parent_all_parameters_frozen=True,source_anchor='caption-guided reference patch attention, weak proxy not human GT',pole_templates=3,enumeration='2-3 separate encodings; .07 logmeanexp scores; same-count shuffled control',val_after_eight_fits=True,max_master_seconds=14400,max_small_job_seconds=3600,recurring_monitor=False)
    atomic_write_json(root/'plan.json',plan);(root/'tests.log').write_text(tests.read_text());print(json.dumps(dict(status='prepared',counts=counts,source_files=len(sources))),flush=True)

def worker(root):
    import fcntl
    plan=json.loads((root/'plan.json').read_text());start=time.monotonic();state=dict(status='starting',identity=identity(os.getpid()),jobs=[],api_calls=0,recurring_monitor=False);child=None;ci=None;lock=open('/dev/shm/fiq_research_gpu.lock','a');env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ};env.update(HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    def save():atomic_write_json(root/'state.json',state)
    def guard():
        require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',2)
        if (root/'cancel.request').exists():raise KeyboardInterrupt('030-P cancelled')
        if time.monotonic()-start>14400:raise TimeoutError('030-P master timeout')
        for n,h in plan['sources'].items():
            if file_sha(CODE/n)!=h:raise ValueError('030-P sealed source changed '+n)
        for p,h in plan['protected'].items():
            s=Path(p).stat()
            if s.st_size!=h['size'] or s.st_mtime_ns!=h['mtime_ns']:raise ValueError('030-P protected input changed')
    def job(name,script,args):
        nonlocal child,ci
        guard()
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
        command=[sys.executable,script,'--root',str(root),*args];row=dict(id=name,command=command,status='running');state['jobs'].append(row);state['status']=name;begin=time.monotonic()
        with (root/(name+'.log')).open('x') as log:child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ci=identity(child.pid);row['identity']=ci;save()
        while child.poll() is None:
            guard()
            if time.monotonic()-begin>3600:raise TimeoutError('030-P child timeout')
            time.sleep(2)
        row.update(status='complete' if child.returncode==0 else 'failed',returncode=child.returncode,seconds=time.monotonic()-begin);save()
        if child.returncode:raise RuntimeError('030-P '+name+' failed; no restart')
        child=None
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard();save()
        for cat in ('shirt','toptee'):
            job('build_'+cat+'_train','src/run_fiq_relative_postprocess_030.py',['--category',cat,'--build','train'])
            for arm in ARMS:job('fit_'+cat+'_'+arm,'src/run_fiq_relative_postprocess_030.py',['--category',cat,'--fit',arm])
        atomic_write_json(root/'fit_complete.json',dict(status='complete',fits=8))
        for cat in ('shirt','toptee'):
            job('build_'+cat+'_val','src/run_fiq_relative_postprocess_030.py',['--category',cat,'--build','val']);job('official_'+cat,'src/run_fiq_relative_postprocess_030.py',['--category',cat,'--official'])
        job('audit','scripts/audit_relative_postprocess_030.py',[]);state.update(status='complete',seconds=time.monotonic()-start);save()
    except BaseException as e:
        if child is not None and child.poll() is None:
            actual=identity(child.pid)
            if ci and actual and all(actual[k]==ci[k] for k in ('pid','start_ticks','cwd','ppid')):
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(status='cancelled' if isinstance(e,KeyboardInterrupt) else 'failed',error=type(e).__name__+': '+str(e),seconds=time.monotonic()-start);save();raise
    finally:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--tests',type=Path);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.root,a.tests)
    elif a.launch:
        if (a.root/'state.json').exists():raise FileExistsError('030-P already started')
        pid=os.fork()
        if pid:print(json.dumps(dict(status='launched030P',pid=pid)),flush=True)
        else:
            os.setsid();fd=os.open(a.root/'master.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.dup2(fd,1);os.dup2(fd,2);os.close(0);sys.stdin=open(os.devnull);worker(a.root);os._exit(0)
    else:p.error('prepare or launch required')
