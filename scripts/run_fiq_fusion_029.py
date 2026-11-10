"""029 finite manual experiment. Does not schedule any recurring monitor."""
import argparse,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(CODE/'src'),str(CODE/'scripts')]
from fashioniq_noise import file_sha
from experiment_storage import require_space
from generate_structured_edits import atomic_write_json
from run_attr_pair_batch import identity
from fiq_fusion_029 import ARMS
PARENT=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023');FILES=('src/fiq_fusion_029.py','src/fiq_assets_029.py','src/train_fiq_fusion_029.py','scripts/run_fiq_fusion_029.py','scripts/audit_fiq_fusion_029.py','tests/test_fiq_fusion_029.py')

def prepare(root,tests):
    if root.exists():raise FileExistsError('029 root exists; inspect first')
    if not tests.read_text().rstrip().endswith('OK'):raise ValueError('029 server tests missing')
    require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',15)
    if Path('/dev/shm/fiq_fusion_029').exists():raise FileExistsError('Existing029 cache')
    old=json.loads((Path('/data/8T1/lcy/CIRData/runs/fiq-gate-patch-028')/'plan.json').read_text());sources=dict(old['source_hashes'])
    for n,h in sources.items():
        if file_sha(CODE/n)!=h:raise ValueError('Old sealed source changed '+n)
    sources.update({n:file_sha(CODE/n) for n in FILES});root.mkdir();(root/'metadata').mkdir();(root/'texts').mkdir();protected={};counts={}
    def bind(p):
        p=Path(p);st=p.stat();protected[str(p)]=dict(sha256=file_sha(p),size=st.st_size,mtime_ns=st.st_mtime_ns)
    for cat in ('shirt','toptee'):
        for phase in ('train','val'):
            for name in ('metadata/'+cat+'_'+phase+'.json','texts/'+cat+'_'+phase+'.json'):bind(PARENT/name)
            asset=json.loads((PARENT/('assets_'+cat+'_'+phase+'.json')).read_text())
            if file_sha(PARENT/'metadata'/(cat+'_'+phase+'.json'))!=asset['metadata_sha256']:raise ValueError('029 inherited metadata SHA mismatch')
            evidence=json.loads((PARENT/('fits/'+cat+'/slots/results.json' if phase=='train' else 'official/'+cat+'/results.json')).read_text())
            if file_sha(PARENT/'texts'/(cat+'_'+phase+'.json'))!=evidence['text_sha256']:raise ValueError('029 inherited text SHA mismatch')
        for name in ('metadata/'+cat+'_train.json','texts/'+cat+'_train.json'):shutil.copyfile(PARENT/name,root/name)
        meta=json.loads((root/'metadata'/(cat+'_train.json')).read_text());counts[cat]=dict(fit=len(meta['offsets']['fit']),dev=len(meta['offsets']['dev']))
        base=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_shirt_toptee_20261022')/cat/'checkpoints'/(cat+'_dqu_original_seed42_best.pt');bind(base)
        for n in ('keywords_in_mods_'+cat+'.json','image_captions_'+cat+'_val.json','correction_dict_'+cat+'.json'):bind(Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/captions')/n)
    plan=dict(id='029_gated_text_native_written_adaptive_fusion',sources=sources,protected=protected,counts=counts,arms=list(ARMS),epochs=3,batch=8,seed=42,weight_directory='/dev/shm/fiq_fusion_029_weights',cache_directory='/dev/shm/fiq_fusion_029',api_calls=0,old_texts_reused=True,vit_frozen=True,text_last_blocks=2,embedding_new_rows=4,attribute_gate='145 parameters, uniform start',text_beta='fixed.5 controls; full learns .25+.5sigmoid within [.25,.75]',native_written_keywords=True,image_fusion='Original DQU combiner/scaler initialized from originalDQU; trainable lr1e-4/wd.01/dropout.5',loss='full legal bank NCE/.07 +.1global+.1slot',selection='mean devR10R50 max, NCE min, earlier',max_master_seconds=43200,max_small_job_seconds=3600,recurring_monitor=False,new_bank_encoding_reason='New instance tmpfs empty; required reference/GT patches and native written query vectors missing; no existing029 cache reencoded',old_queue_resumed=False)
    atomic_write_json(root/'plan.json',plan);shutil.copyfile(tests,root/'tests.log');print(json.dumps(dict(status='prepared029',counts=counts,source_files=len(sources))),flush=True)

def worker(root):
    import fcntl
    plan=json.loads((root/'plan.json').read_text());start=time.monotonic();state=dict(status='starting',identity=identity(os.getpid()),jobs=[],api_calls=0,recurring_monitor=False);child=None;ci=None;lock=open('/dev/shm/fiq_research_gpu.lock','a');env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ};env.update(HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    def save():atomic_write_json(root/'state.json',state)
    def guard():
        require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',2)
        if (root/'cancel.request').exists():raise KeyboardInterrupt('029 cancelled')
        if time.monotonic()-start>43200:raise TimeoutError('029 master timeout')
        for n,h in plan['sources'].items():
            if file_sha(CODE/n)!=h:raise ValueError('029 sealed source changed')
        for p,h in plan['protected'].items():
            s=Path(p).stat()
            if s.st_size!=h['size'] or s.st_mtime_ns!=h['mtime_ns']:raise ValueError('029 protected input changed')
    def job(name,script,args):
        nonlocal child,ci
        guard()
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied; no interference')
        command=[sys.executable,script,'--root',str(root),*args];row=dict(id=name,command=command,status='running');state['jobs'].append(row);state['status']=name;begin=time.monotonic()
        with (root/(name+'.log')).open('x') as log:child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ci=identity(child.pid);row['identity']=ci;save()
        while child.poll() is None:
            guard()
            if time.monotonic()-begin>3600:raise TimeoutError('029 child timeout')
            time.sleep(2)
        row.update(status='complete' if child.returncode==0 else 'failed',returncode=child.returncode,seconds=time.monotonic()-begin);save()
        if child.returncode:raise RuntimeError('029 '+name+' failed; stop and inspect, no retry')
        child=None
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard();save()
        for cat in ('shirt','toptee'):
            job('assets_'+cat+'_train','src/fiq_assets_029.py',['--category',cat,'--phase','train'])
            for arm in ARMS:job('fit_'+cat+'_'+arm,'src/train_fiq_fusion_029.py',['--category',cat,'--arm',arm])
        for cat in ('shirt','toptee'):
            for name in ('metadata/'+cat+'_val.json','texts/'+cat+'_val.json'):shutil.copyfile(PARENT/name,root/name)
            job('assets_'+cat+'_val','src/fiq_assets_029.py',['--category',cat,'--phase','val']);job('official_'+cat,'src/train_fiq_fusion_029.py',['--category',cat,'--official'])
        job('audit','scripts/audit_fiq_fusion_029.py',[]);state.update(status='complete',seconds=time.monotonic()-start);save()
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
        if (a.root/'state.json').exists():raise FileExistsError('029 already started')
        pid=os.fork()
        if pid:print(json.dumps(dict(status='launched029',pid=pid)),flush=True)
        else:
            os.setsid();fd=os.open(a.root/'master.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.dup2(fd,1);os.dup2(fd,2);os.close(0);sys.stdin=open(os.devnull);worker(a.root);os._exit(0)
    else:p.error('prepare or launch required')
