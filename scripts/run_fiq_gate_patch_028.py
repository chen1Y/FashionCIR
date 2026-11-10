"""Independent API0 matched-slot-gating and all-candidate local-scoring study."""
import argparse,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(CODE/'src'),str(CODE/'scripts')]
from fashioniq_noise import file_sha
from experiment_storage import require_space
from generate_structured_edits import atomic_write_json
from run_attr_pair_batch import identity

PARENT=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023');CATS=('shirt','toptee')
NEW=('src/fiq_gate_patch_math_028.py','src/fiq_gate_patch_model_028.py','src/fiq_patch_bank_028.py','src/train_fiq_gated_slots_028.py','src/score_fiq_gate_patch_028.py','scripts/run_fiq_gate_patch_028.py','scripts/audit_fiq_gate_patch_028.py','tests/test_fiq_gate_patch_028.py','src/fiq_cache_batch_geometry_026.py')


def prepare(root,tests):
    if root.exists():raise FileExistsError('028 root already exists; inspect first')
    if not tests.read_text().rstrip().endswith('OK'):raise ValueError('028 tests not passed')
    for name in ('verification.json','reuse_verification.json'):
        report=json.loads((PARENT/'audit'/name).read_text())
        if report.get('status')!='verified':raise ValueError('023 independent audit missing')
    if json.loads((PARENT/'state.json').read_text())['status']!='complete':raise ValueError('023 must be complete')
    require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',10)
    for path in ('/dev/shm/fiq_gate_patch_028','/dev/shm/fiq_gate_patch_028_weights'):
        if Path(path).exists():raise FileExistsError('Existing028 cache/weights need inspection')
    oldplan=json.loads((Path('/data/8T1/lcy/CIRData/runs/fiq-full-024')/'plan.json').read_text());source=dict(oldplan['source_hashes'])
    for name,h in source.items():
        if file_sha(CODE/name)!=h:raise ValueError('Legacy sealed source changed: '+name)
    source.update({name:file_sha(CODE/name) for name in NEW});protected={};root.mkdir();(root/'metadata').mkdir();(root/'texts').mkdir();counts={}
    def protect(path):
        st=path.stat();protected[str(path)]=dict(sha256=file_sha(path),size=st.st_size,mtime_ns=st.st_mtime_ns)
    for name in ('plan.json','audit/verification.json','audit/reuse_verification.json'):protect(PARENT/name)
    for cat in CATS:
        for name in ('metadata/'+cat+'_train.json','texts/'+cat+'_train.json','assets_'+cat+'_train.json','teacher_'+cat+'.json'):
            src=PARENT/name;protect(src);shutil.copyfile(src,root/name)
        for name in ('metadata/'+cat+'_val.json','texts/'+cat+'_val.json','assets_'+cat+'_val.json','official/'+cat+'/results.json'):
            protect(PARENT/name)
        meta=json.loads((root/'metadata'/(cat+'_train.json')).read_text());counts[cat]=dict(fit=len(meta['offsets']['fit']),dev=len(meta['offsets']['dev']))
        result=PARENT/'fits'/cat/'slots/results.json';protect(result);fit=json.loads(result.read_text())
        if fit['status']!='complete' or fit['updates']!=3*((counts[cat]['fit']+7)//8) or set(fit['per_fit_row_exposures'])!={3}:raise ValueError('023 baseline fit budget missing')
        protect(Path(fit['selected_weight']))
        for phase in ('train','val'):
            asset=json.loads((PARENT/('assets_'+cat+'_'+phase+'.json')).read_text());protect(Path(asset['cache_path']));protect(Path(asset['base_path']) if 'base_path' in asset else Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_shirt_toptee_20261022')/cat/'checkpoints'/(cat+'_dqu_original_seed42_best.pt'))
        teacher=json.loads((PARENT/('teacher_'+cat+'.json')).read_text());protect(Path(teacher['cache_path']))
    plan=dict(id='fiq_gate_patch028',source_hashes=source,protected=protected,parent=str(PARENT),counts=counts,arms=['baseline_slots','gated_slots','local_summax','global_plus_local'],gate_parameters=145,gate_inputs='slot/reference cosine, slot/EOS cosine, reference/EOS cosine, four-way slot identity',gate_initialization='zero output layer; equal weights among valid slots',gate_lr=2e-4,gate_weight_decay=.01,seed=42,epochs=3,batch=8,loss='full legal-bank InfoNCE/.07 + .1global + .1slot; unchanged; slot loss not gated',selection='max dev meanR10R50, then minNCE, then earlier',vit_frozen=True,text_last_blocks=2,embedding_hook_new_rows=4,local_arm_fitting=False,local_formula='sum over valid attributes of maximum cosine over all256 projected patches of each candidate',hybrid_formula='old image+equal-slot global cosine + local_summax/valid_count; fixed1 coefficient',invalid_fallback='original modification with reference, same old baseline; local/hybrid fallback baseline score',official_queries=dict(shirt=2038,toptee=1961),official_gallery=dict(shirt=6346,toptee=5373),weight_directory='/dev/shm/fiq_gate_patch_028_weights',cache_directory='/dev/shm/fiq_gate_patch_028',estimated_patch_bytes=11919*256*1024*2,api_calls=0,new_qwen_texts=False,max_small_job_seconds=3600,max_master_seconds=25200,old_queue_resumed=False,limitations=['single seed','dev selects checkpoints','base CLIP previously saw official train','official val repeatedly explored','JPEG independence does not cover near duplicates','local arm is scoring ablation, not matched retraining'])
    atomic_write_json(root/'plan.json',plan);shutil.copyfile(tests,root/'tests.log');print(json.dumps(dict(status='prepared',counts=counts,source_files=len(source),new_api_calls=0)),flush=True)


def worker(root):
    import fcntl
    plan=json.loads((root/'plan.json').read_text());start=time.monotonic();state=dict(status='starting',identity=identity(os.getpid()),jobs=[],api_calls=0);child=None;ci=None
    env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ};env.update(HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false');lock=open('/dev/shm/fiq_research_gpu.lock','a')
    def save():atomic_write_json(root/'state.json',state)
    def guard():
        require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',2)
        if (root/'cancel.request').exists():raise KeyboardInterrupt('028 user cancelled')
        if time.monotonic()-start>plan['max_master_seconds']:raise TimeoutError('028 master timeout')
        for name,h in plan['source_hashes'].items():
            if file_sha(CODE/name)!=h:raise ValueError('028 source changed '+name)
        for path,info in plan['protected'].items():
            st=Path(path).stat()
            if st.st_size!=info['size'] or st.st_mtime_ns!=info['mtime_ns']:raise ValueError('028 protected source changed '+path)
    def run(name,script,args):
        nonlocal child,ci
        guard()
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('Another GPU process is active')
        command=[sys.executable,script,'--root',str(root),*args];row=dict(id=name,status='running',command=command);state['jobs'].append(row);state['status']=name;begin=time.monotonic()
        with (root/(name+'.log')).open('x') as log:child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ci=identity(child.pid);row['identity']=ci;save()
        while child.poll() is None:
            guard()
            if time.monotonic()-begin>3600:raise TimeoutError('028 small job exceeds1h')
            time.sleep(2)
        row.update(status='complete' if child.returncode==0 else 'failed',returncode=child.returncode,seconds=time.monotonic()-begin);save()
        if child.returncode:raise RuntimeError('028 '+name+' failed; no automatic retry')
        child=None
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard();save()
        for cat in CATS:run('gate_fit_'+cat,'src/train_fiq_gated_slots_028.py',['--category',cat])
        for cat in CATS:
            if not (root/'fits'/cat/'slots/fit_complete.json').exists():raise ValueError('028 fit gate incomplete')
        for cat in CATS:
            for name in ('metadata/'+cat+'_val.json','texts/'+cat+'_val.json','assets_'+cat+'_val.json'):shutil.copyfile(PARENT/name,root/name)
        for cat in CATS:run('patches_'+cat,'src/fiq_patch_bank_028.py',['--category',cat])
        for cat in CATS:run('scores_'+cat,'src/score_fiq_gate_patch_028.py',['--category',cat])
        run('independent_audit','scripts/audit_fiq_gate_patch_028.py',[])
        state.update(status='complete',seconds=time.monotonic()-start,audit_verified=True);save()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            actual=identity(child.pid)
            if ci and actual and all(actual[k]==ci[k] for k in ('pid','start_ticks','cwd','ppid')):
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(status='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',error=type(exc).__name__+': '+str(exc),seconds=time.monotonic()-start);save();raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tests',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.root,a.tests)
    elif a.launch:
        if (a.root/'state.json').exists():raise FileExistsError('028 already launched; inspect actual identity')
        pid=os.fork()
        if pid:print(json.dumps(dict(status='launched',pid=pid)),flush=True)
        else:
            os.setsid();log=os.open(a.root/'master.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.dup2(log,1);os.dup2(log,2);os.close(0);sys.stdin=open(os.devnull);worker(a.root);os._exit(0)
    elif a.run:worker(a.root)
    else:p.error('Choose prepare, launch or run')
