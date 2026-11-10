"""Supplemental immutable gate: audit023 reuse before entering sealed024 worker."""
import argparse,getpass,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from expanded_attr_resume_022 import reconcile
from fiq_full_partition_024 import validation_gate_state
from run_attr_pair_batch import identity
from experiment_storage import require_space


def verify_paid_sets(prior,current):
    if set(prior)&set(current):raise ValueError('A previous paid/unknown query was resent')
    if any(s['response'] is None for s in current.values()):raise ValueError('023 has a new operational unknown; stop downstream')


def verify_sample(sample,answers):
    if sample['request_key'] in answers:
        previous=answers[sample['request_key']]
        if any(sample[k]!=previous[k] for k in ('status','target_attrs')):raise ValueError('Saved text differs from durable response')
    elif sample['origin']!='historical_block':raise ValueError('Text has no durable request or historical fallback')


def audit_reuse(validation):
    if json.loads((validation/'state.json').read_text())['status']!='complete':raise ValueError('023 not complete')
    prior={};proof=[]
    for parent in (Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-021'),Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-022')):
        found,p=reconcile(parent)
        if set(prior)&set(found):raise ValueError('Duplicate prior paid identity')
        prior.update(found);proof.append(dict(root=str(parent),proof=p))
    current,p=reconcile(validation);verify_paid_sets(prior,current);proof.append(dict(root=str(validation),proof=p))
    allanswers={**prior,**current};samples=0
    for cat in ('shirt','toptee'):
        for phase in ('train','val'):
            for sample in json.loads((validation/'texts'/(cat+'_'+phase+'.json')).read_text())['samples']:verify_sample(sample,allanswers);samples+=1
        for phase in ('train','val'):
            d=json.loads((validation/('assets_'+cat+'_'+phase+'.json')).read_text())
            if file_sha(d['cache_path'])!=d['cache_sha256'] or file_sha(validation/'metadata'/(cat+'_'+phase+'.json'))!=d['metadata_sha256']:raise ValueError('Image bank/text metadata provenance changed')
        t=json.loads((validation/('teacher_'+cat+'.json')).read_text())
        if file_sha(t['cache_path'])!=t['cache_sha256']:raise ValueError('Teacher SHA changed')
    parent=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-022');reused=['texts/shirt_train.json','teacher_shirt.json']
    for arm in ('original','slots','dual'):reused.extend('fits/shirt/'+arm+'/'+name for name in ('results.json','fit_complete.json','training.jsonl'))
    for name in reused:
        if file_sha(validation/name)!=file_sha(parent/name):raise ValueError('Retained Shirt artifact changed')
    for directory in sorted((validation/'generation').iterdir()):
        if not directory.is_dir():continue
        if json.loads((directory/'completion.json').read_text())['status']!='complete':raise ValueError('023 generation batch failed')
        db=sqlite3.connect('file:'+str(directory/'requests.sqlite3')+'?mode=ro',uri=True)
        bad=db.execute("select count(*) from requests where state!='known'").fetchone()[0];db.close()
        if bad:raise ValueError('023 new unknown or in-flight request')
        for row in json.loads((directory/'protocol.json').read_text())['rows']:
            if file_sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Actual requested JPEG changed')
    db=sqlite3.connect('file:/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3?mode=ro',uri=True);pending=db.execute("select count(*) from batches where state!='settled'").fetchone()[0];db.close()
    if pending:raise ValueError('Paid batches remain unsettled')
    result=dict(status='verified',new_old_paid_key_intersection=0,new_unknown=0,paid_inflight=0,reused_shirt_artifacts=reused,text_samples_bound_to_durable_source=samples,proof=proof)
    dest=validation/'audit/reuse_verification.json';payload=json.dumps(result,indent=2)+'\n'
    if dest.exists():
        if dest.read_text()!=payload:raise ValueError('Do not overwrite different reuse audit')
    else:dest.write_text(payload)
    return result


def prepare(root,tests):
    destination=root/'continuation_gate_plan.json'
    if destination.exists():raise FileExistsError('Supplemental gate is immutable')
    if not tests.read_text().rstrip().endswith('OK'):raise ValueError('Supplemental gate tests failed')
    if json.loads((root/'data_preflight_verification.json').read_text())['status']!='verified':raise ValueError('Real data/mask preflight missing')
    p=dict(full_plan_sha256=file_sha(root/'plan.json'),data_preflight_sha256=file_sha(root/'data_preflight_verification.json'),source_hashes={n:file_sha(ROOT/n) for n in ('scripts/gate_fiq_full_024.py','tests/test_fiq_full_gate_024.py')},tests_sha256=file_sha(tests),max_wait_seconds=18000,zero_full_api_and_gpu_before_audits=True)
    atomic_write_json(destination,p);(root/'continuation_gate_tests.log').write_text(tests.read_text());print('Supplemental gate sealed')


def supervise(root,key):
    gate=json.loads((root/'continuation_gate_plan.json').read_text());plan=json.loads((root/'plan.json').read_text());validation=Path(plan['validation_root']);start=time.monotonic();state=dict(status='waiting_023_success_and_two_audits',identity=identity(os.getpid()),full_worker_started=False,full_api_calls=0,full_fit_updates=0)
    def save():atomic_write_json(root/'continuation_gate_state.json',state)
    def guard():
        require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',2)
        if (root/'cancel.request').exists():raise KeyboardInterrupt('User cancelled full024')
        if file_sha(root/'plan.json')!=gate['full_plan_sha256'] or file_sha(root/'data_preflight_verification.json')!=gate['data_preflight_sha256']:raise ValueError('Full plan/preflight changed')
        for n,h in {**plan['source_hashes'],**gate['source_hashes']}.items():
            if file_sha(ROOT/n)!=h:raise ValueError('Sealed continuation source changed')
    try:
        save()
        while True:
            guard();v=json.loads((validation/'state.json').read_text());status=validation_gate_state(v)
            if status=='blocked':state.update(status='not_started_validation_failed',reason='User requires reporting error before downstream');save();key='';return
            if status=='audit_required':
                actual=identity(v['identity']['pid'])
                if not actual or actual['start_ticks']!=v['identity']['start_ticks']:break
            if time.monotonic()-start>gate['max_wait_seconds']:raise TimeoutError('Validation wait exceeds5h')
            time.sleep(5)
        state['status']='auditing_023';save()
        env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ}
        with (root/'continuation_validation_audit.log').open('x') as log:subprocess.run([sys.executable,'scripts/audit_expanded_attr_020.py','--root',str(validation)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
        guard();audit_reuse(validation);guard()
        state.update(status='passed_entering_full_worker',full_worker_started=True,reuse_audit_sha256=file_sha(validation/'audit/reuse_verification.json'));save()
        from run_fiq_full_024 import worker
        worker(root,key);key='';state.update(status='full_worker_returned');save()
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc).replace(key,'[REDACTED]') if key else str(exc));key='';save();raise


if __name__=='__main__':
    import resource
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--tests',type=Path);p.add_argument('--launch',action='store_true');a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if a.prepare:prepare(a.root,a.tests)
    else:
        if not a.launch or not sys.stdin.isatty():raise ValueError('Hidden interactive launch required')
        if (a.root/'state.json').exists() or (a.root/'continuation_gate_state.json').exists():raise FileExistsError('Inspect an existing run, never relaunch blindly')
        key=getpass.getpass('DashScope API key (hidden, memory only): ').strip()
        if not key or len(key)>4000:raise ValueError('Invalid credential length')
        pid=os.fork()
        if pid:print(json.dumps(dict(conditional_full_master_pid=pid,api_before_validation=0,root=str(a.root))),flush=True)
        else:
            os.setsid()
            with open(os.devnull,'rb') as null,(a.root/'continuation_gate_master.log').open('xb',buffering=0) as log:os.dup2(null.fileno(),0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
            supervise(a.root,key)
