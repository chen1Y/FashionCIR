"""030 finite query-only ambiguity generation; secrets hidden into memory only."""
import argparse,concurrent.futures,getpass,hashlib,json,os,resource,shutil,sqlite3,sys,threading,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(CODE/'src'),str(CODE/'scripts')]
from fiq_relative_schema_030 import SLOTS,PROMPT,eligible,request_key,prompt,validate
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json,extract_json_object
from experiment_storage import require_space
from run_attr_pair_batch import identity

PARENT=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023');BUDGET=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3');FILES=('src/fiq_relative_schema_030.py','scripts/generate_fiq_relative_030.py','tests/test_fiq_relative_030.py')


def rows_for(phase):
    rows=[]
    for cat in ('shirt','toptee'):
        meta=json.loads((PARENT/'metadata'/(cat+'_'+phase+'.json')).read_text());old=json.loads((PARENT/'texts'/(cat+'_'+phase+'.json')).read_text())['samples']
        for i,(r,s) in enumerate(zip(meta['records'],old)):
            if r['source_index']!=s['source_index'] or r['candidate']!=s['reference'] or r['modification_text']!=s['instruction']:raise ValueError('030 inherited query mismatch')
            todo=eligible(r['modification_text']) and s['status']=='ok';sha=meta['image_sha256'][r['candidate']]
            rows.append(dict(category=cat,phase=phase,index=i,source_index=r['source_index'],reference=r['candidate'],instruction=r['modification_text'],base_attrs=s['target_attrs'],reference_sha256=sha,reference_path='/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/resized_image/'+r['candidate']+'.jpg',eligible=todo,reason=None if todo else ('old_failed_or_unknown_blocked' if s['status']!='ok' else 'no_supported_comparative'),request_key=request_key(sha,r['modification_text'],s['target_attrs'])))
    return rows


def check_budget():
    if not BUDGET.is_file():raise ValueError('030 original global budget missing; cannot create/reset')
    d=sqlite3.connect('file:'+str(BUDGET)+'?mode=ro',uri=True)
    try:
        if d.execute('select cap from config where id=1').fetchone()!=(300000000,):raise ValueError('030 original global cap changed')
        if d.execute("select count(*) from batches where state!='settled'").fetchone()[0]:raise ValueError('030 another paid batch is unsettled')
        return (d.execute('select coalesce(sum(upper),0) from history').fetchone()[0]+d.execute('select coalesce(sum(coalesce(charged,cap)),0) from batches').fetchone()[0])/1e6
    finally:d.close()


def prepare(root,tests):
    if root.exists():raise FileExistsError('030 root exists, inspect instead')
    if not tests.read_text().rstrip().endswith('OK'):raise ValueError('030 tests missing')
    require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);before=check_budget();rows=rows_for('train');todo={}
    for r in rows:
        if r['eligible']:todo.setdefault(r['request_key'],r)
    sources=dict(json.loads((Path('/data/8T1/lcy/CIRData/runs/fiq-fusion-029')/'plan.json').read_text())['sources']);sources.update({n:file_sha(CODE/n) for n in FILES});protected={}
    for cat in ('shirt','toptee'):
        for phase in ('train','val'):
            for folder in ('metadata','texts'):
                p=PARENT/folder/(cat+'_'+phase+'.json');protected[str(p)]=file_sha(p)
    root.mkdir();(root/'texts').mkdir();plan=dict(id='030_relative_uncertainty_generation',sources=sources,protected=protected,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),model='qwen3.7-flash',max_tokens=1024,temperature=0,workers=4,rpm=60,global_cap=300,previous_accounted_upper_cny=before,per_batch_reservation_cny=5,max_calls_per_batch=1000,first_format_batch=8,min_program_coverage=.8,automatic_retries=0,known_invalid_repair=False,old_failed_unknown_rows_blocked=True,unique_train_requests=len(todo),all_train_rows=len(rows),reference_only=True,target_images_sent=False,recurring_monitor=False,max_batch_seconds=1800,max_phase_seconds=7200,source_semantics_not_human_gt=True)
    atomic_write_json(root/'generation_plan.json',plan);atomic_write_json(root/'request_manifest_train.json',dict(rows=rows,unique_eligible=len(todo),all_queries_retained=True));shutil.copyfile(tests,root/'generation_tests.log');print(json.dumps(dict(status='prepared030',train_unique_requests=len(todo),all_train_rows=len(rows),accounted_upper_cny=before)),flush=True)


def generate(root,phase,key):
    from expanded_attr_resume_023 import DiagnosticClient
    from shoes_online_pilot import BASE_URL,reference_data_url
    from generate_shoes_online import Ledger,RateLimiter
    from cir_cumulative_budget import CumulativeBudget
    import open_clip
    start=time.monotonic();plan=json.loads((root/'generation_plan.json').read_text());statepath=root/('generation_state_'+phase+'.json')
    if statepath.exists():raise FileExistsError('030 phase already started; never replay requests')
    if phase=='val' and not (root/'fit_complete.json').exists():raise ValueError('030 all direction fits before val generation')
    rows=rows_for(phase);unique={}
    for row in rows:
        if row['eligible']:unique.setdefault(row['request_key'],dict(row,request_index=len(unique)))
    atomic_write_json(root/('request_manifest_'+phase+'.json'),dict(rows=rows,unique_eligible=len(unique),all_queries_retained=True));state=dict(status='generating',identity=identity(os.getpid()),phase=phase,planned_unique=len(unique),returned=0,valid=0);answers={};generation=root/'generation';generation.mkdir(exist_ok=True);tok=open_clip.get_tokenizer('ViT-H-14')
    def guard():
        require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8)
        if (root/'cancel.request').exists():raise KeyboardInterrupt('030 cancelled')
        if time.monotonic()-start>7200:raise TimeoutError('030 generation phase exceeds2h')
        other=Path('/data/8T1/lcy/CIRData/runs/fiq-fusion-029/state.json')
        if other.exists() and json.loads(other.read_text())['status'] in ('failed','cancelled'):raise RuntimeError('029 failed; stop030 generation')
        for n,h in plan['sources'].items():
            if file_sha(CODE/n)!=h:raise ValueError('030 generation sealed source changed')
        for n,h in plan['protected'].items():
            if file_sha(n)!=h:raise ValueError('030 legacy text/metadata changed')
    def save():atomic_write_json(statepath,state)
    try:
        guard();save();todo=list(unique.values());chunks=[todo[:8]]+[todo[j:j+1000] for j in range(8,len(todo),1000)] if todo else []
        for bi,batch in enumerate(chunks):
            guard();check_budget();directory=generation/(phase+'_'+str(bi).zfill(3))
            if directory.exists():raise FileExistsError('030 existing batch; no paid replay')
            directory.mkdir();batch_id='fiq_relative030_'+phase+'_'+str(bi).zfill(3);protocol=dict(id=batch_id,model=plan['model'],prompt_sha256=plan['prompt_sha256'],max_tokens=1024,temperature=0,rows=batch,cap_cny=5,max_calls=len(batch),reference_only=True,targets_sent=False,automatic_retries=0);atomic_write_json(directory/'protocol.json',protocol);budget=CumulativeBudget(BUDGET);budget.reserve(batch_id,5,file_sha(directory/'protocol.json'));atomic_write_json(directory/'global_before.json',budget.summary());budget.db.close();ledger=Ledger(directory/'requests.sqlite3',protocol,max_cny=5,max_calls=len(batch));stop=threading.Event();rate=RateLimiter(60,stop);local=threading.local();done={};mutex=threading.Lock();began=time.monotonic();error=None
            def task(row):
                if stop.is_set():return
                sample=dict(request_key=row['request_key'],query_index=row['request_index'],status='fallback',semantic=None,response=None,errors=[])
                try:
                    guard()
                    if not hasattr(local,'client'):local.client=DiagnosticClient(key,BASE_URL,ledger)
                    if file_sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('030 reference JPEG changed')
                    ledger.begin(phase,row['request_index'],'relative_uncertainty030');rate.wait();response=local.client.ask(reference_data_url(row['reference_path']),prompt(row['instruction'],row['base_attrs']),1024);ledger.finish(response);sample['response']=response
                    if response['finish_reason']!='stop':sample['errors']=['incomplete_output']
                    else:
                        payload=extract_json_object(response['text']);sample['generated_payload']=payload;sample['errors']=validate(payload,row['instruction'],row['base_attrs'],tok.encode)
                        if not sample['errors']:sample.update(status='ok',semantic=payload)
                except BaseException as exc:
                    ledger.abandon();sample['errors']=[local.client.safe_text(str(exc)) if hasattr(local,'client') else type(exc).__name__]
                    if sample['response'] is None:stop.set()
                ledger.save_sample(phase,sample)
                with mutex:done[row['request_key']]=sample
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    iterator=iter(batch);pending={pool.submit(task,next(iterator)) for _ in range(min(4,len(batch)))}
                    while pending:
                        completed,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                        for f in completed:f.result()
                        returned=[v for v in done.values() if v['response'] is not None]
                        if len(returned)>=16 and sum(v['status']=='ok' for v in returned)/len(returned)<.8:stop.set()
                        if time.monotonic()-began>1800:stop.set()
                        state.update(batch=bi,returned=sum(v['response'] is not None for v in answers.values())+len(returned),valid=sum(v['status']=='ok' for v in answers.values())+sum(v['status']=='ok' for v in returned));save()
                        if not stop.is_set():
                            for _ in completed:
                                r=next(iterator,None)
                                if r is not None:pending.add(pool.submit(task,r))
                if any(v['response'] is None for v in done.values()) or len(done)!=len(batch):raise RuntimeError('030 operational stop; preserve unknown and do not retry')
                if sum(v['status']=='ok' for v in done.values())/len(batch)<.8:raise RuntimeError('030 program coverage below80%; no relaxation/retry')
            except BaseException as e:error=e
            finally:
                cost=ledger.summary();inflight=ledger.db.execute("select count(*) from requests where state in ('reserved','metered')").fetchone()[0];budget=CumulativeBudget(BUDGET);budget.settle(batch_id,cost['accounted_upper_cny'],inflight==0);atomic_write_json(directory/'global_after.json',budget.summary());budget.db.close();atomic_write_json(directory/'responses.json',list(done.values()));atomic_write_json(directory/'completion.json',dict(status='failed' if error else 'complete',cost=cost,seconds=time.monotonic()-began,error=str(error).replace(key,'[REDACTED]') if error else None));ledger.db.close()
            if error:raise error
            answers.update(done)
        for cat in ('shirt','toptee'):
            selected=[]
            for row in rows:
                if row['category']!=cat:continue
                answer=answers.get(row['request_key']);selected.append(dict(source_index=row['source_index'],index=row['index'],reference=row['reference'],instruction=row['instruction'],base_attrs=row['base_attrs'],request_key=row['request_key'],status=answer['status'] if answer else 'fallback',semantic=answer['semantic'] if answer else None,errors=answer['errors'] if answer else [row['reason']]))
            target=root/'texts'/(cat+'_'+phase+'_relative.json')
            if target.exists():raise FileExistsError('030 immutable semantic output exists')
            atomic_write_json(target,dict(samples=selected,prompt_sha256=plan['prompt_sha256'],model=plan['model'],hypotheses_are_ground_truth=False,all_queries_retained=True))
        state.update(status='complete',returned=len(answers),valid=sum(v['status']=='ok' for v in answers.values()),hypothesis_queries=sum(bool(v['semantic'] and v['semantic']['hypotheses']) for v in answers.values()),seconds=time.monotonic()-start);save()
    except BaseException as e:state.update(status='failed',error=type(e).__name__+': '+str(e).replace(key,'[REDACTED]'),seconds=time.monotonic()-start);save();raise


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tests',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--launch-phase',choices=('train','val'));a=p.parse_args()
    if a.prepare:prepare(a.root,a.tests)
    elif a.launch_phase:
        if (a.root/('generation_state_'+a.launch_phase+'.json')).exists():raise FileExistsError('030 phase already started')
        key=getpass.getpass('DashScope API key (hidden; memory only): ')
        if not key:raise ValueError('Empty key')
        pid=os.fork()
        if pid:print(json.dumps(dict(status='launched030_generation',phase=a.launch_phase,pid=pid)),flush=True)
        else:
            os.setsid();fd=os.open(a.root/('generation_'+a.launch_phase+'.log'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.dup2(fd,1);os.dup2(fd,2);os.close(0);sys.stdin=open(os.devnull);generate(a.root,a.launch_phase,key);os._exit(0)
    else:p.error('prepare or launch-phase required')
