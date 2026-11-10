"""Candidate constraint pilot on fixed full-training-gallery neighbors; no GPU or fit."""
import argparse
import base64
import concurrent.futures
import datetime
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from cir_cumulative_budget import CumulativeBudget,micro
from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_shoes_online import Ledger,RateLimiter,BudgetStop
from generate_structured_edits import atomic_write_json,extract_json_object
from shoes_online_pilot import Client,BASE_URL,MODEL,reference_data_url

BATCH='candidate_full_gallery_006_20261026'
CAP=1.5
MAX_CALLS=320
TOKENS=512
from fiq_candidate_schema_probe import PROMPT, validate



def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def judgment(text):
    return validate(text)


def answer_judgment(answer):
    if answer.get('finish_reason')!='stop':
        raise ValueError('Incomplete model response; no paid repair')
    return judgment(answer['text'])


from fiq_candidate_full_math import pair_key,historical_action,rank_after,schema_guard


def metrics(ranks):
    return {**{f'r{k}':100*sum(r<=k for r in ranks)/len(ranks) for k in (1,5,10,20,50)},
            'mrr':sum(1/r for r in ranks)/len(ranks),'queries':len(ranks)}


def inventory(results):
    sources=['cirr_shoes_20261015/online_qwen37_v3/requests.sqlite3',
        'fiq_augmentation_ablation_20261019/fiq_online_requests.sqlite3',
        'fiq_shirt_toptee_20261022/shirt/requests.sqlite3',
        'fiq_shirt_toptee_20261022/toptee/requests.sqlite3']
    entries=[]
    for name in sources:
        path=results/name
        db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':
            raise ValueError('Historical ledger integrity failed')
        states=db.execute('SELECT state,COUNT(*),SUM(charge) FROM requests GROUP BY state').fetchall()
        if any(s in ('reserved','metered') for s,_,_ in states):
            raise ValueError('Historical generator has unsettled in-flight requests')
        upper=sum(v for _,_,v in states);db.close()
        entries.append(dict(id='canonical:'+name,upper_cny=upper,evidence=dict(path=str(path),states=states,backup_duplicates_excluded=True)))
    path=results/'shoes_online_pilot_20261016_run1/summary.json'
    j=json.loads(path.read_text())['combined_cost']
    upper=j['conservative_reserved_cny'] if j['usage_missing_calls'] else j['estimated_list_price_cny']
    entries.append(dict(id='shoes_pilot_combined',upper_cny=upper,evidence=dict(path=str(path),summary=j,parent_cost_not_double_counted=True)))
    entries.append(dict(id='legacy_unreconciled_reserve',upper_cny=50,
        evidence=dict(reason='Conservative reserve for older unmetered/aborted project calls, not claimed actual charges; raise if stronger historical evidence appears')))
    return entries


def old_pairs(args):
    history={};audit=[];inputs={}
    global_db=sqlite3.connect('file:'+str(args.global_budget)+'?mode=ro',uri=True)
    if global_db.execute('SELECT cap FROM config WHERE id=1').fetchone()!=(micro(300),):
        raise ValueError('Require the existing300 cumulative registry')
    for name in ('candidate_check_003','candidate_schema_probe_003b','candidate_check_004'):
        folder=args.research/name
        protocol=json.loads((folder/'protocol.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        db=sqlite3.connect('file:'+str(folder/'requests.sqlite3')+'?mode=ro',uri=True)
        if db.execute('PRAGMA quick_check').fetchone()!=('ok',):raise ValueError('Historical ledger integrity failed')
        states=db.execute('SELECT state,COUNT(*),SUM(charge) FROM requests GROUP BY state').fetchall()
        if any(st in ('reserved','metered') for st,_,_ in states):raise ValueError('Old candidate work still in flight')
        row=global_db.execute('SELECT charged,state FROM batches WHERE id=?',(protocol['batch_id'],)).fetchone()
        upper=sum(v for _,_,v in states)
        if row is None or row[1]!='settled' or row[0]<micro(upper):raise ValueError('Old paid ledger is not covered by cumulative allocation')
        request_rows=db.execute('SELECT id,idx,state,response FROM requests ORDER BY id').fetchall();db.close()
        for request_id,index,state,response in request_rows:
            response=json.loads(response) if response else None
            if 'queries' in manifest:
                query=manifest['queries'][index//20];candidate=query['candidates'][index%20]
                reference=query['reference_image'];image=candidate['path'];instruction=query['instruction']
            else:
                pair=manifest['pairs'][index];reference=pair['reference_image'];image=pair['candidate_image'];instruction=pair['instruction']
            key=pair_key(manifest['image_sha256'][reference],manifest['image_sha256'][image],instruction)
            history.setdefault(key,[]).append(dict(state=state,prompt_sha256=protocol['prompt_sha256'],
                model=protocol.get('model',(response or {}).get('model')),response=response,
                origin=dict(batch_id=protocol['batch_id'],request_id=request_id,state=state,ledger=str(folder/'requests.sqlite3'))))
        audit.append(dict(batch_id=protocol['batch_id'],request_states=states,accounted_upper_cny=upper,already_counted_in_global_batches=True))
        for filename in ('protocol.json','manifest.json','requests.sqlite3'):inputs[str(folder/filename)]=file_sha(folder/filename)
    global_db.close()
    return history,audit,inputs


def prepare(args):
    if args.output.exists():raise FileExistsError('Never overwrite or blindly repeat a candidate pilot')
    require_space(args.output)
    history,history_audit,inputs=old_pairs(args)
    completion=args.research/'train_gallery_diagnostic/completion_summary.json'
    summary=json.loads(completion.read_text())
    if summary['status']!='complete' or summary['new_api_calls']!=0 or summary['source_mismatches']:
        raise ValueError('Require completed verified full training-gallery diagnostic')
    inputs[str(completion)]=file_sha(completion)
    queries=[];images={};evidence={};decisions=dict(new=0,reuse=0,blocked=0)
    prompt_sha=hashlib.sha256(PROMPT.encode()).hexdigest()
    for cat in ('shirt','toptee'):
        folder=args.research/'train_gallery_diagnostic'/cat
        result=json.loads((folder/'results.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
        if (result['status']!='complete' or result['queries']!=256
                or file_sha(folder/'manifest.json')!=summary['categories'][cat]['manifest_sha256']
                or result['counts']!=summary['categories'][cat]['counts']):raise ValueError('Full gallery evidence mismatch')
        image_hashes=manifest['image_sha256']
        if hashlib.sha256(json.dumps(image_hashes,sort_keys=True).encode()).hexdigest()!=result['evidence']['images_sha256']:
            raise ValueError('Full gallery image digest changed')
        gallery=set(manifest['gallery_ids'])
        for index in range(16,24):
            row=manifest['queries'][index];neighbors=result['fixed_dual05_neighbors'][index]
            if ((row['candidate'],row['target'],row['modification_text'])
                    !=(neighbors['candidate'],neighbors['target'],neighbors['modification_text'])
                    or neighbors['query_index']!=index or len(neighbors['candidates'])!=20):
                raise ValueError('Full-gallery query binding changed')
            reference=args.root/'resized_image'/(row['candidate']+'.jpg');images[str(reference)]=image_hashes[row['candidate']]
            candidates=[]
            for candidate in neighbors['candidates']:
                image_id=candidate['image_id'];path=args.root/'resized_image'/(image_id+'.jpg')
                if image_id not in gallery or image_id==row['candidate'] or not math.isfinite(candidate['similarity']):
                    raise ValueError('Invalid candidate in fixed head')
                images[str(path)]=image_hashes[image_id]
                key=pair_key(images[str(reference)],images[str(path)],row['modification_text'])
                action=historical_action(history.get(key,[]),prompt_sha,MODEL);decisions[action['action']]+=1
                candidates.append(dict(candidate_id=image_id,path=str(path),similarity=candidate['similarity'],history=action,pair_key=key))
            rank=neighbors['target_rank'];head=[c['candidate_id'] for c in candidates]
            if (len(set(head))!=20 or rank!=result['routes']['fixed_dual05']['ranks'][index]
                    or (rank<=20 and head[rank-1]!=row['target']) or (rank>20 and row['target'] in head)):
                raise ValueError('Candidate list/annotated target rank binding failed')
            queries.append(dict(query_index=len(queries),category=cat,internal_dev_index=index,source_index=row['source_index'],
                reference_image=str(reference),instruction=row['modification_text'],gold_target_id=row['target'],
                baseline_target_rank=rank,gallery_size=len(gallery),candidates=candidates))
        for filename in ('results.json','manifest.json','protocol.json'):inputs[str(folder/filename)]=file_sha(folder/filename)
        evidence[cat]=dict(full_gallery_evidence=result['evidence'],manifest_sha256=file_sha(folder/'manifest.json'))
    for path,digest in images.items():
        if file_sha(path)!=digest:raise ValueError('Bound reference/candidate bytes changed')
    history_entries=inventory(args.results_root)
    source=Path(__file__).resolve().parent
    dependencies={name:file_sha(source/name) for name in ('fiq_candidate_full_math.py','fiq_candidate_schema_probe.py',
        'fiq_candidate_check.py','cir_cumulative_budget.py','shoes_online_pilot.py','generate_shoes_online.py','generate_structured_edits.py')}
    protocol=dict(batch_id=BATCH,model=MODEL,base_url=BASE_URL,max_cny=CAP,max_calls=MAX_CALLS,max_tokens=TOKENS,
        workers=4,rpm=60,automatic_retries=0,repairs=0,seed=42,k=20,queries_per_category=8,
        sampling='Fixed internal-dev positions16:24, full eligible train-gallery fixed05 top20; no GT/rank filtering or gold insertion',
        ranking='Only top20 permutation: cosine+.03*confidence*(score/4-.5)',
        rows=['baseline','strict_rerank','strict_shuffled','neutral_rerank','neutral_shuffled'],
        failure_policy='Primary strict whole-query fallback; secondary zero-correction per failed candidate, matched valid-position-only shuffle; retain all16 queries',
        prompt_sha256=prompt_sha,source_sha256=file_sha(__file__),dependency_hashes=dependencies,input_hashes=inputs,
        history_decisions=decisions,minimum_schema_coverage=.9,early_schema_guard=dict(after_returned=16,minimum=.9),
        privacy='Uniformly compare reference+retrieved candidate+original modification only. Candidate may be GT; never send IDs, ranks, gallery metadata or gold labels.',
        price_source='https://help.aliyun.com/zh/model-studio/qwen3-7-flash',price_checked='2026-10-26',input_cny_per_million=.2,output_cny_per_million=.8,
        limitation='Same Qwen family is not independent human/expert GT. Train-exposed frozen backbone, tiny fixed internal-dev sample; not official val or significance evidence')
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'manifest.json',dict(queries=queries,image_sha256=images,evidence=evidence))
    protocol['manifest_sha256']=file_sha(args.output/'manifest.json')
    atomic_write_json(args.output/'protocol.json',protocol)
    atomic_write_json(args.output/'historical_inventory.json',dict(canonical_entries=history_entries,paid_pair_batches=history_audit,decisions=decisions))
    atomic_write_json(args.output/'state.json',dict(status='prepared',prepared_utc=now()))
    print(json.dumps(dict(status='prepared',queries=len(queries),cap_cny=CAP,max_calls=MAX_CALLS,history_decisions=decisions)),flush=True)


def verify_bindings(output,protocol):
    source=Path(__file__).resolve().parent
    if file_sha(__file__)!=protocol['source_sha256'] or file_sha(output/'manifest.json')!=protocol['manifest_sha256']:
        raise ValueError('Prepared source or manifest changed')
    if any(file_sha(source/name)!=digest for name,digest in protocol['dependency_hashes'].items()):
        raise ValueError('Sealed API/schema/budget dependencies changed')
    if any(file_sha(path)!=digest for path,digest in protocol['input_hashes'].items()):
        raise ValueError('Sealed full-gallery/historical inputs changed')
    if (protocol['prompt_sha256']!=hashlib.sha256(PROMPT.encode()).hexdigest() or protocol['max_cny']!=CAP
            or protocol['max_calls']!=MAX_CALLS or protocol['max_tokens']!=TOKENS):
        raise ValueError('Prepared prompt or ceilings changed')


class PairClient(Client):
    def ask_pair(self,ref,candidate,text):
        if len(text.encode())+2*4096+1024>16384:
            raise ValueError('Input exceeds conservative two-image bound')
        self.budget.reserve(TOKENS)
        body=dict(model=MODEL,messages=[dict(role='user',content=[
            dict(type='image_url',image_url=dict(url=ref)),dict(type='image_url',image_url=dict(url=candidate)),dict(type='text',text=text)])],
            temperature=0,max_tokens=TOKENS,stream=False,enable_thinking=False,response_format=dict(type='json_object'))
        request=urllib.request.Request(self.endpoint,data=json.dumps(body).encode(),method='POST',
            headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
        try:
            with self.opener.open(request,timeout=60) as response:payload=json.loads(response.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'Official API HTTP {e.code}; no retry') from None
        except (urllib.error.URLError,TimeoutError) as e:
            raise RuntimeError(f'Official API transport error ({type(e).__name__}); no retry') from None
        self.budget.account(payload.get('usage'))
        choice=payload['choices'][0]
        return dict(text=self.safe_text(choice['message'].get('content') or ''),usage=payload.get('usage'),
            model=payload.get('model'),request_id=payload.get('id'),finish_reason=choice.get('finish_reason'))


def worker(output,key_fd,global_budget):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    budget=CumulativeBudget(global_budget)
    row=budget.db.execute('SELECT cap,charged,state,protocol_sha256 FROM batches WHERE id=?',(BATCH,)).fetchone()
    budget.db.close()
    if row!=(micro(CAP),None,'reserved',file_sha(output/'protocol.json')):
        raise ValueError('Worker requires the matching cumulative allocation before any API call')
    protocol=json.loads((output/'protocol.json').read_text());verify_bindings(output,protocol)
    if (output/'requests.sqlite3').exists():raise FileExistsError('Never start a second worker or reopen a partial ledger')
    with os.fdopen(key_fd) as stream:key=stream.read(4096).strip()
    if not key:raise ValueError('Empty credential')
    manifest=json.loads((output/'manifest.json').read_text());protocol=json.loads((output/'protocol.json').read_text())
    ledger=Ledger(output/'requests.sqlite3',protocol,max_cny=CAP,max_calls=MAX_CALLS)
    stop=threading.Event();rate=RateLimiter(60,stop);local=threading.local();failures=[0];failure_lock=threading.Lock()
    tasks=[(q,i,c) for q in manifest['queries'] for i,c in enumerate(q['candidates'])]
    random.Random(42).shuffle(tasks)
    def process(task):
        q,rank,c=task;idx=q['query_index']*20+rank
        if stop.is_set():return
        require_space(output)
        sample=dict(query_index=idx,parent_query_index=q['query_index'],candidate_rank=rank,
            candidate_id=c['candidate_id'],status='failed',judgment=None,gold_label_sent=False)
        try:
            if not hasattr(local,'client'):local.client=PairClient(key,BASE_URL,ledger)
            for path in (q['reference_image'],c['path']):
                if file_sha(path)!=manifest['image_sha256'][path]:raise ValueError('Bound image changed')
            action=c['history'];sample['history_action']=action['action']
            if action['action']=='blocked':
                sample['error']=action['reason'];sample['history_origins']=action['origins']
                ledger.save_sample('internal_dev',sample);return
            if action['action']=='reuse':
                answer=action['response'];sample['history_origin']=action['origin']
                sample['response']=answer
                try:sample['judgment']=answer_judgment(answer);sample['status']='ok'
                except ValueError as e:sample['error']=str(e)
                ledger.save_sample('internal_dev',sample);return
            ref=reference_data_url(q['reference_image']);candidate=reference_data_url(c['path'])
            ledger.begin('internal_dev',idx,'full_gallery_constraint_v1');rate.wait()
            answer=local.client.ask_pair(ref,candidate,PROMPT+q['instruction']);ledger.finish(answer)
            with failure_lock:failures[0]=0
            sample['response']=answer
            try:sample['judgment']=answer_judgment(answer);sample['status']='ok'
            except ValueError as e:sample['error']=str(e)
        except Exception as e:
            ledger.abandon()
            sample['error']=local.client.safe_text(str(e)) if hasattr(local,'client') else type(e).__name__
            with failure_lock:
                failures[0]+=1
                if any(f'HTTP {v}' in str(e) for v in (401,403,404)) or failures[0]>=3 or isinstance(e,(BudgetStop,ValueError)):
                    stop.set()
            if 'HTTP 429' in str(e):rate.backoff()
        ledger.save_sample('internal_dev',sample)
    def save(status):
        samples=ledger.samples('internal_dev');summary=ledger.summary()
        atomic_write_json(output/'generation_state.json',dict(status=status,updated_utc=now(),completed=len(samples),
            valid=sum(s['status']=='ok' for s in samples),cost=summary))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        iterator=iter(tasks);pending={pool.submit(process,next(iterator)) for _ in range(4)}
        while pending:
            done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
            current=ledger.samples('internal_dev')
            if schema_guard(current):
                stop.set()
                returned=[s for s in current if s.get('response') is not None]
                atomic_write_json(output/'early_schema_stop.json',dict(completed=len(current),
                    returned=len(returned),valid_returned=sum(s['status']=='ok' for s in returned),stopped_utc=now(),
                    reason='Predeclared early schema guard; no retries or threshold changes'))
            if not stop.is_set():
                for _ in done:
                    task=next(iterator,None)
                    if task is not None:pending.add(pool.submit(process,task))
            save('running')
    samples=ledger.samples('internal_dev');save('stopped' if stop.is_set() else 'complete')
    if stop.is_set() or len(samples)!=320:
        raise RuntimeError('Candidate pilot incomplete; preserve ledger, no automatic retry')
    indexed={s['query_index']:s for s in samples};categories={}
    for cat in ('shirt','toptee'):
        qs=[q for q in manifest['queries'] if q['category']==cat];rows=dict(baseline=[],strict_rerank=[],strict_shuffled=[],neutral_rerank=[],neutral_shuffled=[]);complete=0
        for q in qs:
            judges=[indexed[q['query_index']*20+i]['judgment'] for i in range(20)]
            complete+=all(x is not None for x in judges)
            rows['baseline'].append(q['baseline_target_rank'])
            rows['strict_rerank'].append(rank_after(q,judges));rows['strict_shuffled'].append(rank_after(q,judges,True))
            rows['neutral_rerank'].append(rank_after(q,judges,neutral=True));rows['neutral_shuffled'].append(rank_after(q,judges,True,True))
        categories[cat]=dict(queries=len(qs),gallery_size=qs[0]['gallery_size'],fully_scored_queries=complete,
            metrics={name:metrics(ranks) for name,ranks in rows.items()},ranks=rows)
    quality_ok=sum(s['status']=='ok' for s in samples)/len(samples)>=.9
    atomic_write_json(output/'results.json',dict(status='complete' if quality_ok else 'quality_failed',categories=categories,cost=ledger.summary(),
        protocol=protocol,interpretation='Full train-gallery tiny fixed internal-dev pilot, not official validation or independent backbone holdout; no GT-based sampling; all queries and all predeclared rows retained, same-model semantic scores are not GT.'))
    ledger.db.close()
    if not quality_ok:
        raise ValueError('Schema coverage below fixed90% threshold; do not expand the pilot')


def run_prepared(args):
    import resource
    import fcntl
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock=(args.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((args.output/'state.json').read_text())
    if state['status']!='prepared':raise ValueError('Inspect existing run; no blind restart')
    protocol=json.loads((args.output/'protocol.json').read_text())
    verify_bindings(args.output,protocol)
    require_space(args.output)
    budget=CumulativeBudget(args.global_budget)
    for row in inventory(args.results_root):budget.import_history(row['id'],row['upper_cny'],row['evidence'])
    budget.reserve(BATCH,CAP,file_sha(args.output/'protocol.json'))
    atomic_write_json(args.output/'global_allocation_before.json',budget.summary());budget.db.close()
    key=getpass.getpass('Qwen API key (memory only): ')
    if not key:raise ValueError('Empty API key')
    if args.background:
        sys.stdout.flush();pid=os.fork()
        if pid:
            print(json.dumps(dict(detached_pid=pid,output=str(args.output))),flush=True);return
        os.setsid();signal.signal(signal.SIGHUP,signal.SIG_IGN)
        with open(os.devnull,'rb') as null,(args.output/'parent.log').open('ab',buffering=0) as log:
            os.dup2(null.fileno(),0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    (args.output/'parent.pid').write_text(str(os.getpid()))
    state.update(status='running',started_utc=now(),parent_pid=os.getpid(),actual_calls_source='requests.sqlite3')
    atomic_write_json(args.output/'state.json',state)
    readfd,writefd=os.pipe();child=None
    try:
        with (args.output/'api.log').open('x') as log:
            child=subprocess.Popen([sys.executable,__file__,'--worker','--output',str(args.output),
                '--global-budget',str(args.global_budget),'--api-key-fd',str(readfd)],
                # The path is not a credential; the secret itself only crosses the anonymous FD.
                pass_fds=(readfd,),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=Path(__file__).resolve().parents[1])
            os.close(readfd);os.write(writefd,key.encode());os.close(writefd);key=''
            state['worker_pid']=child.pid;atomic_write_json(args.output/'state.json',state)
            started=time.monotonic()
            while child.poll() is None:
                require_space(args.output)
                if (args.output/'cancel.request').exists():raise KeyboardInterrupt('User cancelled candidate pilot')
                if time.monotonic()-started>3600:raise TimeoutError('Candidate pilot one-hour deadline')
                time.sleep(5)
        db=sqlite3.connect('file:'+str(args.output/'requests.sqlite3')+'?mode=ro',uri=True)
        inflight=db.execute("SELECT COUNT(*) FROM requests WHERE state IN ('reserved','metered')").fetchone()[0]
        upper=db.execute('SELECT COALESCE(SUM(charge),0) FROM requests').fetchone()[0];db.close()
        budget=CumulativeBudget(args.global_budget);budget.settle(BATCH,upper,no_inflight=inflight==0)
        atomic_write_json(args.output/'global_allocation_after.json',budget.summary());budget.db.close()
        state.update(status='complete' if child.returncode==0 else 'failed',returncode=child.returncode,finished_utc=now(),elapsed_seconds=time.monotonic()-started)
        atomic_write_json(args.output/'state.json',state)
    except BaseException as e:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(status='stopped',error_type=type(e).__name__,error=str(e),finished_utc=now(),full_allocation_retained=True)
        atomic_write_json(args.output/'state.json',state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ'))
    parser.add_argument('--research',type=Path,default=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_small_improvements_20261026'))
    parser.add_argument('--results-root',type=Path,default=Path('/data/8T1/lcy/CIRData/autodl-tmp/results'))
    parser.add_argument('--feature-cache',type=Path,default=Path('/dev/shm/fiq_gate_features_20261026'))
    parser.add_argument('--global-budget',type=Path,default=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3'))
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run-prepared',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--api-key-fd',type=int)
    args=parser.parse_args()
    if sum((args.prepare,args.run_prepared,args.worker))!=1:parser.error('Choose exactly one action')
    if args.prepare:prepare(args)
    elif args.run_prepared:run_prepared(args)
    else:worker(args.output,args.api_key_fd,args.global_budget)


if __name__=='__main__':
    main()
