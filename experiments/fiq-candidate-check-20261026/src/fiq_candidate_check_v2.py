"""Fresh train-internal-dev candidate pilot with explicit schema and early quality stop."""
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

BATCH='candidate_check_004_20261026'
CAP=1.5
MAX_CALLS=320
TOKENS=512
from fiq_candidate_schema_probe import PROMPT, validate



def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def judgment(text):
    return validate(text)


def schema_guard(samples):
    return len(samples)>=8 and sum(s['status']=='ok' for s in samples)/len(samples)<.9


def rank_after(query,judgments,shuffled=False):
    baseline=query['baseline_target_rank']
    if len(judgments)!=20 or any(x is None for x in judgments):
        return baseline
    penalties=[x['confidence']*(x['overall_score']/4-.5) for x in judgments]
    if shuffled:
        random.Random(42+query['query_index']).shuffle(penalties)
    order=sorted(range(20),key=lambda i:-(query['candidates'][i]['similarity']+.03*penalties[i]))
    for rank,i in enumerate(order,1):
        if query['candidates'][i]['candidate_id']==query['gold_target_id']:
            return rank
    return baseline


def metrics(ranks):
    return {**{f'r{k}':100*sum(r<=k for r in ranks)/len(ranks) for k in (1,5,10,20)},
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


def prepare(args):
    import torch
    from fiq_small_math import blend
    if args.output.exists():
        raise FileExistsError('Inspect existing candidate pilot; do not overwrite or repeat')
    require_space(args.output)
    queries=[];images={};evidence={}
    for cat in ('shirt','toptee'):
        folder=args.research/'train_only_route_gate'/cat
        protocol=json.loads((folder/'protocol.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        assert len(manifest['fit'])==1024 and len(manifest['internal_dev'])==256
        expected=protocol['evidence'];matches=[]
        for file in args.feature_cache.glob(cat+'_*.pt'):
            cache=torch.load(file,map_location='cpu',weights_only=True)
            if cache['evidence']==expected:matches.append((file,cache))
        if len(matches)!=1:
            raise ValueError('Require exact verified train-only feature cache')
        file,cache=matches[0];nfit=len(manifest['fit']);dev=manifest['internal_dev']
        if cache['source_ids'][nfit:]!=[r['candidate'] for r in dev] or cache['target_ids'][nfit:]!=[r['target'] for r in dev]:
            raise ValueError('Cache and internal-dev query identity mismatch')
        gallery_ids=sorted(set(cache['target_ids'][nfit:]))
        first={}
        for i,image_id in enumerate(cache['target_ids'][nfit:]):first.setdefault(image_id,nfit+i)
        gallery=torch.stack([cache['targets'][first[i]] for i in gallery_ids])
        feature=blend(cache['qd'][nfit+8:nfit+16],cache['qq'][nfit+8:nfit+16],.5)
        scores=feature@gallery.T
        for row_idx,row in enumerate(dev[8:16]):
            if row['candidate'] in first:scores[row_idx,gallery_ids.index(row['candidate'])]=-torch.inf
            order=torch.argsort(scores[row_idx],descending=True).tolist()
            if len(order)<20:raise ValueError('Internal gallery too small')
            ref=args.root/'resized_image'/(row['candidate']+'.jpg')
            images[str(ref)]=file_sha(ref)
            candidates=[]
            for index in order[:20]:
                image_id=gallery_ids[index];path=args.root/'resized_image'/(image_id+'.jpg');images[str(path)]=file_sha(path)
                candidates.append(dict(candidate_id=image_id,path=str(path),similarity=float(scores[row_idx,index])))
            queries.append(dict(query_index=len(queries),category=cat,source_index=row['source_index'],
                reference_image=str(ref),instruction=row['modification_text'],gold_target_id=row['target'],
                baseline_target_rank=order.index(gallery_ids.index(row['target']))+1,
                gallery_size=len(gallery_ids),candidates=candidates))
        evidence[cat]=dict(cache=str(file),cache_evidence=expected,gate_manifest_sha256=file_sha(folder/'manifest.json'))
    # Refuse all image/instruction pairs covered by the original batch or schema probe,
    # including interrupted/unknown requests; selecting fresh queries is predeclared.
    prior=args.research/'candidate_check_003'/'manifest.json'
    previous=json.loads(prior.read_text())
    blocked={(q['reference_image'],q['instruction'],c['path'])
             for q in previous['queries'] for c in q['candidates']}
    for q in queries:
        for c in q['candidates']:
            if (q['reference_image'],q['instruction'],c['path']) in blocked:
                raise ValueError('Fresh manifest overlaps an earlier attempted/unknown pair')
    history=inventory(args.results_root)
    protocol=dict(batch_id=BATCH,model=MODEL,base_url=BASE_URL,max_cny=CAP,max_calls=MAX_CALLS,
        workers=4,rpm=60,max_tokens=TOKENS,automatic_retries=0,repairs=0,seed=42,queries_per_category=8,k=20,
        ranking='strict top20 permutation: cosine + .03*confidence*(overall_score/4-.5)',
        failures='Any invalid/missing candidate judgment falls back whole query; no query deletion',
        sampling='positions8:16 of fixed image-disjoint train-internal-dev manifest; no gold-rank selection or gold injection',
        privacy='API gets reference image + uniformly scored retrieved candidate image + original modification only; no IDs/ranks/gold labels. A retrieved candidate may be gold.',
        image_max_side=512,query_input_bound_tokens=16384,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        source_sha256=file_sha(__file__),budget_source_sha256=file_sha(Path(__file__).with_name('cir_cumulative_budget.py')),
        transport_source_sha256=file_sha(Path(__file__).with_name('shoes_online_pilot.py')),
        ledger_source_sha256=file_sha(Path(__file__).with_name('generate_shoes_online.py')),
        minimum_schema_coverage=.9,early_schema_guard=dict(after_completed=8,minimum=.9),
        schema_probe_source_sha256=file_sha(Path(__file__).with_name('fiq_candidate_schema_probe.py')),
        original_candidate_source_sha256=file_sha(Path(__file__).with_name('fiq_candidate_check.py')),
        prior_manifest_sha256=file_sha(prior),
        price_source='https://help.aliyun.com/zh/model-studio/qwen3-7-flash',price_checked='2026-10-26',input_cny_per_million=.2,output_cny_per_million=.8,
        limitation='Same Qwen family as generator, not independent model/human GT; small train-internal-dev gallery, not official-val retrieval improvement')
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'manifest.json',dict(queries=queries,image_sha256=images,evidence=evidence))
    protocol['manifest_sha256']=file_sha(args.output/'manifest.json')
    atomic_write_json(args.output/'protocol.json',protocol)
    atomic_write_json(args.output/'historical_inventory.json',dict(entries=history))
    atomic_write_json(args.output/'state.json',dict(status='prepared',prepared_utc=now(),api_calls=0))
    print(json.dumps(dict(status='prepared',queries=len(queries),requests=len(queries)*20,cap_cny=CAP)),flush=True)


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
            ref=reference_data_url(q['reference_image']);candidate=reference_data_url(c['path'])
            ledger.begin('internal_dev',idx,'candidate_check_v2');rate.wait()
            answer=local.client.ask_pair(ref,candidate,PROMPT+q['instruction']);ledger.finish(answer)
            with failure_lock:failures[0]=0
            sample['response']=answer
            try:sample['judgment']=judgment(answer['text']);sample['status']='ok'
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
                atomic_write_json(output/'early_schema_stop.json',dict(completed=len(current),
                    valid=sum(s['status']=='ok' for s in current),stopped_utc=now(),
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
        qs=[q for q in manifest['queries'] if q['category']==cat];rows=dict(baseline=[],vlm_rerank=[],shuffled_control=[]);complete=0
        for q in qs:
            judges=[indexed[q['query_index']*20+i]['judgment'] for i in range(20)]
            complete+=all(x is not None for x in judges)
            rows['baseline'].append(q['baseline_target_rank']);rows['vlm_rerank'].append(rank_after(q,judges));rows['shuffled_control'].append(rank_after(q,judges,True))
        categories[cat]=dict(queries=len(qs),gallery_size=qs[0]['gallery_size'],fully_scored_queries=complete,
            metrics={name:metrics(ranks) for name,ranks in rows.items()},ranks=rows)
    quality_ok=sum(s['status']=='ok' for s in samples)/len(samples)>=.9
    atomic_write_json(output/'results.json',dict(status='complete' if quality_ok else 'quality_failed',categories=categories,cost=ledger.summary(),
        protocol=protocol,interpretation='Tiny train-internal-dev pilot, not official FashionIQ validation; no target-rank-based sampling; all queries retained with failure fallback; same-model semantic scores are not GT.'))
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
    if protocol['source_sha256']!=file_sha(__file__) or protocol['budget_source_sha256']!=file_sha(Path(__file__).with_name('cir_cumulative_budget.py')):
        raise ValueError('Prepared source changed')
    if (protocol['manifest_sha256']!=file_sha(args.output/'manifest.json')
            or protocol['transport_source_sha256']!=file_sha(Path(__file__).with_name('shoes_online_pilot.py'))
            or protocol['ledger_source_sha256']!=file_sha(Path(__file__).with_name('generate_shoes_online.py'))
            or protocol['prompt_sha256']!=hashlib.sha256(PROMPT.encode()).hexdigest()
            or protocol['schema_probe_source_sha256']!=file_sha(Path(__file__).with_name('fiq_candidate_schema_probe.py'))
            or protocol['original_candidate_source_sha256']!=file_sha(Path(__file__).with_name('fiq_candidate_check.py'))
            or protocol['prior_manifest_sha256']!=file_sha(args.research/'candidate_check_003'/'manifest.json')
            or protocol['max_cny']!=CAP or protocol['max_calls']!=MAX_CALLS or protocol['max_tokens']!=TOKENS):
        raise ValueError('Prepared protocol, manifest or API dependencies changed')
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
    state.update(status='running',started_utc=now(),parent_pid=os.getpid())
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
