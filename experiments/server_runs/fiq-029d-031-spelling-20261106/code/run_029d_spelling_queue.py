"""Queue the same paired spelling experiment for all FashionIQ categories."""

import argparse
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from attr_transfer_data import official_inputs
from run_029d_gate_refresh import CODE, EXPECTED, FIQ_EXPECTED, PARENT_031, inputs_029, load_training_sources
from run_029d_seed_queue import FIT_COUNTS, run_queue
from run_029d_spelling_pair import CATEGORIES, SPELLINGS, TRAINED, cached_bank, text_manifest, text_views
from run_fiq_native_caption_031 import read, sha, snapshot, write


SEEDS = (42,43,44,45,46)
OFFICIAL_COUNTS = {'dress':EXPECTED['dress'], **{c:v[:2] for c,v in FIQ_EXPECTED.items()}}


def placement(count):
    samples = [snapshot()]
    time.sleep(5)
    samples.append(snapshot())
    occupied = {line.split(',')[0].strip() for s in samples for line in s['compute_apps']}
    first = {d['uuid']:d for d in samples[0]['devices']}
    eligible = []
    for gpu in samples[1]['devices']:
        uid = gpu['uuid']
        if uid in occupied or uid not in first or gpu['index'] not in (2,3):
            continue
        previous = first[uid]
        available = min(gpu['free_mib'],previous['free_mib'])
        if max(gpu['utilization'],previous['utilization']) <= 10 and available >= 12288:
            eligible.append((available,gpu['index'],uid))
    eligible.sort(reverse=True)
    if len(eligible) >= count:
        return [uid for _,_,uid in eligible[:count]],samples
    if eligible and eligible[0][0] >= 20480 and count == 2:
        return [eligible[0][2]]*count,samples
    raise RuntimeError('No suitable placement for the next worker pair')


def check_result(root, category, seed):
    state = read(root/'state.json')
    if (state['status'],state['category'],state['seed'],state['api_calls']) != ('complete',category,seed,0):
        raise ValueError('Task identity or completion state mismatch')
    selections = {}
    nfit = FIT_COUNTS[category]
    for arm in TRAINED:
        fit = read(root/arm/'fit_results.json')
        if fit['updates'] != 3*((nfit+7)//8) or fit['exposures'] != [3]*nfit:
            raise ValueError('Incorrect matched training exposure')
        selected = max(fit['history'],key=lambda h:(h['metrics']['r10']+h['metrics']['r50'],
                                                   -h['metrics']['retrieval_nce'],-h['epoch']))
        if selected != fit['selected'] or sha(root/arm/selected['weight']) != selected['sha256']:
            raise ValueError('Selected checkpoint failed verification')
        selections[arm]={k:selected[k] for k in ('epoch','weight','sha256')}
    path = root/'official_results.json'
    result = read(path)
    n,g = OFFICIAL_COUNTS[category]
    if (result['queries'],result['gallery']) != (n,g) or len(result['rows']) != 28:
        raise ValueError('Expected 28 conditions and the complete official dataset')
    recall = {}
    for name,row in result['rows'].items():
        ranks=row['ranks']
        if len(ranks)!=n or any(type(rank) is not int or not 1<=rank<g for rank in ranks):
            raise ValueError('Invalid official ranking')
        recall[name]={}
        for k in (1,10,50):
            metric=f'r{k}'
            value=100*sum(rank<=k for rank in ranks)/n
            if not math.isclose(value,row[metric]):
                raise ValueError('Recall recomputation failed')
            recall[name][metric]=value
    return dict(category=category,seed=seed,queries=n,gallery=g,status='verified',root=str(root),
                seconds=state['seconds'],selections=selections,result_sha256=sha(path),metrics=recall,
                comparisons=result['comparisons'],dictionary_changed_queries=result['dictionary_changed_queries'])


def summarize(root, runs):
    aggregates={}
    for category in CATEGORIES:
        rows=sorted((r for r in runs if r['category']==category),key=lambda r:r['seed'])
        if not rows:
            continue
        metrics,comparisons={},{}
        for method in rows[0]['metrics']:
            metrics[method]={}
            for k in ('r1','r10','r50'):
                values=[r['metrics'][method][k] for r in rows]
                metrics[method][k]=dict(mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None)
        for method in rows[0]['comparisons']:
            comparisons[method]={}
            for k in ('1','10','50'):
                values=[r['comparisons'][method][k]['net_pp'] for r in rows]
                comparisons[method][k]=dict(mean_pp=statistics.mean(values),sample_std_pp=statistics.stdev(values) if len(values)>1 else None,
                                             wins=sum(v>0 for v in values),ties=sum(v==0 for v in values))
        aggregates[category]=dict(seeds=[r['seed'] for r in rows],metrics=metrics,comparisons=comparisons)
    write(root/'results_summary.json',dict(runs=runs,aggregates=aggregates))


def launch(root):
    if (root/'plan.json').exists() or (root/'queue_state.json').exists():
        raise FileExistsError('Existing queue plan or state')
    if shutil.disk_usage(root).free < 25*(1<<30):
        raise RuntimeError('Insufficient disk space for new checkpoints')
    text_sources,preflight={},{}
    for category in CATEGORIES:
        meta,delta,delta_sha,base,base_sha=load_training_sources(category)
        if sha(delta)!=delta_sha or not base.is_file():
            raise ValueError('Historical checkpoint failed preflight')
        train_bank=cached_bank(category,'train',dict(base=base_sha,metadata=meta),cuda=False)
        if category=='dress':
            rows,ids,hashes,_=official_inputs(category)
            val_bank=cached_bank(category,'val',dict(base=base_sha,rows=rows,ids=ids,hashes=hashes),cuda=False)
        else:
            val,_=inputs_029(category,'val')
            rows,ids=val['records'],val['gallery_ids']
            saved=read(PARENT_031/f'{category}_assets.json')
            val_bank=sha(saved['path'])
            if val_bank!=saved['sha256']:
                raise ValueError('031 validation bank failed preflight')
        if (len(rows),len(ids))!=OFFICIAL_COUNTS[category]:
            raise ValueError('Official denominator changed')
        text_sources[category]=text_manifest(category)
        changed={}
        for phase,source in (('train',meta['records']),('val',rows)):
            mods,_=text_views(category,phase,source)
            changed[phase]=sum(a!=b for a,b in zip(mods[SPELLINGS[0]],mods[SPELLINGS[1]]))
        preflight[category]=dict(train=len(meta['records']),fit=len(meta['offsets']['fit']),
                                 dev=len(meta['offsets']['dev']),val=len(rows),gallery=len(ids),
                                 changed=changed,train_bank_sha256=train_bank,val_bank_sha256=val_bank)
        print(category,preflight[category],flush=True)
    tasks=[dict(category=c,seed=s) for s in SEEDS for c in CATEGORIES]
    batches=[tasks[i:i+2] for i in range(0,len(tasks),2)]
    plan=dict(batches=batches,preflight=preflight,text_manifest=text_sources,api_calls=0,
              max_parallel=2,slots='cached, fixed in both arms',initialization='native DQU per category',
              train_and_evaluation='same punctuation/case normalization; dictionary lookup is the only input switch',
              sources={p.name:sha(p) for p in (root/'code').glob('*.py')})
    write(root/'plan.json',plan)
    env={key:os.environ[key] for key in ('HOME','PATH','LANG') if key in os.environ}
    env['PYTHONPATH']=str(CODE)
    with (root/'queue.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),'--run'],
                               env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'launch.json',dict(pid=child.pid,plan_sha256=sha(root/'plan.json')))
    print(f'Queue PID {child.pid}, {len(tasks)} tasks in {len(batches)} batches',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:
        run_queue(args.root,placement=placement,worker='run_029d_spelling_pair.py',extra_worker_args=[],
                  verifier=check_result,summarize=summarize)
    else:
        launch(args.root)
