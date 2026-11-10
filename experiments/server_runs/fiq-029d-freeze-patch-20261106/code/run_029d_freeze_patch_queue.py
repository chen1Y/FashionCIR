"""Run the four seed-42 freeze and patch tasks on one GPU, two at a time."""

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from attr_transfer_data import official_inputs
from run_029d_freeze_patch import ARMS, CATEGORIES
from run_029d_gate_refresh import CODE, EXPECTED, FIQ_EXPECTED, PARENT_031, SHIRT_TOPTEE, load_training_sources
from run_029d_seed_queue import FIT_COUNTS, run_queue
from run_029d_spelling_pair import cached_bank
from run_fiq_native_caption_031 import read, sha, write


COUNTS = {**EXPECTED, **{c:v[:2] for c,v in FIQ_EXPECTED.items()}}


def verify(root, category, seed):
    state=read(root/'state.json')
    if (state['status'],state['category'],state['seed'],state['api_calls'])!=('complete',category,seed,0) or seed!=42:
        raise ValueError('Task completion or seed mismatch')
    selected={}
    count=FIT_COUNTS[category]
    for arm in ARMS:
        train=read(root/arm/'fit_results.json')
        if train['updates']!=3*((count+7)//8) or train['exposures']!=[3]*count:
            raise ValueError('Mismatched training exposure')
        chosen=max(train['history'],key=lambda h:(h['metrics']['r10']+h['metrics']['r50'],
                                                 -h['metrics']['retrieval_nce'],-h['epoch']))
        if chosen!=train['selected'] or sha(root/arm/chosen['weight'])!=chosen['sha256']:
            raise ValueError('Selection or weight checksum changed')
        selected[arm]=dict(epoch=chosen['epoch'],weight=chosen['weight'],sha256=chosen['sha256'])
    path=root/'official_results.json'
    r=read(path)
    n,g=COUNTS[category]
    if (r['queries'],r['gallery'])!=(n,g) or len(r['rows'])!=11:
        raise ValueError('Incomplete official evaluation')
    if r['params']['frozen_text']['text_transformer']!=0 or r['patch']['images']!=g or r['patch_fit_updates']!=0:
        raise ValueError('Frozen/text or patch protocol mismatch')
    recall={}
    for name,row in r['rows'].items():
        if len(row['ranks'])!=n or any(type(x) is not int or not 1<=x<=g for x in row['ranks']):
            raise ValueError('Rank denominator/geometry changed')
        recall[name]={}
        for k in (1,10,50):
            key=f'r{k}'
            value=100*sum(x<=k for x in row['ranks'])/n
            if not math.isclose(row[key],value):
                raise ValueError('Recall recomputation mismatch')
            recall[name][key]=value
    return dict(category=category,seed=seed,queries=n,gallery=g,metrics=recall,comparisons=r['comparisons'],
                selected=selected,params=r['params'],patch=r['patch'],root=str(root),
                result_sha256=sha(path),seconds=state['seconds'])


def summarize(root, runs):
    write(root/'results_summary.json',dict(runs=runs,seed=42,parallel_gpu=3,patch_fit_updates=0))


def launch(root):
    if (root/'plan.json').exists() or (root/'queue_state.json').exists():
        raise FileExistsError('Run already sealed or started')
    if shutil.disk_usage(root).free<10*(1<<30):
        raise RuntimeError('Insufficient space for paired checkpoints')
    sources={}
    for category in CATEGORIES:
        meta,selected,selected_sha,base,base_sha=load_training_sources(category)
        if sha(selected)!=selected_sha or not base.is_file():
            raise ValueError('Parent checkpoint failed preflight')
        train_sha=cached_bank(category,'train',dict(base=base_sha,metadata=meta),cuda=False)
        if category in SHIRT_TOPTEE:
            saved=read(PARENT_031/f'{category}_assets.json')
            val_sha=sha(saved['path'])
            if val_sha!=saved['sha256']:
                raise ValueError('031 validation bank changed')
        else:
            rows,ids,hashes,_=official_inputs(category)
            val_sha=cached_bank(category,'val',dict(base=base_sha,rows=rows,ids=ids,hashes=hashes),cuda=False)
        sources[category]=dict(fit=len(meta['offsets']['fit']),dev=len(meta['offsets']['dev']),
                               queries=COUNTS[category][0],gallery=COUNTS[category][1],
                               train_bank_sha256=train_sha,val_bank_sha256=val_sha,patch_parent_sha256=selected_sha)
    batches=[[dict(category=c,seed=42) for c in group] for group in (CATEGORIES[:2],CATEGORIES[2:])]
    plan=dict(batches=batches,api_calls=0,gpu_index=3,max_parallel=2,seed=42,inputs=sources,
              freeze_arms=ARMS,patch_weights=[0,0.1,0.2],patch_fit_updates=0,
              sources={p.name:sha(p) for p in (root/'code').glob('*.py')})
    write(root/'plan.json',plan)
    env={key:os.environ[key] for key in ('HOME','PATH','LANG') if key in os.environ}
    env['PYTHONPATH']=str(CODE)
    with (root/'queue.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),'--run'],
                               env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'launch.json',dict(pid=child.pid,plan_sha256=sha(root/'plan.json')))
    print(f'Queue PID {child.pid}; two batches on GPU 3',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:
        run_queue(args.root,worker='run_029d_freeze_patch.py',extra_worker_args=[],verifier=verify,summarize=summarize)
    else:
        launch(args.root)
