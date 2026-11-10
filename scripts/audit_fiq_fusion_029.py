"""029 read-only independent saved-rank, selection, exposure and SHA audit."""
import argparse,json,math,sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'src'))
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from fiq_fusion_029 import ARMS

def run(root):
    plan=json.loads((root/'plan.json').read_text());weights=[];out={}
    for p,h in plan['sources'].items():
        if file_sha(CODE/p)!=h:raise ValueError('029 sealed source changed')
    for p,h in plan['protected'].items():
        if file_sha(p)!=h['sha256']:raise ValueError('029 protected data changed')
    def check(metric,n):
        ranks=metric['ranks']
        if len(ranks)!=n or any(type(x)!=int or x<1 for x in ranks):raise ValueError('029 rank counts invalid')
        for k in (1,10,50):
            if abs(100*sum(r<=k for r in ranks)/n-metric['r'+str(k)])>1e-9:raise ValueError('029 independent metric mismatch')
    for cat in ('shirt','toptee'):
        n=plan['counts'][cat]['fit'];dev=plan['counts'][cat]['dev']
        for arm in ARMS:
            d=json.loads((root/'fits'/cat/arm/'results.json').read_text())
            if d['updates']!=3*math.ceil(n/8) or d['presentations']!=3*n or d['per_fit_row_exposures']!=[3]*n or d['frozen_sha256_before']!=d['frozen_sha256_after'] or not d['old_vocabulary_exact']:raise ValueError('029 training audit failed')
            best=max(d['history'],key=lambda h:((h['metrics']['r10']+h['metrics']['r50'])/2,-h['metrics']['retrieval_nce'],-h['epoch']))
            if best['epoch']!=d['selected_epoch']:raise ValueError('029 selection mismatch')
            for h in d['history']:
                check(h['metrics'],dev)
                if file_sha(h['weight_path'])!=h['weight_sha256']:raise ValueError('029 weight SHA mismatch')
                weights.append(dict(path=h['weight_path'],sha256=h['weight_sha256']))
        official=json.loads((root/('official_'+cat+'.json')).read_text())
        if official['queries']!={'shirt':2038,'toptee':1961}[cat] or official['gallery']!={'shirt':6346,'toptee':5373}[cat] or len(official['rows'])!=6:raise ValueError('029 missing official rows')
        for value in official['rows'].values():check(value,official['queries'])
        out[cat]={name:{k:d[k] for k in ('r1','r10','r50')} for name,d in official['rows'].items()}
    atomic_write_json(root/'verification.json',dict(status='verified',metrics=out,epoch_weights=weights,api_calls=0,model_forwards=0,new_training_updates=0));print(json.dumps(dict(status='verified',metrics=out)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
