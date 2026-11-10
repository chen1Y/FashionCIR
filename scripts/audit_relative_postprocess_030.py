"""030-P independent saved-rank/selection/frozen-source audit; no model forward."""
import argparse,json,math,sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'src'))
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
ARMS=('anchor_all','anchor_source','polarity_axis','target_minus_source')
def run(root):
    plan=json.loads((root/'plan.json').read_text());weights=[];out={}
    for n,h in plan['sources'].items():
        if file_sha(CODE/n)!=h:raise ValueError('030-P source changed')
    for p,h in plan['protected'].items():
        if file_sha(p)!=h['sha256']:raise ValueError('030-P protected input changed')
    def check(d,n):
        if len(d['ranks'])!=n or any(type(r)!=int or r<1 for r in d['ranks']):raise ValueError('030-P ranks')
        for k in (1,10,50):
            if abs(d['r'+str(k)]-100*sum(r<=k for r in d['ranks'])/n)>1e-9:raise ValueError('030-P metrics')
    for cat,n in (('shirt',972),('toptee',991)):
        for arm in ARMS:
            d=json.loads((root/'fits'/cat/arm/'results.json').read_text())
            if d['exposure']!=[3]*n or d['updates']!=3*math.ceil(n/8) or d['presentations']!=3*n or not d['parent_frozen'] or not d['initial029_ranks_exact']:raise ValueError('030-P fit verification')
            best=max(d['history'],key=lambda h:((h['metrics']['r10']+h['metrics']['r50'])/2,-h['metrics']['retrieval_nce'],-h['epoch']))
            if best['epoch']!=d['selected_epoch']:raise ValueError('030-P selection')
            for h in d['history']:
                check(h['metrics'],249)
                if file_sha(h['path'])!=h['sha256']:raise ValueError('030-P epoch SHA')
                if any(not 0<=x<=.25000001 for x in h['blend']) or any(not 0<=x<=.30000002 for x in h['gamma']):raise ValueError('030-P parameter bounds')
                weights.append(dict(path=h['path'],sha256=h['sha256']))
        d=json.loads((root/('official_'+cat+'.json')).read_text());nq={'shirt':2038,'toptee':1961}[cat]
        if d['queries']!=nq or len(d['rows'])!=7 or not d['baseline029_all_ranks_exact'] or d['failed_online_responses_used']:raise ValueError('030-P official coverage')
        for value in d['rows'].values():check(value,nq)
        parent=json.loads((Path(plan['parent'])/('official_'+cat+'.json')).read_text())['rows']['full_learned_written']['ranks']
        if d['rows']['baseline029']['ranks']!=parent:raise ValueError('030-P parent rank equality')
        for c in d['comparisons']:
            a=d['rows'][c['a']]['ranks'];b=d['rows'][c['b']]['ranks'];k=c['k'];g=[i for i,(u,v) in enumerate(zip(a,b)) if u<=k<v];l=[i for i,(u,v) in enumerate(zip(a,b)) if v<=k<u]
            if g!=c['gain_indices'] or l!=c['loss_indices'] or abs(c['delta_pp']-100*(len(g)-len(l))/nq)>1e-10:raise ValueError('030-P transitions')
        for group in d['strata'].values():
            ix=group['indices']
            if len(ix)!=group['queries'] or len(set(ix))!=len(ix):raise ValueError('030-P stratum indices')
            for name,metric in group['metrics'].items():
                for k in (1,10,50):
                    expected=100*sum(d['rows'][name]['ranks'][i]<=k for i in ix)/len(ix) if ix else None
                    if metric['r'+str(k)]!=expected:raise ValueError('030-P stratum metrics')
        for phase in ('train','val'):
            a=json.loads((root/('data_'+cat+'_'+phase+'.json')).read_text())
            if file_sha(a['path'])!=a['sha256'] or a['new_api_calls'] or a['new_image_forwards'] or a['failed_online_outputs_used']:raise ValueError('030-P cache/provenance')
        out[cat]={name:{k:v[k] for k in ('r1','r10','r50')} for name,v in d['rows'].items()}
    atomic_write_json(root/'verification.json',dict(status='verified',epoch_weights=weights,metrics=out,new_api_calls=0,image_forwards=0,online030_still_failed=True));print(json.dumps(dict(status='verified',metrics=out)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
