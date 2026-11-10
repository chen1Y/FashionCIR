"""Independent028 saved-score/weight/source audit; no fitting or model forwards."""
import argparse,json,math
from pathlib import Path
import sys
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'src'))
import torch
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from fiq_gate_patch_math_028 import summax_reference


def audit(root):
    torch.set_num_threads(4);plan=json.loads((root/'plan.json').read_text());verified={};weights=[]
    for name,h in plan['source_hashes'].items():
        if file_sha(CODE/name)!=h:raise ValueError('028 source SHA changed')
    for path,info in plan['protected'].items():
        if file_sha(Path(path))!=info['sha256']:raise ValueError('028 protected SHA changed '+path)
    for cat in ('shirt','toptee'):
        fit=json.loads((root/'fits'/cat/'slots/results.json').read_text());count=plan['counts'][cat]['fit']
        if fit['updates']!=3*math.ceil(count/8) or fit['sample_presentations']!=count*3 or fit['per_fit_row_exposures']!=[3]*count or not fit['old_vocabulary_exact'] or fit['frozen_sha256_before']!=fit['frozen_sha256_after']:raise ValueError('028 training exposure/freeze audit failed')
        best=max(fit['history'],key=lambda h:((h['metrics']['r10']+h['metrics']['r50'])/2,-h['metrics']['retrieval_nce'],-h['epoch']))
        if best['epoch']!=fit['selected_epoch']:raise ValueError('028 selected checkpoint wrong')
        for h in fit['history']:
            if file_sha(Path(h['weight_path']))!=h['weight_sha256']:raise ValueError('028 epoch SHA failed')
            ranks=h['metrics']['ranks']
            for k in (1,10,50):
                if abs(100*sum(r<=k for r in ranks)/len(ranks)-h['metrics']['r'+str(k)])>1e-10:raise ValueError('028 internal dev metric mismatch')
            weights.append(dict(path=h['weight_path'],sha256=h['weight_sha256']))
        initial=json.loads((root/('initial_gate_'+cat+'.json')).read_text());smoke=json.loads((root/('native_'+cat+'_slots.json')).read_text())
        if not initial['all_initial_dev_ranks_exact'] or initial['max_query_delta']>1e-6 or smoke['batch']!=8 or not smoke['old_vocabulary_exact'] or smoke['vit_gradients']:raise ValueError('028 native smoke failed')
        meta=json.loads((root/'metadata'/(cat+'_val.json')).read_text());p=json.loads((root/('patches_'+cat+'.json')).read_text());ids=meta['gallery_ids'];covered=[];first={}
        if p['max_reused_global_delta']>1e-4 or p['global_threshold_unchanged']!=1e-4 or p['ids']!=ids:raise ValueError('028 patch/global binding failed')
        for s in p['shards']:
            if file_sha(Path(s['path']))!=s['sha256']:raise ValueError('028 patch SHA failed')
            data=torch.load(s['path'],map_location='cpu',weights_only=True)
            if data['ids']!=s['ids'] or data['positions']!=s['positions'] or data['plan_sha256']!=file_sha(root/'plan.json') or tuple(data['patches'].shape)!=(len(s['ids']),256,1024):raise ValueError('028 patch shard binding failed')
            for pos,i,v in zip(data['positions'],data['ids'],data['patches']):
                if ids[pos]!=i:raise ValueError('028 patch ID/index mismatch')
                if pos<3:first[pos]=v
            covered+=s['positions']
        if len(covered)!=len(ids) or set(covered)!=set(range(len(ids))):raise ValueError('028 full-gallery coverage failed')
        result=json.loads((root/('results_'+cat+'.json')).read_text())
        if file_sha(Path(result['score_cache_path']))!=result['score_cache_sha256']:raise ValueError('028 score cache changed')
        saved=torch.load(result['score_cache_path'],map_location='cpu',weights_only=True);labels=saved['labels'].tolist();refs=saved['references'].tolist();recomputed={}
        for name,matrix in saved['scores'].items():
            ranks=[]
            for row,label,ref in zip(matrix,labels,refs):
                values=row.tolist();target=values[label]
                if not all(math.isfinite(x) for x in values) or label==ref:raise ValueError('028 nonfinite/ref==GT score')
                ranks.append(1+sum(i!=ref and (x>target or x==target and i<label) for i,x in enumerate(values)))
            reported=result['rows'][name]
            if ranks!=reported['ranks']:raise ValueError('028 independent rank mismatch')
            for k in (1,10,50):
                if abs(100*sum(r<=k for r in ranks)/len(ranks)-reported['r'+str(k)])>1e-10:raise ValueError('028 metric mismatch')
            recomputed[name]=ranks
        for t in result['transitions']:
            x=recomputed[t['a']];y=recomputed[t['b']];k=t['k'];g=[i for i,(a,b) in enumerate(zip(x,y)) if a<=k<b];l=[i for i,(a,b) in enumerate(zip(x,y)) if b<=k<a]
            if g!=t['gain_indices'] or l!=t['loss_indices'] or t['gains']!=len(g) or t['losses']!=len(l) or abs(t['delta_pp']-100*(len(g)-len(l))/len(x))>1e-10:raise ValueError('028 transition mismatch')
        ix=next(i for i,v in enumerate(saved['valid']) if v.any());scalar=[]
        for pos in range(3):
            exact=summax_reference(saved['slots'][ix].tolist(),saved['valid'][ix].tolist(),first[pos].tolist());actual=float(saved['scores']['local_summax'][ix,pos]);error=abs(exact-actual)
            if error>1e-5:raise ValueError('028 independent scalar cosine formula differs')
            scalar.append(dict(query=ix,gallery_index=pos,id=ids[pos],python_scalar=exact,saved_score=actual,error=error))
        no_valid=~saved['valid'].any(1)
        if not torch.equal(saved['scores']['local_summax'][no_valid],saved['scores']['baseline_slots'][no_valid]) or not torch.equal(saved['scores']['global_plus_local'][no_valid],saved['scores']['baseline_slots'][no_valid]):raise ValueError('028 fallback query dropped/changed')
        gate=saved['gate_weights'];valid=saved['valid']
        if not torch.isfinite(gate).all() or (gate[~valid]!=0).any() or ((gate.sum(1)[~no_valid]-1).abs()>1e-5).any():raise ValueError('028 gate normalization/masking failed')
        verified[cat]=dict(queries=len(labels),gallery=len(ids),score_rows=len(recomputed),all_ranks_metrics_transitions_exact=True,full_patch_coverage=True,scalar_formula_checks=scalar,selected_epoch=fit['selected_epoch'],freeze_and_exposure=True)
    value=dict(status='verified',categories=verified,all_epoch_weights=weights,source_files=len(plan['source_hashes']),protected_files=len(plan['protected']),api_calls=0,model_forwards=0,fit_updates=0)
    atomic_write_json(root/'verification.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();audit(a.root)
