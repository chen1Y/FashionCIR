"""Full-gallery028 scoring: matched gate and exact frozen-token sum/max ablation."""
import argparse,gc,json,time
from pathlib import Path
import torch
import train_expanded_attr as engine
from expanded_attr_model import PairAlignedText,branch_forward,compose,metrics,ranks_from_scores
from fiq_gate_patch_model_028 import GatedAttributeText,branch_forward as gated_branch,compose as gated_compose,local_scores
from fiq_patch_bank_028 import CACHE,PARENT
from train_attr_slot_pilot import load_clip
from fashioniq_noise import file_sha
from experiment_storage import atomic_torch_save,require_space
from generate_structured_edits import atomic_write_json

NAMES=('baseline_slots','gated_slots','local_summax','global_plus_local')


def transitions(rows):
    result=[]
    for a,b in (('gated_slots','baseline_slots'),('local_summax','baseline_slots'),('global_plus_local','baseline_slots'),('global_plus_local','local_summax')):
        for k in (1,10,50):
            x=rows[a]['ranks'];y=rows[b]['ranks'];gain=[i for i,(s,t) in enumerate(zip(x,y)) if s<=k<t];loss=[i for i,(s,t) in enumerate(zip(x,y)) if t<=k<s]
            result.append(dict(a=a,b=b,k=k,queries=len(x),gains=len(gain),losses=len(loss),gain_indices=gain,loss_indices=loss,both=sum(s<=k and t<=k for s,t in zip(x,y)),neither=sum(s>k and t>k for s,t in zip(x,y)),delta_pp=100*(len(gain)-len(loss))/len(x)))
    return result


def run(root,cat):
    start=time.monotonic();torch.set_num_threads(4)
    for c in ('shirt','toptee'):
        if not (root/'fits'/c/'slots/fit_complete.json').exists():raise ValueError('Both gate fits must be sealed before official queries')
    output=root/('results_'+cat+'.json')
    if output.exists():raise FileExistsError('028 scores never blindly rerun')
    plan=json.loads((root/'plan.json').read_text());context=engine.load_context(root,cat,'val');bank=context['banks']['val'];n=len(context['rows']);g=len(bank['ids']);scores={};features={};weights=None
    def guard():
        require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',2)
        if (root/'cancel.request').exists():raise KeyboardInterrupt('028 cancelled')
        if time.monotonic()-start>3300:raise TimeoutError('028 scoring exceeds3300s')
    for name,gated in (('baseline_slots',False),('gated_slots',True)):
        guard();fitroot=root if gated else PARENT;fit=json.loads((fitroot/'fits'/cat/'slots/results.json').read_text());path=Path(fit['selected_weight'])
        if file_sha(path)!=fit['selected_weight_sha256']:raise ValueError('028 selected weight changed')
        clip,_,tok=load_clip(engine.base_path(cat));model=(GatedAttributeText if gated else PairAlignedText)(clip,tok,2).cuda();model.load_delta(torch.load(path,map_location='cpu',weights_only=True)['state']);model.eval()
        def forbid(*args):raise RuntimeError('028 scoring may not forward query images')
        hook=clip.visual.register_forward_pre_hook(forbid);matrix=torch.empty(n,g);slots=[];valid=[];gate=[]
        with torch.no_grad():
            for j in range(0,n,8):
                guard();ix=torch.arange(j,min(j+8,n),device='cuda');batch=[context['rows'][i] for i in ix.tolist()]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=(gated_branch if gated else branch_forward)(model,batch,'slots');q=(gated_compose if gated else compose)(out,context['reference'][ix],'slots')
                matrix[j:j+len(ix)]=(q@bank['gallery'].T).cpu()
                if gated:gate.append(out['gate_weights'].float().cpu())
                else:slots.append(out['slots'].float().cpu());valid.append(out['valid'].cpu())
        if not torch.isfinite(matrix).all():raise FloatingPointError('Nonfinite028 global scores')
        scores[name]=matrix
        if gated:weights=torch.cat(gate)
        else:features=dict(slots=torch.cat(slots),valid=torch.cat(valid),ids=bank['ids'],labels=bank['labels'].cpu(),references=bank['references'].cpu())
        hook.remove();del model,clip;gc.collect();torch.cuda.empty_cache()
    baseline_ranks=ranks_from_scores(scores['baseline_slots'],features['labels'],features['references']).tolist();old=json.loads((PARENT/'official'/cat/'results.json').read_text())['rows']['slots']['ranks']
    if baseline_ranks!=old:raise ValueError('028 baseline official ranks fail exact023 reproduction')
    patch_manifest=json.loads((root/('patches_'+cat+'.json')).read_text());local=torch.full((n,g),float('nan'));covered=set();example_index=next(i for i,v in enumerate(features['valid']) if v.any());example_max=torch.empty(4,g);example_patch=torch.empty(4,g,dtype=torch.long)
    with torch.no_grad():
        for shard in patch_manifest['shards']:
            guard();path=Path(shard['path'])
            if file_sha(path)!=shard['sha256']:raise ValueError('028 patch shard changed')
            data=torch.load(path,map_location='cpu',weights_only=True);positions=data['positions']
            if data['ids']!=shard['ids'] or positions!=shard['positions'] or data['plan_sha256']!=file_sha(root/'plan.json') or covered&set(positions):raise ValueError('028 patch mapping conflict')
            covered.update(positions);patches=data['patches'].cuda()
            for j in range(0,n,64):
                last=min(j+64,n);s,maximum,winner=local_scores(features['slots'][j:last].cuda(),features['valid'][j:last].cuda(),patches);local[j:last,positions]=s.cpu()
                if j<=example_index<last:example_max[:,positions]=maximum[example_index-j].cpu();example_patch[:,positions]=winner[example_index-j].cpu()
            del patches,data;gc.collect();torch.cuda.empty_cache()
    if covered!=set(range(g)) or not torch.isfinite(local).all():raise ValueError('028 every candidate must be scored exactly once')
    any_valid=features['valid'].any(1);counts=features['valid'].sum(1).clamp_min(1);local[~any_valid]=scores['baseline_slots'][~any_valid];scores['local_summax']=local;scores['global_plus_local']=scores['baseline_slots']+torch.where(any_valid[:,None],local/counts[:,None],torch.zeros_like(local));rows={name:metrics(ranks_from_scores(value,features['labels'],features['references']).tolist()) for name,value in scores.items()}
    saved=CACHE/(cat+'_scores.pt');atomic_torch_save(dict(scores=scores,**features,gate_weights=weights,example_index=example_index,example_max=example_max,example_patch=example_patch),saved)
    examples={}
    for name,value in scores.items():
        order=sorted((i for i in range(g) if i!=int(features['references'][example_index])),key=lambda i:(-float(value[example_index,i]),i));ids=order[:5];gt=int(features['labels'][example_index]);examples[name]=dict(top5=[dict(id=bank['ids'][i],score=float(value[example_index,i]),is_target=i==gt,attribute_max=example_max[:,i].tolist(),best_patch=example_patch[:,i].tolist()) for i in ids],gt_rank=rows[name]['ranks'][example_index])
    valid=features['valid'];gate_stats=[]
    for i,name in enumerate(('color','shape','detail','texture')):
        x=weights[:,i][valid[:,i]];gate_stats.append(dict(slot=name,valid_queries=len(x),mean=float(x.mean()) if len(x) else None,min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None))
    result=dict(status='complete',category=cat,queries=n,gallery=g,rows=rows,transitions=transitions(rows),baseline023_all_ranks_exact=True,all_candidate_patches=True,reference_id_excluded=True,gt_inserted=False,all_official_queries_retained=True,invalid_slot_fallback_queries=int((~any_valid).sum()),local_arm='Frozen023 token checkpoint, inference-only exact valid-slot sum_i max_patch cosine; no direct reference-global or EOS score',hybrid_arm='baseline_slots global cosine + local_summax / valid_slot_count, fixed coefficient1; invalid fallback baseline',gate_stats=gate_stats,example=dict(source_index=context['rows'][example_index]['source_index'],reference=context['rows'][example_index]['candidate'],target=context['rows'][example_index]['target'],instruction=context['rows'][example_index]['modification_text'],target_attrs=context['rows'][example_index]['target_attrs'],valid=valid[example_index].tolist(),gate_weights=weights[example_index].tolist(),rows=examples),score_cache_path=str(saved),score_cache_sha256=file_sha(saved),patch_manifest_sha256=file_sha(root/('patches_'+cat+'.json')),api_calls=0,local_fit_updates=0,seconds=time.monotonic()-start)
    atomic_write_json(output,result);print(json.dumps(dict(status='complete',category=cat,metrics={name:{k:v for k,v in m.items() if k!='ranks'} for name,m in rows.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',choices=('shirt','toptee'),required=True);a=p.parse_args();run(a.root,a.category)
