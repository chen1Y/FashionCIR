"""012: same011 checkpoint, two fixed additional200-update arms, no API or encoder training."""
import argparse
import copy
import json
import math
from pathlib import Path

import torch

from experiment_storage import require_space,atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from fiq_candidate_head import CandidateHead,candidate_features,head_ranks,shuffled_correction
from fiq_rank4_head import Rank4Head,pack_vectors
from fiq_small_math import blend,metrics
from fiq_supported_head import masked_loss
from train_fiq_supported_head import prepare_supported
from train_fiq_candidate_head import load_val_after_fit


def prepare_rank4(args):
    c=prepare_supported(args)
    folder=args.research/'supported_candidate_head_011'/args.dataset
    summary=json.loads((args.research/'supported_candidate_head_011/completion_summary.json').read_text())['categories'][args.dataset]
    for name,digest in [('head.pt',summary['head_sha256']),('results.json',summary['results_sha256']),('manifest.json',summary['manifest_sha256'])]:
        if file_sha(folder/name)!=digest:raise ValueError('Sealed011 artifact changed')
    old=json.loads((folder/'results.json').read_text())
    if old['status']!='complete' or old['counts']!=c['counts'] or old['isolation_proof']!=c['proof']:
        raise ValueError('011 supervision/proof does not reproduce')
    artifacts=json.loads((folder.parent/'artifact_sha256.json').read_text())
    if file_sha(folder/'candidate_pools.pt')!=artifacts[str((folder/'candidate_pools.pt').relative_to(args.research))]:
        raise ValueError('Original011 pools changed')
    saved=torch.load(folder/'candidate_pools.pt',map_location='cpu',weights_only=True)
    checkpoint=torch.load(folder/'head.pt',map_location='cpu',weights_only=True)
    initial=CandidateHead(checkpoint['model_state']['mean'],checkpoint['model_state']['std'])
    initial.load_state_dict(checkpoint['model_state']);initial.eval()
    dimension=c['gallery'].shape[1]
    norm_error=float((c['gallery'].norm(dim=-1)-1).abs().max())
    if norm_error>1e-3:raise ValueError('Existing candidate vectors are not normalized')
    c['evidence'].update(initial011_checkpoint_sha256=summary['head_sha256'],initial011_results_sha256=summary['results_sha256'],
        rank4_source_sha256=file_sha(__file__),rank4_math_sha256=file_sha(Path(__file__).with_name('fiq_rank4_head.py')))
    for name,pool in c['pools'].items():
        if any(not torch.equal(pool[k],saved[name][k]) for k in ('indices','mask','original_rows')):
            raise ValueError('012 changed011 real support/padding/row order')
        active=pool['original_rows'];mask=pool['mask']
        query=blend(c['query']['qd'][active],c['query']['qq'][active],.5)
        candidates=torch.zeros(len(active),20,dimension,dtype=c['gallery'].dtype)
        candidates[mask]=c['gallery'][pool['indices'][mask]]
        pool['x']=pack_vectors(pool['x'],query,candidates,mask)
    torch.manual_seed(42)
    head=Rank4Head(copy.deepcopy(initial),dimension,args.arm)
    expected_parameters=129 if args.arm=='scalar_continue' else dimension*8
    if sum(p.numel() for p in head.parameters() if p.requires_grad)!=expected_parameters:
        raise ValueError('Trainable parameter count changed')
    errors={}
    for name,pool in c['pools'].items():
        with torch.no_grad():
            original=initial(pool['x'][...,:6],pool['valid']);fresh=head(pool['x'],pool['valid'])
        errors[name]=float((original-fresh).abs().max())
    if any(errors.values()):raise ValueError('012 initial output must exactly reproduce011 on train/dev')
    with torch.no_grad():loss=float(masked_loss(head,c['pools']['dev']))
    if abs(loss-old['training']['selected_internal_dev_loss'])>1e-6:
        raise ValueError('Initial011 supported dev loss does not reproduce')
    c.update(dimension=dimension,head=head,initial=initial,initial_errors=errors,initial_dev_loss=loss,
        normalization_error=norm_error,initial011_training=checkpoint['summary'])
    return c


def fit_rank4(args,c):
    head=c['head'];train,dev=c['pools']['fit'],c['pools']['dev']
    parameters=[p for p in head.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=5e-4,weight_decay=.01)
    initial_vector=torch.cat([p.detach().flatten().clone() for p in parameters])
    initial_named={n:p.detach().clone() for n,p in head.named_parameters() if p.requires_grad}
    initial_scalar=copy.deepcopy(head.scalar.state_dict())
    rng=torch.Generator().manual_seed(42);exposure=torch.zeros(len(train['x']),dtype=torch.long)
    best,best_step,best_state=c['initial_dev_loss'],0,copy.deepcopy(head.state_dict())
    gradient_evidence={};projection_first_nonzero={}
    with (args.output/'training.jsonl').open('x') as stream:
        stream.write(json.dumps(dict(step=0,internal_dev_loss=best))+'\n')
        for step in range(1,201):
            indices=torch.randperm(len(train['x']),generator=rng)[:64];exposure[indices]+=1
            optimizer.zero_grad(set_to_none=True);loss=masked_loss(head,train,indices)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite012 loss')
            loss.backward()
            if args.arm=='rank4_branch':
                for name,p in head.named_parameters():
                    if p.requires_grad and p.grad is not None and float(p.grad.norm())>0:
                        projection_first_nonzero.setdefault(name,step)
            if step==1:
                gradient_evidence={n:float(p.grad.norm()) for n,p in head.named_parameters() if p.requires_grad and p.grad is not None}
                if args.arm=='rank4_branch' and not gradient_evidence.get('candidate_projection.weight',0)>0:
                    raise FloatingPointError('Zero-initialized candidate projection received no learnable gradient')
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite012 gradient')
            optimizer.step()
            if step%50==0:
                with torch.no_grad():value=float(masked_loss(head,dev))
                delta=float((torch.cat([p.detach().flatten() for p in parameters])-initial_vector).norm())
                if not math.isfinite(value) or not math.isfinite(delta):raise FloatingPointError('Nonfinite012 parameters/dev loss')
                if value<best-1e-6:best,best_step,best_state=value,step,copy.deepcopy(head.state_dict())
                row=dict(step=step,train_loss=float(loss.detach()),internal_dev_loss=value,gradient_norm=float(norm),parameter_delta_l2=delta,
                    trainable_parameter_deltas={n:float((p.detach()-initial_named[n]).norm()) for n,p in head.named_parameters() if p.requires_grad})
                stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
    head.load_state_dict(best_state)
    scalar_unchanged=all(torch.equal(v,head.scalar.state_dict()[k]) for k,v in initial_scalar.items())
    if args.arm=='rank4_branch' and not scalar_unchanged:raise ValueError('Frozen011 scalar branch changed')
    if not torch.equal(head.scalar.mean,c['initial'].mean) or not torch.equal(head.scalar.std,c['initial'].std):
        raise ValueError('Frozen011 normalization changed')
    summary=dict(steps=200,selected_step=best_step,initial_internal_dev_loss=c['initial_dev_loss'],selected_internal_dev_loss=best,
        final_parameter_delta_l2=delta,selected_parameter_delta_l2=float((torch.cat([p.detach().flatten() for p in parameters])-initial_vector).norm()),
        trainable_parameters=sum(p.numel() for p in parameters),total_parameters=sum(p.numel() for p in head.parameters()),
        first_gradient_norms=gradient_evidence,scalar_branch_unchanged=scalar_unchanged,normalization_unchanged=True,
        projection_first_nonzero_gradient_steps=projection_first_nonzero,
        selected_trainable_parameter_deltas={n:float((p.detach()-initial_named[n]).norm()) for n,p in head.named_parameters() if p.requires_grad},
        optimizer_state='New AdamW, not exact optimizer resume',official_validation_used_for_fit=False,
        sample_presentations=int(exposure.sum()),fit_original_rows=train['original_rows'].tolist(),per_eligible_fit_row_exposures=exposure.tolist())
    if summary['sample_presentations']!=12800:raise ValueError('012 exposure budget changed')
    atomic_torch_save(dict(model_state=head.state_dict(),summary=summary,evidence=c['evidence'],dimension=c['dimension'],arm=args.arm),args.output/'head.pt')
    summary['checkpoint_sha256']=file_sha(args.output/'head.pt');atomic_write_json(args.output/'fit_complete.json',summary)
    return head,summary


def evaluate_rank4(head,initial,cache,stable):
    qd,qq,gallery=cache['qd'],cache['qq'],cache['gallery'];sources,targets=cache['sources'],cache['targets']
    fixed=blend(qd,qq,.5);scores=fixed@gallery.T;scores[torch.arange(len(qd)),sources]=-torch.inf
    order=torch.argsort(scores,descending=True,stable=stable);indices=order[:,:20]
    scalar=candidate_features(qd,qq,gallery[sources],gallery[indices],cache['confidence'])
    packed=pack_vectors(scalar,fixed,gallery[indices])
    with torch.no_grad():correction=head(packed,cache['valid']);original=initial(scalar,cache['valid'])
    if not torch.isfinite(correction).all() or float(correction.abs().max())>.03000001:
        raise FloatingPointError('012 nonfinite/unbounded correction')
    baseline=(order==targets[:,None]).nonzero(as_tuple=False)[:,1]+1
    if len(baseline)!=len(qd):raise ValueError('Missing annotated rank')
    base=scores.gather(1,indices)
    ranks=dict(fixed_dual05=baseline,initial011=head_ranks(order,targets,base+original),candidate_head=head_ranks(order,targets,base+correction),
        shuffled_candidate_control=head_ranks(order,targets,base+shuffled_correction(correction)))
    dqu=qd@gallery.T;dqu[torch.arange(len(qd)),sources]=-torch.inf
    ranks['v7_off_dqu_written']=(torch.argsort(dqu,descending=True,stable=stable)==targets[:,None]).nonzero(as_tuple=False)[:,1]+1
    for n in ('initial011','candidate_head','shuffled_candidate_control'):
        if not torch.equal(ranks[n]<=20,baseline<=20) or not torch.equal(ranks[n]<=50,baseline<=50):raise ValueError('012 changed tail recall')
    return dict(metrics={n:metrics(r) for n,r in ranks.items()},ranks={n:r.tolist() for n,r in ranks.items()},queries=len(qd),gallery=len(gallery),
        correction_abs_max=float(correction.abs().max()),correction_mean=float(correction.mean()),delta_from011_mean=float((correction-original).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('shirt','toptee'),required=True);p.add_argument('--arm',choices=('scalar_continue','rank4_branch'),required=True)
    for name in ('root','primary','research','output'):p.add_argument('--'+name,type=Path,required=True)
    for name,folder in [('query-cache','fiq_gate_features_20261026'),('gallery-cache','fiq_train_gallery_20261026'),('val-cache','fiq_small_features_20261026')]:
        p.add_argument('--'+name,type=Path,default=Path('/dev/shm')/folder)
    p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve012 output; no blind repeat')
    require_space(args.output);torch.set_num_threads(4);torch.manual_seed(42);c=prepare_rank4(args)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',arm=args.arm,dimension=c['dimension'],counts=c['counts'],proof=c['proof'],
            initial_output_errors=c['initial_errors'],initial_dev_loss=c['initial_dev_loss'],candidate_norm_error=c['normalization_error'],
            trainable_parameters=sum(p.numel() for p in c['head'].parameters() if p.requires_grad),official_val_features_loaded=False,api_calls=0)),flush=True);return
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json',dict(evidence=c['evidence'],arm=args.arm,rank=4,dimension=c['dimension'],seed=42,steps=200,lr=5e-4,
        weight_decay=.01,batch=64,temperature=.07,anchor=.01,gradient_clip=1.,delta_bound=.03,
        selection='Same011 supported dev NCE at0/50/100/150/200, includes starting checkpoint',
        architecture='A:129 scalar parameters continue. B: frozen011 scalar logit+sum(Wq(q)*Wc(c))/sqrt4 within one .03*tanh; Wc zero, Wq standard PyTorch Linear initialization with seed42.',
        statistics='Reuse frozen011 scalar mean/std; original normalized frozen vectors, no refitting normalization.',
        initialization='Same selected011 head; new AdamW state for both, not exact optimizer resume',
        hypothesis_timing='Designed after previous official-val reports; exploratory, not independent confirmation or val hyperparameter scan.',
        limitation='Backbone saw train; byte-clean support is subset of inferencehead; single seed and repeated official-val exploration.',api_calls=0))
    atomic_write_json(args.output/'preparation.json',dict(counts=c['counts'],proof=c['proof'],initial_output_errors=c['initial_errors'],initial_dev_loss=c['initial_dev_loss']))
    atomic_write_json(args.output/'state.json',dict(status='fitting012'))
    head,training=fit_rank4(args,c);atomic_write_json(args.output/'state.json',dict(status='evaluating_after_fit'))
    internal=dict(qd=c['query']['qd'][1024:],qq=c['query']['qq'][1024:],gallery=c['gallery'],sources=c['sources'][1024:],targets=c['targets'][1024:],
        confidence=c['query']['confidence'][1024:],valid=c['query']['valid'][1024:])
    internal_result=evaluate_rank4(head,c['initial'],internal,True)
    cache,proof,val_evidence=load_val_after_fit(args,c['evidence']);result=evaluate_rank4(head,c['initial'],cache,False)
    old011=json.loads((args.research/'supported_candidate_head_011'/args.dataset/'results.json').read_text())
    for row,old in [('fixed_dual05','fixed_dual_alpha05'),('v7_off_dqu_written','v7_off_dqu_written')]:
        if any(abs(result['metrics'][row][k]-proof['results'][old][k])>.0001 for k in ('r1','r10','r50')):raise ValueError('012 baseline not reproduced')
    if any(abs(result['metrics']['initial011'][k]-old011['official_validation']['metrics']['candidate_head'][k])>.0001 for k in ('r1','r10','r50')):
        raise ValueError('Initial011 official metrics failed reproduction')
    require_space(args.output)
    atomic_write_json(args.output/'results.json',dict(status='complete',category=args.dataset,arm=args.arm,training=training,counts=c['counts'],proof=c['proof'],
        evidence=c['evidence'],validation_evidence=val_evidence,initial_output_errors=c['initial_errors'],internal_dev_retrieval_all_256=internal_result,
        official_validation=result,fixed_baselines_reproduced=True,initial011_reproduced=True,
        deltas_pp_vs_fixed05={n:{k:m[k]-result['metrics']['fixed_dual05'][k] for k in ('r1','r10','r50')} for n,m in result['metrics'].items()},
        head_delta_pp_vs011={k:result['metrics']['candidate_head'][k]-result['metrics']['initial011'][k] for k in ('r1','r10','r50')},api_calls=0))
    atomic_write_json(args.output/'state.json',dict(status='complete'));print(json.dumps(dict(status='complete',arm=args.arm,category=args.dataset,official_metrics=result['metrics'])),flush=True)


if __name__=='__main__':main()
