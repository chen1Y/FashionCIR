"""013: frozen rank4 contribution audit with a query-local vector-term-only control."""
import argparse
import json
from pathlib import Path

import torch

from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from fiq_candidate_head import CandidateHead,candidate_features,head_ranks
from fiq_rank4_head import Rank4Head,pack_vectors
from fiq_rank4_control_math import branch_controls,transition_counts
from fiq_small_math import blend,metrics
from train_fiq_candidate_head import one_cache


def load_frozen(args):
    root=args.research/'rank4_candidate_head_012';folder=root/'rank4_branch'/args.dataset
    summary=json.loads((root/'completion_summary.json').read_text())['categories'][args.dataset]['arms']['rank4_branch']
    for name,digest in [('head.pt',summary['head_sha256']),('results.json',summary['results_sha256'])]:
        if file_sha(folder/name)!=digest:raise ValueError('Sealed012 model/result changed')
    fit=json.loads((folder/'fit_complete.json').read_text())
    if fit['checkpoint_sha256']!=summary['head_sha256']:raise ValueError('012 fit_complete mismatch')
    prior=json.loads((folder/'results.json').read_text());e=prior['evidence']
    if prior['status']!='complete' or not prior['initial011_reproduced']:raise ValueError('012 did not complete verified evaluation')
    for key,name in [('rank4_source_sha256','train_fiq_rank4_head.py'),('rank4_math_sha256','fiq_rank4_head.py')]:
        if file_sha(Path(__file__).with_name(name))!=e[key]:raise ValueError('Sealed012 source changed')
    initial_folder=args.research/'supported_candidate_head_011'/args.dataset
    if file_sha(initial_folder/'head.pt')!=e['initial011_checkpoint_sha256'] or file_sha(initial_folder/'results.json')!=e['initial011_results_sha256']:
        raise ValueError('Initial011 head/result changed')
    old011=json.loads((initial_folder/'results.json').read_text())
    initial=torch.load(initial_folder/'head.pt',map_location='cpu',weights_only=True)
    weights=torch.load(folder/'head.pt',map_location='cpu',weights_only=True)
    if weights['arm']!='rank4_branch' or weights['dimension']!=1024:raise ValueError('Wrong012 model identity')
    scalar=CandidateHead(initial['model_state']['mean'],initial['model_state']['std'])
    model=Rank4Head(scalar,1024,'rank4_branch');model.load_state_dict(weights['model_state']);model.requires_grad_(False);model.eval()
    if any(not torch.equal(model.scalar.state_dict()[k],v) for k,v in initial['model_state'].items()):
        raise ValueError('Frozen scalar no longer equals011')
    for name,digest in e['inherited_gallery_evidence']['metadata_sha256'].items():
        if file_sha(args.root/'image_splits'/name)!=digest:raise ValueError('Dataset metadata changed')
    expected=prior['validation_evidence']['evidence']
    if file_sha(args.primary/'protocol.json')!=expected['primary_protocol_sha256'] or file_sha(Path(__file__).with_name('newModel.py'))!=expected['model_source_sha256']:
        raise ValueError('Encoder/primary protocol changed')
    if file_sha(args.primary/args.dataset/f'structured_{args.dataset}_val.json')!=expected['val_structured_sha256']:
        raise ValueError('Validation structured data changed')
    path,cache=one_cache(args.val_cache,args.dataset,expected)
    if file_sha(path)!=prior['validation_evidence']['cache_sha256']:raise ValueError('Validation cache bytes changed')
    if (len(cache['qd']),len(cache['gallery']))!={'shirt':(2038,6346),'toptee':(1961,5373)}[args.dataset]:
        raise ValueError('Incomplete validation queries/gallery')
    for key in ('qd','qq','gallery','confidence'):
        if not torch.isfinite(cache[key]).all():raise FloatingPointError('Nonfinite frozen cache')
    return model,cache,prior,old011,dict(inherited012_evidence=e,cache_sha256=file_sha(path),source_sha256=file_sha(__file__),
        control_math_sha256=file_sha(Path(__file__).with_name('fiq_rank4_control_math.py')),checkpoint_sha256=summary['head_sha256'])


def evaluate_control(model,cache,prior,old011):
    qd,qq,g=cache['qd'],cache['qq'],cache['gallery'];sources,targets=cache['sources'],cache['targets']
    if ((sources<0)|(targets<0)|(sources>=len(g))|(targets>=len(g))|(sources==targets)).any():raise ValueError('Invalid retrieval bindings')
    fixed=blend(qd,qq,.5);scores=fixed@g.T;scores[torch.arange(len(qd)),sources]=-torch.inf
    order=torch.argsort(scores,descending=True);top=order[:,:20]
    scalar=candidate_features(qd,qq,g[sources],g[top],cache['confidence']);packed=pack_vectors(scalar,fixed,g[top]);dim=model.dimension
    with torch.no_grad():
        logit=model.scalar.network((packed[...,:6]-model.scalar.mean)/model.scalar.std).squeeze(-1)
        u=(model.query_projection(packed[...,6:6+dim])*model.candidate_projection(packed[...,6+dim:])).sum(-1)/2
        corrections=branch_controls(logit,u,cache['valid'])
        if not torch.equal(corrections['paired_rank4'],model(packed,cache['valid'])):raise ValueError('Control decomposition changed original model output')
    baseline=(order==targets[:,None]).nonzero(as_tuple=False)[:,1]+1
    if len(baseline)!=len(qd):raise ValueError('A query lost its target rank')
    ranks={'fixed_dual05':baseline}
    for name,correction in corrections.items():
        if not torch.isfinite(correction).all() or float(correction.abs().max())>.03000001:raise FloatingPointError('Invalid control correction')
        ranks[name]=head_ranks(order,targets,scores.gather(1,top)+correction)
        if not torch.equal(ranks[name]<=20,baseline<=20) or not torch.equal(ranks[name]<=50,baseline<=50):raise ValueError('Control changed tail recall')
    expected={'fixed_dual05':prior['official_validation']['ranks']['fixed_dual05'],
        'zero_rank4':old011['official_validation']['ranks']['candidate_head'],
        'paired_rank4':prior['official_validation']['ranks']['candidate_head'],
        'whole_correction_shuffle':prior['official_validation']['ranks']['shuffled_candidate_control']}
    if any(ranks[name].tolist()!=value for name,value in expected.items()):raise ValueError('Frozen full-query ranks failed exact reproduction')
    transitions={}
    for ref,new in [('zero_rank4','paired_rank4'),('zero_rank4','rank4_only_shuffle'),('rank4_only_shuffle','paired_rank4')]:
        transitions[new+'_minus_'+ref]={f'r{k}':transition_counts(ranks[ref].tolist(),ranks[new].tolist(),k) for k in (1,10,50)}
    valid=cache['valid'].bool();changed=(corrections['paired_rank4']-corrections['zero_rank4'])[valid]
    return dict(metrics={n:metrics(value) for n,value in ranks.items()},ranks={n:value.tolist() for n,value in ranks.items()},transitions=transitions,
        queries=len(qd),gallery=len(g),exact_reproduction={n:True for n in expected},
        amplitudes=dict(valid_queries=int(valid.sum()),frozen_logit_abs_mean=float(logit[valid].abs().mean()),frozen_logit_abs_max=float(logit[valid].abs().max()),
            rank4_u_abs_mean=float(u[valid].abs().mean()),rank4_u_abs_max=float(u[valid].abs().max()),
            correction_delta_abs_mean=float(changed.abs().mean()),correction_delta_abs_max=float(changed.abs().max())))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=('shirt','toptee'),required=True)
    for name in ('root','primary','research','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--val-cache',type=Path,default=Path('/dev/shm/fiq_small_features_20261026'));p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing013; no blind repeat')
    require_space(args.output);torch.set_num_threads(4);model,cache,prior,old011,evidence=load_frozen(args)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',queries=len(cache['qd']),gallery=len(cache['gallery']),all_parameters_frozen=not any(p.requires_grad for p in model.parameters()),
            prior012_fit_complete_verified=True,api_calls=0,fitted_parameters=0)),flush=True);return
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json',dict(evidence=evidence,training=False,api_calls=0,shuffle_seed=42,head_size=20,correction_bound=.03,
        rows=['fixed_dual05','zero_rank4','paired_rank4','whole_correction_shuffle','rank4_only_shuffle'],
        control='Only permute u within each query; frozen011 candidate-specific logit stays aligned, then unified.03*tanh. Invalid structured query0.',
        evaluation='All original validation queries/gallery, ref ID excluded; no GT injection/query filtering.',
        timing='Post-hoc after012 report, no fitting or parameter search; not independent confirmation/significance.',
        limitation='One training seed, prior repeated official-val exploration, frozen backbone saw train; embeddings not VQA/expert GT.'))
    atomic_write_json(args.output/'state.json',dict(status='frozen_branch_control'))
    result=evaluate_control(model,cache,prior,old011);require_space(args.output)
    atomic_write_json(args.output/'results.json',dict(status='complete',category=args.dataset,evidence=evidence,**result,api_calls=0,fitted_parameters=0))
    atomic_write_json(args.output/'state.json',dict(status='complete'))
    print(json.dumps(dict(status='complete',category=args.dataset,metrics=result['metrics'],transitions={n:{k:{key:value for key,value in x.items() if not key.endswith('_indices')} for k,x in t.items()} for n,t in result['transitions'].items()})),flush=True)


if __name__=='__main__':main()
