"""011: train the unchanged bounded head only on byte-eligible frozen top20 support."""
import argparse
import copy
import gzip
import json
import math
from pathlib import Path

import torch

from experiment_storage import require_space, atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from fiq_candidate_head import CandidateHead, candidate_features, fit_statistics
from fiq_byte_clean_pools import positive_byte_sets
from fiq_small_math import blend
from fiq_supported_head import select_supported_rows, masked_loss
from run_fiq_candidate_alignment import load_inputs
from train_fiq_candidate_head import evaluate, load_val_after_fit


def prepare_supported(args):
    q, g, inherited_manifest, original_rows, old_pools, hashes, evidence = load_inputs(args)
    summary010 = json.loads((args.research/'candidate_alignment_010/completion_summary.json').read_text())['categories'][args.dataset]
    result_path = args.research/'candidate_alignment_010'/args.dataset/'results.json'
    if file_sha(result_path) != summary010['results_sha256']:
        raise ValueError('Sealed010 result changed')
    diagnostic = json.loads(result_path.read_text())
    if diagnostic['status'] != 'complete' or diagnostic['evidence']['inherited_009_evidence'] != evidence:
        raise ValueError('010 cache evidence mismatch')
    index = {image_id:j for j,image_id in enumerate(g['ids'])}
    modifications = [r['modification_text'] for r in original_rows['fit']+original_rows['dev']]
    held = json.loads(gzip.decompress((args.research/'train_only_candidate_head/official_heldout_byte_index.json.gz').read_bytes()))
    held_sha = set(held['image_sha256'].values())
    pools, selected, reasons, input_sha = {}, {}, {}, {}
    buckets = {}
    for name in ('fit', 'dev'):
        allowed = set(inherited_manifest[name+'_gallery_ids'])
        known = positive_byte_sets(original_rows[name], hashes)
        details = diagnostic['splits'][name]['queries']
        active, ids, why = select_supported_rows(details, hashes, allowed, known, q['source_ids'], modifications)
        expected = {'shirt':{'fit':641,'dev':150}, 'toptee':{'fit':771,'dev':190}}[args.dataset][name]
        if len(active) != expected or len(active) < {'fit':128, 'dev':32}[name]:
            raise ValueError('Supported supervision count changed; inspect without relaxing')
        if [r['original_query_row'] for r in details] != old_pools[name]['original_rows'].tolist():
            raise ValueError('010 inherited row order changed')
        active_tensor = torch.tensor(active, dtype=torch.long)
        indices = torch.full((len(active),20), -1, dtype=torch.long)
        mask = torch.zeros_like(indices, dtype=torch.bool)
        for row, images in enumerate(ids):
            indices[row,:len(images)] = torch.tensor([index[i] for i in images])
            mask[row,:len(images)] = True
        candidates = torch.zeros(len(active),20,g['gallery'].shape[1], dtype=g['gallery'].dtype)
        candidates[mask] = g['gallery'][indices[mask]]
        sources = torch.tensor([index[q['source_ids'][i]] for i in active])
        x = candidate_features(q['qd'][active_tensor],q['qq'][active_tensor],g['gallery'][sources],candidates,q['confidence'][active_tensor])
        x = torch.where(mask[...,None], x, torch.zeros_like(x))
        base = torch.einsum('bd,bkd->bk', blend(q['qd'][active_tensor],q['qq'][active_tensor],.5), candidates)
        lookup = {r['original_query_row']:r for r in details}
        for j, original in enumerate(active):
            score_by_id = dict(zip(lookup[original]['top20_ids'],lookup[original]['top20_scores']))
            expected_base = torch.tensor([score_by_id[i] for i in ids[j]])
            if float((base[j,mask[j]]-expected_base).abs().max()) > 1e-6:
                raise ValueError('Frozen head candidate scores do not reproduce')
        input_sha[name] = {hashes[i] for pool in ids for i in pool} | {hashes[q['source_ids'][i]] for i in active}
        if input_sha[name] & held_sha:
            raise ValueError('Actual supported input overlaps official heldout bytes')
        pools[name] = dict(indices=indices,mask=mask,x=x,base=base,valid=q['valid'][active_tensor],original_rows=active_tensor)
        if not pools[name]['valid'].all():
            raise ValueError('Invalid structured row entered support fit')
        selected[name], reasons[name] = active, why
        buckets[name] = {b:sum(lookup[i]['bucket']==b for i in active) for b in ('top1','rank2_20','outside20')}
    if input_sha['fit'] & input_sha['dev']:
        raise ValueError('Actual fit/dev byte overlap')
    proof = dict(fit_dev_byte_overlap=0,fit_heldout_byte_overlap=0,dev_heldout_byte_overlap=0,
        actual_input_unique_sha={n:len(s) for n,s in input_sha.items()},all_supervised_candidates_in_original_top20=True,
        gt_injection=False,padding_in_statistics_or_loss=False)
    evidence = dict(evidence, support_source_sha256=file_sha(__file__),
        support_math_sha256=file_sha(Path(__file__).with_name('fiq_supported_head.py')),diagnostic010_sha256=file_sha(result_path))
    return dict(query=q,gallery=g['gallery'],ids=g['ids'],sources=torch.tensor([index[i] for i in q['source_ids']]),
        targets=torch.tensor([index[i] for i in q['target_ids']]),pools=pools,evidence=evidence,proof=proof,
        manifest=dict(original_fit=original_rows['fit'],original_dev=original_rows['dev'],inherited009_masks=inherited_manifest['supervision_masks'],
            selected_original_rows=selected,additional_reasons=reasons,selected_rank_buckets=buckets,
            candidate_ids={n:[[g['ids'][i] for i in row[m].tolist()] for row,m in zip(p['indices'],p['mask'])] for n,p in pools.items()}),
        counts=dict(fit_total=1024,dev_total=256,fit_byte_eligible=len(old_pools['fit']['original_rows']),
            dev_byte_eligible=len(old_pools['dev']['original_rows']),fit_supported=len(selected['fit']),dev_supported=len(selected['dev']),
            selected_rank_buckets=buckets,real_candidates={n:int(p['mask'].sum()) for n,p in pools.items()}))


def fit_supported(args, context):
    train, dev = context['pools']['fit'],context['pools']['dev']
    mean,std = fit_statistics(train['x'][train['mask']])
    head = CandidateHead(mean,std)
    optimizer = torch.optim.AdamW(head.parameters(),lr=5e-4,weight_decay=.01)
    rng = torch.Generator().manual_seed(42)
    initial = torch.cat([p.detach().flatten().clone() for p in head.parameters()])
    exposure = torch.zeros(len(train['x']),dtype=torch.long)
    with torch.no_grad():initial_dev=float(masked_loss(head,dev))
    if not math.isfinite(initial_dev):raise FloatingPointError('Nonfinite initial support dev loss')
    best,best_step,best_state=initial_dev,0,copy.deepcopy(head.state_dict())
    with (args.output/'training.jsonl').open('x') as stream:
        stream.write(json.dumps(dict(step=0,internal_dev_loss=initial_dev))+'\n')
        for step in range(1,201):
            indices=torch.randperm(len(train['x']),generator=rng)[:64];exposure[indices]+=1
            optimizer.zero_grad(set_to_none=True);value=masked_loss(head,train,indices)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite support training loss')
            value.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite support gradient')
            optimizer.step()
            if step%50==0:
                with torch.no_grad():score=float(masked_loss(head,dev))
                delta=float((torch.cat([p.detach().flatten() for p in head.parameters()])-initial).norm())
                if not math.isfinite(score) or not math.isfinite(delta):raise FloatingPointError('Nonfinite support parameters/dev loss')
                if score<best-1e-6:best,best_step,best_state=score,step,copy.deepcopy(head.state_dict())
                row=dict(step=step,train_loss=float(value.detach()),internal_dev_loss=score,gradient_norm=float(norm),parameter_delta_l2=delta)
                stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
    head.load_state_dict(best_state)
    summary=dict(steps=200,selected_step=best_step,initial_internal_dev_loss=initial_dev,selected_internal_dev_loss=best,
        final_parameter_delta_l2=delta,selected_parameter_delta_l2=float((torch.cat([p.detach().flatten() for p in head.parameters()])-initial).norm()),
        trainable_parameters=sum(p.numel() for p in head.parameters()),official_validation_used_for_fit=False,
        sample_presentations=int(exposure.sum()),fit_original_rows=train['original_rows'].tolist(),per_eligible_fit_row_exposures=exposure.tolist(),
        normalizer_candidate_count=int(train['mask'].sum()),padding_normalizer_count=0)
    if summary['sample_presentations']!=12800:raise ValueError('Update budget changed')
    atomic_torch_save(dict(model_state=head.state_dict(),summary=summary,evidence=context['evidence']),args.output/'head.pt')
    summary['checkpoint_sha256']=file_sha(args.output/'head.pt');atomic_write_json(args.output/'fit_complete.json',summary)
    return head,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('shirt','toptee'),required=True)
    for name in ('root','primary','research','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name,folder in [('query-cache','fiq_gate_features_20261026'),('gallery-cache','fiq_train_gallery_20261026'),('val-cache','fiq_small_features_20261026')]:
        parser.add_argument('--'+name,type=Path,default=Path('/dev/shm')/folder)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing011; no blind restart')
    require_space(args.output);torch.set_num_threads(4);torch.manual_seed(42);c=prepare_supported(args)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',counts=c['counts'],proof=c['proof'],official_val_features_loaded=False,api_calls=0)),flush=True);return
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json',dict(evidence=c['evidence'],seed=42,steps=200,lr=5e-4,weight_decay=.01,batch=64,
        hidden=16,temperature=.07,anchor=.01,delta_bound=.03,gradient_clip=1.,statistics='Eligible fit real candidates only, no padding, std>=1e-3',
        selection='Supported internal-dev NCE at0/50/100/150/200',supervision='GT already in frozen top20, valid structured query, >=1 clean unique negative from same head. Include original top1.',
        variation_from009='Candidate support/eligible rows and variable candidate count only; unchanged model, optimizer, update count and inference.',
        official_eval='Only after fit_complete, complete original ID queries/gallery, ref excluded; top20 correction and per-query shuffle.',
        limitation='Supervision only uses byte-eligible intersection, not full inference distribution. Backbone saw train; recompressed duplicates and semantic false negatives remain.',api_calls=0))
    atomic_write_json(args.output/'manifest.json',c['manifest']);atomic_write_json(args.output/'isolation_proof.json',c['proof'])
    atomic_torch_save({n:{k:p[k] for k in ('indices','mask','original_rows')} for n,p in c['pools'].items()},args.output/'candidate_pools.pt')
    atomic_write_json(args.output/'state.json',dict(status='fitting_supported_head'))
    head,training=fit_supported(args,c);atomic_write_json(args.output/'state.json',dict(status='evaluating_after_fit'))
    internal=dict(qd=c['query']['qd'][1024:],qq=c['query']['qq'][1024:],gallery=c['gallery'],sources=c['sources'][1024:],targets=c['targets'][1024:],
                  confidence=c['query']['confidence'][1024:],valid=c['query']['valid'][1024:])
    internal_result=evaluate(head,internal,True);cache,proof,val_evidence=load_val_after_fit(args,c['evidence']);result=evaluate(head,cache,False)
    for row,old in [('fixed_dual05','fixed_dual_alpha05'),('v7_off_dqu_written','v7_off_dqu_written')]:
        if any(abs(result['metrics'][row][k]-proof['results'][old][k])>.0001 for k in ('r1','r10','r50')):raise ValueError('Frozen baseline did not reproduce')
    old009=json.loads((args.research/'train_only_candidate_head_byte_clean'/args.dataset/'results.json').read_text())
    require_space(args.output)
    atomic_write_json(args.output/'results.json',dict(status='complete',category=args.dataset,training=training,counts=c['counts'],isolation_proof=c['proof'],
        evidence=c['evidence'],validation_evidence=val_evidence,internal_dev_retrieval_all_256=internal_result,official_validation=result,fixed_baselines_reproduced=True,
        deltas_pp_vs_fixed05={n:{k:m[k]-result['metrics']['fixed_dual05'][k] for k in ('r1','r10','r50')} for n,m in result['metrics'].items()},
        head_delta_pp_vs009={k:result['metrics']['candidate_head'][k]-old009['official_validation']['metrics']['candidate_head'][k] for k in ('r1','r10','r50')},api_calls=0,
        scope='Single-seed supported training objective trial; complete ID evaluation, all rows/shuffle retained, no significance or independent-backbone claim.'))
    atomic_write_json(args.output/'state.json',dict(status='complete'));print(json.dumps(dict(status='complete',category=args.dataset,official_metrics=result['metrics'])),flush=True)


if __name__=='__main__':main()
