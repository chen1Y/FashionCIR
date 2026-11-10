"""009: same candidate head/optimizer, byte-clean head supervision, untouched official ID eval."""
import argparse
import copy
import gzip
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment_storage import require_space, atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from fiq_candidate_head import CandidateHead, candidate_features, fit_statistics
from fiq_byte_clean_pools import clean_partition, clean_training_pool, verify_actual_inputs
from train_fiq_candidate_head import prepare, evaluate, load_val_after_fit


def prepare_clean(args):
    # Reuse sealed cache/weight/data checks, never the old supervision pools or fit statistics.
    context = prepare(args)
    context.pop('pools')
    index_path = args.research/'train_only_candidate_head/official_heldout_byte_index.json.gz'
    byte_index = json.loads(gzip.decompress(index_path.read_bytes()))
    if len(byte_index['image_sha256']) != 30566 or len(byte_index['metadata_sha256']) != 6:
        raise ValueError('Unexpected official heldout byte index')
    for name, digest in byte_index['metadata_sha256'].items():
        if file_sha(args.root/'image_splits'/name) != digest:
            raise ValueError('Official split metadata changed; inspect byte index version')
    audit = json.loads((args.research/'train_only_candidate_head/byte_partition_audit_v2.json').read_text())
    # Bind the reusable byte index to its previous independently verified audit.
    expected_sha = audit['heldout_byte_index_sha256']
    if expected_sha is None:
        raise ValueError('Missing audited heldout index digest')
    if file_sha(index_path) != expected_sha:
        raise ValueError('Heldout byte index changed')
    gm = json.loads((args.research/'train_gallery_diagnostic'/args.dataset/'manifest.json').read_text())
    hashes = gm['image_sha256']
    rows = dict(fit=context['manifest']['fit'], dev=context['manifest']['internal_dev'])
    heldout_sha = set(byte_index['image_sha256'].values())
    policy = clean_partition(context['ids'], hashes, rows['fit'], rows['dev'], heldout_sha)
    if len(policy['active']['fit']) < 128 or len(policy['active']['dev']) < 32:
        raise ValueError('Insufficient byte-clean supervision; do not relax isolation')
    pools = {}
    for name, start, end in (('fit', 0, 1024), ('dev', 1024, 1280)):
        active = policy['active'][name]
        selected = torch.tensor(active, dtype=torch.long) + start
        indices, base = clean_training_pool(context['query']['qd'][start:end], context['query']['qq'][start:end],
            context['gallery'], context['sources'][start:end], context['targets'][start:end], context['ids'], hashes,
            rows[name], active, policy['banks'][name], policy['canonical'][name])
        x = candidate_features(context['query']['qd'][selected], context['query']['qq'][selected],
            context['gallery'][context['sources'][selected]], context['gallery'][indices], context['query']['confidence'][selected])
        pools[name] = dict(indices=indices, base=base, x=x, valid=context['query']['valid'][selected], original_rows=selected)
    proof = verify_actual_inputs(context['ids'], hashes, rows, policy['active'],
        {n: p['indices'].tolist() for n, p in pools.items()}, policy['banks'], heldout_sha)
    context['pools'] = pools
    context['counts'] = dict(fit_total=1024, dev_total=256, fit_eligible=len(policy['active']['fit']),
        dev_eligible=len(policy['active']['dev']), fit_gallery=len(policy['banks']['fit']),
        dev_gallery=len(policy['banks']['dev']), fit_negative_sha=len(policy['canonical']['fit']),
        dev_negative_sha=len(policy['canonical']['dev']), reason_counts=policy['reason_counts'])
    context['manifest'].update(supervision_masks=policy['masks'], supervision_reasons=policy['reasons'],
        active_original_rows=policy['active'], fit_gallery_ids=[context['ids'][j] for j in policy['banks']['fit']],
        dev_gallery_ids=[context['ids'][j] for j in policy['banks']['dev']],
        canonical_negative_ids={n: [context['ids'][j] for j in pool] for n, pool in policy['canonical'].items()},
        candidate_pool_policy='Actual GT ID+31 unique JPEG SHA negatives; reference and all exact-known-query positives by byte excluded. No eval GT insertion.')
    context['evidence'].update(byte_index_sha256=file_sha(index_path), byte_clean_source_sha256=file_sha(__file__),
        byte_pool_source_sha256=file_sha(Path(__file__).with_name('fiq_byte_clean_pools.py')))
    context['isolation_proof'] = proof
    return context


def fit_clean(args, context):
    train, dev = context['pools']['fit'], context['pools']['dev']
    mean, std = fit_statistics(train['x'])
    head = CandidateHead(mean, std)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=.01)
    rng = torch.Generator().manual_seed(42)
    initial = torch.cat([p.detach().flatten().clone() for p in head.parameters()])
    exposure = torch.zeros(len(train['x']), dtype=torch.long)

    def loss(pool, indices=None):
        x, base, valid = (pool[k] for k in ('x', 'base', 'valid'))
        if indices is not None:
            x, base, valid = x[indices], base[indices], valid[indices]
        correction = head(x, valid)
        return F.cross_entropy((base+correction)/.07, torch.zeros(len(x), dtype=torch.long)) + .01*correction.square().mean()

    with torch.no_grad():
        initial_dev = float(loss(dev))
    if not math.isfinite(initial_dev):
        raise FloatingPointError('Nonfinite initial internal dev loss')
    best, best_step, best_state = initial_dev, 0, copy.deepcopy(head.state_dict())
    with (args.output/'training.jsonl').open('x') as stream:
        stream.write(json.dumps(dict(step=0, internal_dev_loss=initial_dev))+'\n')
        for step in range(1, 201):
            indices = torch.randperm(len(train['x']), generator=rng)[:64]
            exposure[indices] += 1
            optimizer.zero_grad(set_to_none=True)
            value = loss(train, indices)
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite training loss')
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if step % 50 == 0:
                with torch.no_grad():
                    score = float(loss(dev))
                delta = float((torch.cat([p.detach().flatten() for p in head.parameters()])-initial).norm())
                if not math.isfinite(score) or not math.isfinite(delta):
                    raise FloatingPointError('Nonfinite dev loss/parameters')
                if score < best-1e-6:
                    best, best_step, best_state = score, step, copy.deepcopy(head.state_dict())
                row = dict(step=step, train_loss=float(value.detach()), internal_dev_loss=score,
                           gradient_norm=float(norm), parameter_delta_l2=delta)
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                print(json.dumps(row), flush=True)
    head.load_state_dict(best_state)
    selected_delta = float((torch.cat([p.detach().flatten() for p in head.parameters()])-initial).norm())
    summary = dict(steps=200, selected_step=best_step, initial_internal_dev_loss=initial_dev,
        selected_internal_dev_loss=best, final_parameter_delta_l2=delta, selected_parameter_delta_l2=selected_delta,
        trainable_parameters=sum(p.numel() for p in head.parameters()), official_validation_used_for_fit=False,
        fit_eligible=len(train['x']), dev_eligible=len(dev['x']), sample_presentations=int(exposure.sum()),
        per_eligible_fit_row_exposures=exposure.tolist(), fit_original_rows=train['original_rows'].tolist(),
        normalization_original_rows=train['original_rows'].tolist())
    if summary['sample_presentations'] != 12800:
        raise ValueError('Training update/exposure budget changed')
    atomic_torch_save(dict(model_state=head.state_dict(), summary=summary, evidence=context['evidence']), args.output/'head.pt')
    summary['checkpoint_sha256'] = file_sha(args.output/'head.pt')
    atomic_write_json(args.output/'fit_complete.json', summary)
    return head, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('shirt', 'toptee'), required=True)
    for name in ('root', 'primary', 'research', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name, default in (('query-cache', 'fiq_gate_features_20261026'), ('gallery-cache', 'fiq_train_gallery_20261026'),
                          ('val-cache', 'fiq_small_features_20261026')):
        parser.add_argument('--'+name, type=Path, default=Path('/dev/shm')/default)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Inspect existing009 output; no blind repeat')
    require_space(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    context = prepare_clean(args)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed', counts=context['counts'], proof=context['isolation_proof'],
                              official_val_features_loaded=False, api_calls=0)), flush=True)
        return
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json', dict(evidence=context['evidence'], seed=42, steps=200, lr=5e-4,
        weight_decay=.01, batch=64, hidden=16, temperature=.07, anchor=.01, delta_bound=.03,
        statistics='Eligible fit pools only; std>=1e-3', selection='Eligible dev NCE at0/50/100/150/200',
        isolation='Dev-priority raw JPEG SHA. Exclude all official heldout bytes from fit/dev supervision; entire ID-dev bank bytes forbidden to fit.',
        negatives='Actual GT ID+31 SHA-unique frozen hard negatives, ref/known alternative positive byte aliases excluded.',
        official_eval='Only after fit_complete: every original validation query/gallery, reference ID excluded, top20 correction, per-query shuffle.',
        unchanged_from_008='Architecture, seed, optimizer, update count, loss, selection metric and official ID evaluation.',
        limitation='Only new-head supervision cleaned. Frozen backbone saw official train; exact JPEG SHA misses recompressed/near duplicate images. Other semantic false negatives remain.', api_calls=0))
    atomic_write_json(args.output/'manifest.json', context['manifest'])
    atomic_write_json(args.output/'isolation_proof.json', context['isolation_proof'])
    atomic_torch_save({n: dict(indices=p['indices'], original_rows=p['original_rows']) for n, p in context['pools'].items()}, args.output/'candidate_pools.pt')
    atomic_write_json(args.output/'state.json', dict(status='fitting_byte_clean_head'))
    head, training = fit_clean(args, context)
    atomic_write_json(args.output/'state.json', dict(status='evaluating_after_fit'))
    internal = dict(qd=context['query']['qd'][1024:], qq=context['query']['qq'][1024:], gallery=context['gallery'],
        sources=context['sources'][1024:], targets=context['targets'][1024:], confidence=context['query']['confidence'][1024:], valid=context['query']['valid'][1024:])
    internal_result = evaluate(head, internal, True)
    cache, proof, val_evidence = load_val_after_fit(args, context['evidence'])
    result = evaluate(head, cache, False)
    for row, old in (('fixed_dual05', 'fixed_dual_alpha05'), ('v7_off_dqu_written', 'v7_off_dqu_written')):
        if any(abs(result['metrics'][row][k]-proof['results'][old][k]) > .0001 for k in ('r1', 'r10', 'r50')):
            raise ValueError('Frozen official baseline failed reproduction')
    require_space(args.output)
    atomic_write_json(args.output/'results.json', dict(status='complete', category=args.dataset, training=training,
        counts=context['counts'], isolation_proof=context['isolation_proof'], evidence=context['evidence'], validation_evidence=val_evidence,
        internal_dev_retrieval_all_256=internal_result, official_validation=result, fixed_baselines_reproduced=True,
        deltas_pp_vs_fixed05={name: {k: values[k]-result['metrics']['fixed_dual05'][k] for k in ('r1', 'r10', 'r50')}
                             for name, values in result['metrics'].items()}, api_calls=0,
        scope='Exploratory seed42, byte-clean supervision only, untouched official ID benchmark, no significance or whole-backbone independence claim.'))
    atomic_write_json(args.output/'state.json', dict(status='complete'))
    print(json.dumps(dict(status='complete', category=args.dataset, counts=context['counts'], official_metrics=result['metrics'])), flush=True)


if __name__ == '__main__':
    main()
