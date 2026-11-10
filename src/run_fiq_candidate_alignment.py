"""010 train/internal-dev-only candidate coverage diagnostic; no official val queries, fit or API."""
import argparse
import gzip
import json
from pathlib import Path

import torch

from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from train_fiq_candidate_head import one_cache
from fiq_byte_clean_pools import verify_actual_inputs, positive_byte_sets
from fiq_candidate_alignment_math import analyze_row, summarize
from fiq_small_math import blend


def load_inputs(args):
    clean = args.research/'train_only_candidate_head_byte_clean'
    summary = json.loads((clean/'completion_summary.json').read_text())['categories'][args.dataset]
    folder = clean/args.dataset
    for name, digest in [('results.json', summary['results_sha256']), ('manifest.json', summary['manifest_sha256']),
                         ('head.pt', summary['head_sha256'])]:
        if file_sha(folder/name) != digest:
            raise ValueError('Sealed009 artifact changed')
    artifacts = json.loads((clean/'artifact_sha256.json').read_text())
    if file_sha(folder/'candidate_pools.pt') != artifacts[str((folder/'candidate_pools.pt').relative_to(args.research))]:
        raise ValueError('Saved009 loss pools changed')
    result = json.loads((folder/'results.json').read_text())
    if result['status'] != 'complete' or not result['fixed_baselines_reproduced']:
        raise ValueError('009 incomplete')
    evidence = result['evidence']
    for key, filename in [('byte_clean_source_sha256', 'train_fiq_byte_clean_head.py'),
                          ('byte_pool_source_sha256', 'fiq_byte_clean_pools.py')]:
        if file_sha(Path(__file__).with_name(filename)) != evidence[key]:
            raise ValueError('009 source changed')
    qpath, query = one_cache(args.query_cache, args.dataset, evidence['inherited_query_evidence'])
    gpath, gallery = one_cache(args.gallery_cache, args.dataset, evidence['inherited_gallery_evidence'])
    if file_sha(qpath) != evidence['query_cache_sha256'] or file_sha(gpath) != evidence['gallery_cache_sha256']:
        raise ValueError('Frozen feature cache SHA changed')
    ge = evidence['inherited_gallery_evidence']
    for name, digest in ge['metadata_sha256'].items():
        if file_sha(args.root/'image_splits'/name) != digest:
            raise ValueError('Metadata changed')
    gm_path = args.research/'train_gallery_diagnostic'/args.dataset/'manifest.json'
    expected = json.loads((args.research/'train_gallery_diagnostic/completion_summary.json').read_text())
    if file_sha(gm_path) != expected['categories'][args.dataset]['manifest_sha256']:
        raise ValueError('Gallery byte manifest changed')
    gm = json.loads(gm_path.read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    rows = {'fit': manifest['fit'], 'dev': manifest['internal_dev']}
    if gallery['ids'] != gm['gallery_ids'] or query['source_ids'] != [r['candidate'] for r in rows['fit']+rows['dev']]:
        raise ValueError('Query/gallery identities changed')
    if query['target_ids'] != [r['target'] for r in rows['fit']+rows['dev']]:
        raise ValueError('Positive identities changed')
    pools = torch.load(folder/'candidate_pools.pt', map_location='cpu', weights_only=True)
    held_path = args.research/'train_only_candidate_head/official_heldout_byte_index.json.gz'
    if file_sha(held_path) != evidence['byte_index_sha256']:
        raise ValueError('Official heldout byte exclusion changed')
    held = json.loads(gzip.decompress(held_path.read_bytes()))
    index = {i: j for j, i in enumerate(gallery['ids'])}
    banks = {n: [index[i] for i in manifest[n+'_gallery_ids']] for n in rows}
    proof = verify_actual_inputs(gallery['ids'], gm['image_sha256'], rows, manifest['active_original_rows'],
        {n: p['indices'].tolist() for n, p in pools.items()}, banks, set(held['image_sha256'].values()))
    if proof != result['isolation_proof']:
        raise ValueError('009 input isolation not reproducible')
    for name, offset in [('fit', 0), ('dev', 1024)]:
        if pools[name]['original_rows'].tolist() != [i+offset for i in manifest['active_original_rows'][name]]:
            raise ValueError('Eligible pool row mapping changed')
    return query, gallery, manifest, rows, pools, gm['image_sha256'], evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('shirt', 'toptee'), required=True)
    for name in ('root', 'research', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--query-cache', type=Path, default=Path('/dev/shm/fiq_gate_features_20261026'))
    parser.add_argument('--gallery-cache', type=Path, default=Path('/dev/shm/fiq_train_gallery_20261026'))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing010; no blind repeat')
    require_space(args.output)
    torch.set_num_threads(4)
    query, gallery, manifest, rows, pools, hashes, inherited = load_inputs(args)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed', counts={n:len(p['original_rows']) for n,p in pools.items()},
            gallery=len(gallery['ids']), official_val_queries_loaded=0, api_calls=0)), flush=True)
        return
    args.output.mkdir(parents=True)
    evidence = dict(inherited_009_evidence=inherited, source_sha256=file_sha(__file__),
                    math_source_sha256=file_sha(Path(__file__).with_name('fiq_candidate_alignment_math.py')))
    atomic_write_json(args.output/'protocol.json', dict(evidence=evidence, correction_bound=.03, head_size=20,
        score='Normalized0.5 frozen query fusion', inference='Full original train gallery; reference ID excluded; no GT injection.',
        supervision='Only009 eligible fit/internaldev rows. Diagnostic reads training labels; no fitting/selection/official-val queries.',
        bound='Closed +/-0.03 independent candidate oracle, stable ties. Optimistic attainable-set upper bound, not model result; actual tanh may not attain endpoints.',
        rows='All eligible rows; report invalid structured fallback and full-gallery GT outside20. No query selection by GT ranking.',
        limitation='Frozen backbone saw official train; shared gallery candidates. Exact JPEG-byte hygiene applies only to new-head supervision.', api_calls=0))
    atomic_write_json(args.output/'state.json', dict(status='diagnosing'))
    index = {i:j for j,i in enumerate(gallery['ids'])}
    categories = {}
    for name in ('fit', 'dev'):
        active = pools[name]['original_rows']
        scores = blend(query['qd'][active], query['qq'][active], .5) @ gallery['gallery'].T
        sources = torch.tensor([index[query['source_ids'][i]] for i in active.tolist()])
        targets = torch.tensor([index[query['target_ids'][i]] for i in active.tolist()])
        scores[torch.arange(len(active)), sources] = -torch.inf
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        rank = (order==targets[:,None]).nonzero(as_tuple=False)[:,1]+1
        known = positive_byte_sets(rows[name], hashes)
        allowed = set(manifest[name+'_gallery_ids'])
        details = []
        for j, original in enumerate(active.tolist()):
            target_id, source_id = query['target_ids'][original], query['source_ids'][original]
            local = original if name=='fit' else original-1024
            forbidden = known[(hashes[source_id],rows[name][local]['modification_text'])] | {hashes[source_id]}
            top = order[j,:20]
            detail = analyze_row([gallery['ids'][i] for i in top.tolist()], scores[j,top].tolist(),
                target_id, int(rank[j]), [gallery['ids'][i] for i in pools[name]['indices'][j].tolist()], hashes,
                allowed, forbidden, bool(query['valid'][original]))
            detail['original_query_row'] = original
            details.append(detail)
        categories[name] = dict(summary=summarize(details), queries=details)
    require_space(args.output)
    atomic_write_json(args.output/'results.json', dict(status='complete', category=args.dataset, evidence=evidence,
        gallery=len(gallery['ids']), splits=categories, official_val_queries_loaded=0, fitted_parameters=0, api_calls=0))
    atomic_write_json(args.output/'state.json', dict(status='complete'))
    print(json.dumps(dict(status='complete', category=args.dataset, splits={k:v['summary'] for k,v in categories.items()})), flush=True)


if __name__ == '__main__':
    main()
