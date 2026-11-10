"""015 saved internal-dev rank audit: no fitting, features, model loading or official-val parsing."""
import argparse
import json
from pathlib import Path

from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from fiq_dev_support_audit import selected_fields, support_groups, audit_ranks, recall


def prepare(args):
    r=args.research; cat=args.dataset; f=r/'supported_candidate_head_011'/cat
    s011=json.loads((f.parent/'completion_summary.json').read_text())['categories'][cat]
    for name,key in [('manifest.json','manifest_sha256'),('results.json','results_sha256')]:
        if file_sha(f/name)!=s011[key]: raise ValueError('011 sealed artifact changed')
    m=json.loads((f/'manifest.json').read_text())
    old=json.loads((r/'train_only_route_gate'/cat/'manifest.json').read_text())
    if m['original_dev']!=old['internal_dev'] or m['original_fit']!=old['fit']: raise ValueError('Original query order changed')
    artifact=json.loads((f.parent/'artifact_sha256.json').read_text())
    pool_sha=file_sha(f/'candidate_pools.pt')
    if pool_sha!=artifact[str((f/'candidate_pools.pt').relative_to(r))]: raise ValueError('011 pool changed')
    v011=selected_fields((f/'results.json').read_text(),{'status','evidence','internal_dev_retrieval_all_256'})
    if v011['status']!='complete': raise ValueError('011 incomplete')
    groups=support_groups(m['selected_original_rows']['dev'],m['inherited009_masks']['dev'],m['additional_reasons']['dev'],len(m['original_fit']))
    if len(groups['all'])!=256 or len(groups['supported'])!={'shirt':150,'toptee':190}[cat] or sum(m['inherited009_masks']['dev'])!=249:
        raise ValueError('Original support/byte counts changed')
    summary012=json.loads((r/'rank4_candidate_head_012/completion_summary.json').read_text())['categories'][cat]['arms']
    arms={}; evidence=dict(manifest011_sha256=file_sha(f/'manifest.json'),results011_sha256=file_sha(f/'results.json'),pool011_sha256=pool_sha,
        source_sha256=file_sha(__file__),math_sha256=file_sha(Path(__file__).with_name('fiq_dev_support_audit.py')))
    for arm,s in summary012.items():
        path=r/'rank4_candidate_head_012'/arm/cat/'results.json'
        if file_sha(path)!=s['results_sha256'] or s['source_pool_sha256']!=pool_sha or s['training']['fit_original_rows']!=m['selected_original_rows']['fit']:
            raise ValueError('012 artifact/support lineage changed')
        d=selected_fields(path.read_text(),{'status','evidence','internal_dev_retrieval_all_256','initial011_reproduced'})
        if d['status']!='complete' or not d['initial011_reproduced']: raise ValueError('012 incomplete/unverified initial head')
        for key,name in [('rank4_source_sha256','train_fiq_rank4_head.py'),('rank4_math_sha256','fiq_rank4_head.py'),
                ('support_source_sha256','train_fiq_supported_head.py'),('support_math_sha256','fiq_supported_head.py')]:
            if file_sha(Path(__file__).with_name(name))!=d['evidence'][key]: raise ValueError('Sealed scientific source changed')
        if d['evidence']['initial011_results_sha256']!=s011['results_sha256']: raise ValueError('012 initial result binding changed')
        arms[arm]=d['internal_dev_retrieval_all_256'];evidence[arm+'_results_sha256']=file_sha(path)
    a,b=arms['scalar_continue'],arms['rank4_branch'];v=v011['internal_dev_retrieval_all_256']
    if any(x['queries']!=256 or x['gallery']!=v['gallery'] for x in (a,b)): raise ValueError('Incomplete internal dev')
    for x in (a,b):
        if x['ranks']['fixed_dual05']!=v['ranks']['fixed_dual05'] or x['ranks']['initial011']!=v['ranks']['candidate_head']:
            raise ValueError('Saved full rank vectors fail initial011 reproduction')
    ranks=dict(fixed05=v['ranks']['fixed_dual05'],head011=v['ranks']['candidate_head'],armA=a['ranks']['candidate_head'],
        armB=b['ranks']['candidate_head'],shuffleA=a['ranks']['shuffled_candidate_control'],shuffleB=b['ranks']['shuffled_candidate_control'])
    checks={}
    for label,x in [('011',v),('A',a),('B',b)]:
        checks[label]={}
        for name,row in x['ranks'].items():
            error=max(abs(value-x['metrics'][name][k]) for k,value in recall(row).items())
            if error>0.0001: raise ValueError('Saved internal metrics not reproduced')
            checks[label][name]=error
    audit=audit_ranks(ranks,groups,v['gallery'])
    return dict(category=cat,evidence=evidence,groups=groups,ranks=ranks,audit=audit,original_dev=m['original_dev'],
        additional_reasons=m['additional_reasons']['dev'],metric_reproduction_error_pp=checks,gallery=v['gallery'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,choices=('shirt','toptee'))
    p.add_argument('--research',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve015; do not blindly rerun')
    require_space(args.output); result=prepare(args)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',groups={k:len(v) for k,v in result['groups'].items()},saved_rank_vectors_exact=True,
            metrics_reproduced=True,model_loaded=False,official_validation_parsed=False,api_calls=0)),flush=True);return
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json',dict(evidence=result['evidence'],api_calls=0,model_loading=False,feature_loading=False,
        training=False,official_validation_parsed=False,selection='Fixed011 dev membership and frozen fixed05 rank buckets; no parameter/query selection.',
        scope='Post-hoc internal NCE-selection data description, not independent confirmation. Backbone saw train; galleries shared; whole shuffle is not branch-only control.',
        empty_groups='Report n=0 and null rates; never omit. Original256 queries and raw ranks retained.'))
    result.update(status='complete',api_calls=0,fit_updates=0,official_validation_parsed=False,feature_loading=False,model_loading=False)
    require_space(args.output);atomic_write_json(args.output/'results.json',result);atomic_write_json(args.output/'state.json',dict(status='complete'))
    print(json.dumps(dict(status='complete',category=args.dataset,groups={k:v['queries'] for k,v in result['audit'].items()},
        overall_metrics=result['audit']['all']['metrics'])),flush=True)


if __name__=='__main__':main()
