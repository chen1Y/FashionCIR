"""016 pure metadata competitor coverage; no model/embedding/API/new retrieval."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from fiq_visible_competition import analyze,summarize


def prepare(a):
    r=a.research;cat=a.dataset;f=r/'supported_candidate_head_011'/cat
    s011=json.loads((f.parent/'completion_summary.json').read_text())['categories'][cat]
    if file_sha(f/'manifest.json')!=s011['manifest_sha256']:raise ValueError('011 manifest changed')
    m=json.loads((f/'manifest.json').read_text());poolsha=file_sha(f/'candidate_pools.pt');art=json.loads((f.parent/'artifact_sha256.json').read_text())
    if poolsha!=art[str((f/'candidate_pools.pt').relative_to(r))]:raise ValueError('PT changed')
    f014=r/'unused_head_query_inventory_014';art014=json.loads((f014/'artifact_sha256.json').read_text())
    for p in (f014/cat/'results.json',f014/'verification.json'):
        if file_sha(p)!=art014[str(p.relative_to(r))]:raise ValueError('014 independent audit changed')
    proof=json.loads((f014/'verification.json').read_text())[cat];old=json.loads((f014/cat/'results.json').read_text())['evidence']
    if not proof['independent_pool_input_reconstruction'] or old['current011_pool_sha256']!=poolsha or old['current011_manifest_sha256']!=file_sha(f/'manifest.json'):raise ValueError('PT audit binding failed')
    f010=r/'candidate_alignment_010';s010=json.loads((f010/'completion_summary.json').read_text())['categories'][cat]
    if file_sha(f010/cat/'results.json')!=s010['results_sha256'] or file_sha(f010/cat/'protocol.json')!=s010['protocol_sha256']:raise ValueError('010 changed')
    d=json.loads((f010/cat/'results.json').read_text());e=d['evidence']['inherited_009_evidence']
    if d['status']!='complete' or d['official_val_queries_loaded']!=0:raise ValueError('Wrong010 scope')
    f009=r/'train_only_candidate_head_byte_clean'/cat;s009=json.loads((f009.parent/'completion_summary.json').read_text())['categories'][cat]
    if file_sha(f009/'manifest.json')!=s009['manifest_sha256']:raise ValueError('009 manifest changed')
    clean=json.loads((f009/'manifest.json').read_text())
    if clean['fit']!=m['original_fit'] or clean['internal_dev']!=m['original_dev'] or clean['supervision_masks']!=m['inherited009_masks']:raise ValueError('Row lineage changed')
    for key,name in [('byte_clean_source_sha256','train_fiq_byte_clean_head.py'),('byte_pool_source_sha256','fiq_byte_clean_pools.py')]:
        if file_sha(Path(__file__).with_name(name))!=e[key]:raise ValueError('Frozen source changed')
    gm=r/'train_gallery_diagnostic'/cat/'manifest.json';s005=json.loads((gm.parent.parent/'completion_summary.json').read_text())['categories'][cat]
    if file_sha(gm)!=s005['manifest_sha256']:raise ValueError('005 mapping changed')
    hashes=json.loads(gm.read_text())['image_sha256']
    for name,h in e['inherited_gallery_evidence']['metadata_sha256'].items():
        if file_sha(a.root/'image_splits'/name)!=h:raise ValueError('Split metadata changed')
    hp=r/'train_only_candidate_head/official_heldout_byte_index.json.gz'
    if file_sha(hp)!=e['byte_index_sha256']:raise ValueError('Heldout byte index changed')
    held=set(json.loads(gzip.decompress(hp.read_bytes()))['image_sha256'].values());allrows=m['original_fit']+m['original_dev'];used={};splits={}
    for name,offset in [('fit',0),('dev',1024)]:
        rows=m['original_fit'] if name=='fit' else m['original_dev'];known=defaultdict(set)
        for row in rows:known[(hashes[row['candidate']],row['modification_text'])].add(hashes[row['target']])
        details={q['original_query_row']:q for q in d['splits'][name]['queries']}
        if len(details)!=len(d['splits'][name]['queries']) or list(details)!=[i+offset for i,v in enumerate(clean['supervision_masks'][name]) if v]:raise ValueError('010 row order changed')
        selected=m['selected_original_rows'][name];pools=m['candidate_ids'][name]
        if len(selected)!={'shirt':{'fit':641,'dev':150},'toptee':{'fit':771,'dev':190}}[cat][name] or len(selected)!=len(pools):raise ValueError('Support count changed')
        allowed=set(clean[name+'_gallery_ids']);records=[];used[name]=set()
        for index,pool in zip(selected,pools):
            row=allrows[index];q=details[index]
            if len(q['top20_ids'])!=20 or q['target_id']!=row['target'] or not q['structured_valid']:raise ValueError('Wrong supported head')
            record=analyze(q,pool,hashes,allowed,known[(hashes[row['candidate']],row['modification_text'])],row['candidate']);record.update(source_index=row['source_index'],modification_text=row['modification_text'],query_id=row['query_id']);records.append(record);used[name].update(hashes[i] for i in pool+[row['candidate']])
        if used[name]&held:raise ValueError('Actual heldout overlap')
        splits[name]=dict(queries=records,summary={k:summarize(v) for k,v in [('all',records),('top1',[q for q in records if q['full_gt_rank']==1]),('rank2_20',[q for q in records if q['full_gt_rank']>1])]})
    if used['fit']&used['dev']:raise ValueError('Actual fit/dev overlap')
    return dict(category=cat,splits=splits,proof=dict(reused014_pt_id_verification=True,actual_fit_dev_byte_overlap=0,actual_heldout_byte_overlap=0,gt_injected=False),evidence=dict(source_sha256=file_sha(__file__),math_sha256=file_sha(Path(__file__).with_name('fiq_visible_competition.py')),diagnostic010_sha256=file_sha(f010/cat/'results.json'),manifest011_sha256=file_sha(f/'manifest.json'),pool011_sha256=poolsha,independent014_audit_sha256=file_sha(f014/'verification.json'),manifest009_sha256=file_sha(f009/'manifest.json'),gallery005_manifest_sha256=file_sha(gm),heldout_index_sha256=file_sha(hp)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=('shirt','toptee'),required=True)
    for name in ('root','research','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve016; no blind repeat')
    require_space(a.output);d=prepare(a)
    if a.preflight:print(json.dumps(dict(status='preflight_passed',support={n:len(s['queries']) for n,s in d['splits'].items()},proof=d['proof'],api_calls=0)),flush=True);return
    a.output.mkdir(parents=True);atomic_write_json(a.output/'protocol.json',dict(evidence=d['evidence'],scope='All011 fixed fit/internal NCE-dev support; post-hoc coverage, not causal or independent confirmation. Shared gallery and seen backbone remain.',rank='Original010 head order; never positive pool storage column.',exclusions='Keep allowed intersection, actualGT, known-positive/ref exclusions; no reintroduction or semantic-positive invention.',api_calls=0,model_loading=False,embedding_loading=False,training=False,official_val_queries_loaded=0))
    d.update(status='complete',api_calls=0,fit_updates=0,official_val_queries_loaded=0);require_space(a.output);atomic_write_json(a.output/'results.json',d);atomic_write_json(a.output/'state.json',dict(status='complete'));print(json.dumps(dict(status='complete',category=a.dataset,summaries={n:s['summary'] for n,s in d['splits'].items()})),flush=True)


if __name__=='__main__':main()
