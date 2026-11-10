"""Full024 read-only verification of saved outputs; no fitting, embeddings or API."""
import argparse,json,math,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from audit_expanded_attr_020 import sha,read,check,metrics,transition
from fiq_full_metadata_024 import CATS,heldout
from fiq_full_partition_024 import partition
from expanded_attr_resume_022 import reconcile


def audit(root):
    state=read(root/'state.json');plan=read(root/'plan.json');check(state['status'] in ('independent_full_audit','complete'),'Full retrieval is not finished')
    gate=read(root/'validation_gate_certificate.json');validation=Path(plan['validation_root'])
    check(gate['status']=='passed' and sha(validation/'audit/verification.json')==gate['validation_audit_sha256'],'Successful023 validation gate missing')
    check(sha(validation/'state.json')==gate['validation_state_sha256'] and read(validation/'state.json')['status']=='complete','Validation completion changed')
    for name,h in plan['metadata_hashes'].items():check(sha(name)==h,'Sealed full metadata changed')
    for name,h in plan['source_hashes'].items():check(sha(ROOT/name)==h,'Sealed source changed: '+name)
    for name,d in plan['protected_texts'].items():check(sha(name)==d['sha256'],'Protected source changed: '+name)
    for name,d in plan['protected_checkpoints'].items():check(sha(name)==d['sha256'],'Frozen category checkpoint changed')
    arms=('original','slots','dual');output=dict(status='verified',categories={},macro_metrics={},new_api_batches=[],limitations=['Single seed exploratory development','Original backbone already saw train images','Internal dev used for checkpoint selection','Exact bytes do not detect reencoded near duplicates'])
    prior={}
    for parent in map(Path,('/data/8T1/lcy/CIRData/runs/fiq-expanded-021','/data/8T1/lcy/CIRData/runs/fiq-expanded-022','/data/8T1/lcy/CIRData/runs/fiq-expanded-023')):
        values,_=reconcile(parent);check(not set(values)&set(prior),'Duplicate prior request identity');prior.update(values)
    requests=set();heldout_bytes=set(heldout()['image_sha256'].values())
    for cat in CATS:
        m=read(root/'metadata'/(cat+'_train.json'));held=read(root/'full_train_inventory.json')['categories'][cat];n=plan['fit_counts'][cat];ndev=plan['dev_counts'][cat]
        check(len(m['all_original_records'])==m['original_train_rows'] and len(m['supervision_audit'])==m['original_train_rows'],'Original query audit coverage')
        check(len(m['records'])==n+ndev and sum(a['eligible'] for a in m['supervision_audit'])==n+ndev,'Supervised query coverage')
        check(not any(m['byte_intersections'].values()),'Learning bytes overlap')
        hashes=m['image_sha256'];used={s:{hashes[i] for i in m['banks'][s]} for s in ('fit','dev')};check(not used['fit']&used['dev'],'Actual fit/dev bank overlap')
        check(not (used['fit']|used['dev'])&heldout_bytes,'Actual learning bank overlaps official heldout bytes')
        policy=partition(list(hashes),hashes,m['all_original_records'],heldout_bytes)
        check(policy['banks']==m['banks'] and policy['canonical']==m['canonical'] and policy['audit']==m['supervision_audit'],'Component partition or mask does not reproduce')
        check(policy['selected']['fit']+policy['selected']['dev']==m['records'],'Original query selection changed')
        for row in m['records']:check(row['candidate'] in m['banks'][row['split020']] and row['target'] in m['banks'][row['split020']],'Reference/GT outside byte-clean learning bank')
        fits={}
        for arm in arms:
            directory=root/'fits'/cat/arm;d=read(directory/'results.json');check(d['updates']==3*math.ceil(n/8) and d['sample_presentations']==3*n and len(d['per_fit_row_exposures'])==n and set(d['per_fit_row_exposures'])=={3},'Incomplete full fitting')
            check(d['old_vocabulary_exact'] and d['frozen_sha256_before']==d['frozen_sha256_after'],'Freeze integrity failed')
            check([h['epoch'] for h in d['history']]==[0,1,2,3],'Missing full epoch')
            for h in d['history']:
                v=h['metrics'];r=v['ranks'];check(len(r)==ndev and all(type(x) is int and 1<=x<len(m['banks']['dev']) for x in r),'Invalid dev ranks');check(math.isfinite(v['retrieval_nce']),'Nonfinite dev loss');check(all(abs(metrics(r)[k]-v[k])<.0001 for k in ('r1','r10','r50')),'Dev metric mismatch');check(sha(h['weight_path'])==h['weight_sha256'],'Full epoch weight changed');check(Path(h['weight_path']).resolve().is_relative_to(Path(plan['weight_directory']).resolve()),'Full weight is outside declared storage')
            selected=max(d['history'],key=lambda h:((h['metrics']['r10']+h['metrics']['r50'])/2,-h['metrics']['retrieval_nce']));check(selected['epoch']==d['selected_epoch'],'Wrong R10/R50 checkpoint')
            check(d['selected_weight']==selected['weight_path'] and d['selected_weight_sha256']==selected['weight_sha256'],'Selected weight provenance mismatch')
            expected_text_sha=m['caption_sha256'] if arm=='original' else sha(root/'texts'/(cat+'_train.json'));check(d['text_sha256']==expected_text_sha,'Training text provenance mismatch')
            for line in (directory/'training.jsonl').read_text().splitlines():
                row=json.loads(line);check(all(math.isfinite(v) for v in row['loss'].values()) and math.isfinite(row['gradient_norm']),'Nonfinite training trace')
            fits[arm]=dict(selected_epoch=d['selected_epoch'],updates=d['updates'],presentations=d['sample_presentations'],selected_weight=d['selected_weight'],selected_weight_sha256=d['selected_weight_sha256'],dev_metrics={k:selected['metrics'][k] for k in ('r1','r10','r50','retrieval_nce')})
        result=read(root/'official'/cat/'results.json');check(result['queries']==plan['official_queries'][cat] and result['gallery']==plan['official_gallery'][cat],'Full official query/gallery count mismatch');check(result['reference_id_excluded'] and not result['gt_inserted'] and result['all_official_queries_retained'],'Full original-ID protocol mismatch')
        for category in CATS:
            for arm in arms:check((root/'fits'/category/arm/'fit_complete.json').stat().st_mtime_ns<(root/'metadata'/(cat+'_val.json')).stat().st_mtime_ns,'Official labels loaded before all fits')
        rows=result['rows'];check(set(rows)=={'frozen_original',*arms},'Missing full comparison row')
        for label,v in rows.items():
            rr=v['ranks'];check(len(rr)==result['queries'] and all(type(x) is int and 1<=x<result['gallery'] for x in rr),'Invalid full official ranks');check(all(abs(metrics(rr)[k]-v[k])<.0001 for k in ('r1','r10','r50')),'Official rank metric mismatch')
        comparisons={a+'-minus-'+b:{str(k):transition(rows[a]['ranks'],rows[b]['ranks'],k) for k in (1,10,50)} for a,b in [(a,'frozen_original') for a in arms]+[('slots','original'),('dual','original'),('dual','slots')]}
        generation={}
        for phase in ('train','val'):
            text=read(root/'texts'/(cat+'_'+phase+'.json'));expected=n+ndev if phase=='train' else result['queries'];check(len(text['samples'])==expected and text['all_rows_retained'] and not text['targets_sent'],'Full text query coverage')
            for s in text['samples']:
                if s['origin'].startswith('reused:'):
                    previous=prior.get(s['request_key']);check(previous is not None and s['status']==previous['status'] and s['target_attrs']==previous['target_attrs'],'A reused response changed')
            generation[phase]=dict(rows=expected,accepted=sum(s['status']=='ok' for s in text['samples']),fallback=sum(s['status']!='ok' for s in text['samples']),sha256=sha(root/'texts'/(cat+'_'+phase+'.json')))
        output['categories'][cat]=dict(data=held,fits=fits,queries=result['queries'],gallery=result['gallery'],generation=generation,metrics={label:metrics(v['ranks']) for label,v in rows.items()},comparisons=comparisons)
    db=sqlite3.connect('file:/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3?mode=ro',uri=True)
    for directory in sorted((root/'generation').iterdir()):
        if not directory.is_dir():continue
        p=read(directory/'protocol.json');c=read(directory/'completion.json');check(c['status']=='complete','A full generation batch failed');allocation=db.execute('select charged,state from batches where id=?',(p['id'],)).fetchone();check(allocation and allocation[1]=='settled','API allocation not settled')
        for row in p['rows']:
            check(row['request_key'] not in prior and row['request_key'] not in requests,'Old request duplicated');requests.add(row['request_key']);check(not {'target','target_id','target_path','gt_rank'}&set(row),'Target information entered generation task')
        ledger=sqlite3.connect('file:'+str(directory/'requests.sqlite3')+'?mode=ro',uri=True);check(ledger.execute("select count(*) from requests where state in ('unknown','reserved','metered')").fetchone()[0]==0,'Operational API failure/inflight in a complete full study');ledger.close();output['new_api_batches'].append(dict(id=p['id'],accounted_upper_cny=allocation[0]/1e6,cost=c['cost']))
    output['new_accounted_upper_cny']=sum(v['accounted_upper_cny'] for v in output['new_api_batches']);output['cumulative_accounted_upper_cny']=(db.execute('select sum(upper) from history').fetchone()[0]+db.execute('select sum(coalesce(charged,cap)) from batches').fetchone()[0])/1e6;db.close()
    for label in ('frozen_original',*arms):output['macro_metrics'][label]={k:sum(output['categories'][cat]['metrics'][label][k] for cat in CATS)/3 for k in ('r1','r10','r50')}
    destination=root/'audit';destination.mkdir(exist_ok=True);payload=json.dumps(output,indent=2)+'\n';p=destination/'verification.json'
    if p.exists():check(p.read_text()==payload,'Do not overwrite different verification')
    else:p.write_text(payload)
    print(json.dumps(dict(status='verified',macro_metrics=output['macro_metrics'],new_accounted_upper_cny=output['new_accounted_upper_cny'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();audit(a.root)
