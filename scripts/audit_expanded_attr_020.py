"""Read-only completion verification for020; no model, fitting or provider calls."""
import argparse, hashlib, json, math, sqlite3
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def check(condition,message):
    if not condition:raise ValueError(message)


def metrics(ranks):return {'r'+str(k):100*sum(r<=k for r in ranks)/len(ranks) for k in (1,10,50)}


def transition(after,before,k):
    a=[r<=k for r in after];b=[r<=k for r in before]
    out=dict(gains=sum(x and not y for x,y in zip(a,b)),losses=sum(y and not x for x,y in zip(a,b)),both=sum(x and y for x,y in zip(a,b)),neither=sum(not x and not y for x,y in zip(a,b)))
    check(sum(out.values())==len(a),'Transition coverage')
    return dict(out,net_pp=100*(out['gains']-out['losses'])/len(a))


def audit(root,code):
    state=read(root/'state.json');plan=read(root/'plan.json')
    check(state['status']=='complete','020 must finish before auditing results')
    for name,h in plan['source_hashes'].items():check(sha(code/name)==h,'Sealed source changed: '+name)
    for name,info in plan['protected_texts'].items():check(sha(name)==info['sha256'],'Old text changed: '+name)
    old=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_small_improvements_20261026')
    import gzip
    heldout_path=old/'train_only_candidate_head/official_heldout_byte_index.json.gz'
    heldout= json.loads(gzip.decompress(heldout_path.read_bytes()))
    check(sha(heldout_path)==read(heldout_path.parent/'byte_partition_audit_v2.json')['heldout_byte_index_sha256'],'Heldout index changed')
    data=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ')
    for name,h in heldout['metadata_sha256'].items():check(sha(data/'image_splits'/name)==h,'Official split source changed')
    output=dict(status='verified',scope='Single seed exploratory full-ID validation; not an independent backbone holdout',categories={},api_batches=[])
    arms=('original','slots','dual')
    for cat in ('shirt','toptee'):
        m=read(root/'metadata'/(cat+'_train.json'));hashes=m['image_sha256']
        sets={s:{hashes[i] for i in m['banks'][s]} for s in ('fit','dev')};held=set(heldout['image_sha256'].values())
        intersections=dict(fit_dev=len(sets['fit']&sets['dev']),fit_heldout=len(sets['fit']&held),dev_heldout=len(sets['dev']&held))
        check(not any(intersections.values()),'Learning banks overlap bytes')
        for row in m['records']:
            check(row['candidate'] in m['banks'][row['split020']] and row['target'] in m['banks'][row['split020']],'Reference/GT outside clean bank')
        fits={}
        for arm in arms:
            directory=root/'fits'/cat/arm;fit=read(directory/'results.json');history=fit['history']
            check([x['epoch'] for x in history]==[0,1,2,3],'Missing epoch')
            n=plan['fit_counts'][cat]
            check(fit['updates']==3*math.ceil(n/8) and fit['sample_presentations']==3*n,'Incomplete training')
            check(len(fit['per_fit_row_exposures'])==n and set(fit['per_fit_row_exposures'])=={3},'Rows did not receive three epochs')
            check(fit['frozen_sha256_before']==fit['frozen_sha256_after'] and fit['old_vocabulary_exact'],'Freeze failed')
            for item in history:
                v=item['metrics'];ranks=v['ranks']
                check(len(ranks)==249 and all(type(r) is int and 1<=r<len(m['banks']['dev']) for r in ranks),'Invalid dev ranks')
                check(all(abs(metrics(ranks)[k]-v[k])<.0001 for k in ('r1','r10','r50')),'Dev metrics do not reproduce')
                check(math.isfinite(v['retrieval_nce']),'Invalid dev NCE')
                check(sha(item['weight_path'])==item['weight_sha256'],'Epoch checkpoint changed')
            best=max(history,key=lambda x:((x['metrics']['r10']+x['metrics']['r50'])/2,-x['metrics']['retrieval_nce']))
            check(fit['selected_epoch']==best['epoch'] and fit['selected_weight_sha256']==best['weight_sha256'],'R10/R50 selection mismatch')
            check((directory/'fit_complete.json').stat().st_mtime_ns<(root/'metadata'/(cat+'_val.json')).stat().st_mtime_ns,'Official labels loaded before all fits')
            fits[arm]=dict(selected_epoch=best['epoch'],dev_metrics={k:best['metrics'][k] for k in ('r1','r10','r50','retrieval_nce')},updates=fit['updates'],presentations=fit['sample_presentations'],selected_weight_sha256=fit['selected_weight_sha256'])
        result=read(root/'official'/cat/'results.json');check(result['queries']==plan['official_queries'][cat] and result['gallery']==plan['official_gallery'][cat],'Official count mismatch')
        check(result['reference_id_excluded'] and not result['gt_inserted'] and result['all_official_queries_retained'],'Official ID protocol mismatch')
        rows=result['rows'];check(set(rows)=={'frozen_original',*arms},'Missing fixed comparison row')
        for label,value in rows.items():
            ranks=value['ranks'];check(len(ranks)==result['queries'] and all(type(r) is int and 1<=r<result['gallery'] for r in ranks),'Invalid official ranks')
            check(all(abs(metrics(ranks)[k]-value[k])<.0001 for k in ('r1','r10','r50')),'Official metrics do not reproduce')
        comparisons={}
        for a,b in [(a,'frozen_original') for a in arms]+[('slots','original'),('dual','original'),('dual','slots')]:
            comparisons[a+'-minus-'+b]={str(k):transition(rows[a]['ranks'],rows[b]['ranks'],k) for k in (1,10,50)}
        generation={}
        for phase in ('train','val'):
            text=read(root/'texts'/(cat+'_'+phase+'.json'));expected=len(m['records']) if phase=='train' else result['queries']
            check(len(text['samples'])==expected and text['all_rows_retained'] and not text['targets_sent'],'Generated query coverage')
            generation[phase]=dict(rows=expected,accepted=sum(r['status']=='ok' for r in text['samples']),fallback=sum(r['status']!='ok' for r in text['samples']),sha256=sha(root/'texts'/(cat+'_'+phase+'.json')))
        output['categories'][cat]=dict(fits=fits,queries=result['queries'],gallery=result['gallery'],byte_intersections=intersections,generation=generation,metrics={label:metrics(v['ranks']) for label,v in rows.items()},comparisons=comparisons)
    db=sqlite3.connect('file:/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3?mode=ro',uri=True)
    for directory in sorted((root/'generation').iterdir()):
        if not directory.is_dir():continue
        p=read(directory/'protocol.json');c=read(directory/'completion.json');row=db.execute('SELECT charged,state FROM batches WHERE id=?',(p['id'],)).fetchone()
        check(c['status']=='complete' and row and row[1]=='settled','Unsettled/failed paid batch')
        ledger=sqlite3.connect('file:'+str(directory/'requests.sqlite3')+'?mode=ro',uri=True)
        check(ledger.execute("SELECT count(*) FROM requests WHERE state IN ('reserved','metered')").fetchone()[0]==0,'Paid requests still in flight');ledger.close()
        output['api_batches'].append(dict(id=p['id'],accounted_upper_cny=row[0]/1e6,cost=c['cost']))
    db.close();output['new_api_accounted_upper_cny']=sum(r['accounted_upper_cny'] for r in output['api_batches'])
    destination=root/'audit';destination.mkdir(exist_ok=True)
    payload=json.dumps(output,indent=2,ensure_ascii=False)+'\n';target=destination/'verification.json'
    if target.exists():check(target.read_text()==payload,'Do not overwrite different audit')
    else:target.write_text(payload)
    artifacts={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.parent!=destination}
    (destination/'artifact_sha256.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--code',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();audit(args.root,args.code)
