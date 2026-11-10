"""020 metadata only. Official query labels are not parsed until six fits finish."""
import argparse,gzip,json,hashlib,sqlite3
from pathlib import Path
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json,load_fashioniq_queries
from fiq_byte_clean_pools import clean_partition,positive_byte_sets

DATA=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ')
OLD=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_small_improvements_20261026')
PRIMARY=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_shirt_toptee_20261022')
ARMS=('original','slots','dual')

def heldout():
 p=OLD/'train_only_candidate_head/official_heldout_byte_index.json.gz';d=json.loads(gzip.decompress(p.read_bytes()))
 if file_sha(p)!=json.loads((p.parent/'byte_partition_audit_v2.json').read_text())['heldout_byte_index_sha256']:raise ValueError('Heldout byte index changed')
 for name,h in d['metadata_sha256'].items():
  if file_sha(DATA/'image_splits'/name)!=h:raise ValueError('Heldout metadata changed')
 return d

def prepare(root,phase):
 h=heldout();root.mkdir(exist_ok=True);metadata=root/'metadata';metadata.mkdir(exist_ok=True)
 if phase=='val':
  for cat in ('shirt','toptee'):
   for arm in ARMS:
    p=root/'fits'/cat/arm/'fit_complete.json'
    if not p.exists() or json.loads(p.read_text())['status']!='complete':raise ValueError('All fits must finish BEFORE official labels are parsed')
 for cat in ('shirt','toptee'):
  target=metadata/(cat+'_'+phase+'.json')
  if target.exists():raise FileExistsError('Do not overwrite020 metadata')
  if phase=='train':
   f=OLD/'train_only_candidate_head_byte_clean'/cat;m=json.loads((f/'manifest.json').read_text())
   audit=json.loads((f.parent/'artifact_sha256.json').read_text())
   if file_sha(f/'manifest.json')!=audit[str((f/'manifest.json').relative_to(OLD))]:raise ValueError('Sealed009 manifest changed')
   gallery_path=OLD/'train_gallery_diagnostic'/cat/'manifest.json';gm=json.loads(gallery_path.read_text());ids=gm['gallery_ids'];hashes=gm['image_sha256']
   if file_sha(gallery_path)!=json.loads((gallery_path.parent.parent/'completion_summary.json').read_text())['categories'][cat]['manifest_sha256']:raise ValueError('005 gallery provenance')
   policy=clean_partition(ids,hashes,m['fit'],m['internal_dev'],h['image_sha256'].values())
   if policy['masks']!=m['supervision_masks'] or policy['active']!=m['active_original_rows']:raise ValueError('009 exact isolation did not reproduce')
   records=[];offsets={}
   for name,rows in [('fit',m['fit']),('dev',m['internal_dev'])]:
    start=len(records)
    for i in policy['active'][name]:
     row=rows[i];records.append(dict(row,split020=name,original_row=i+(1024 if name=='dev' else 0)))
    offsets[name]=list(range(start,len(records)))
   full_rows=load_fashioniq_queries(DATA,cat,'train')
   for row in records:
    actual=full_rows[row['source_index']]
    if any(row[k]!=actual[k] for k in ('candidate','target','modification_text')):raise ValueError('Original train query binding changed')
   available=[row for row in full_rows if row['candidate'] in hashes and row['target'] in hashes]
   known=positive_byte_sets(available,hashes)
   # All official-heldout IDs were removed from005; full train labels may refer
   # to IDs outside005. Only store positive SHA for those present in this bank.
   banks={name:[ids[j] for j in policy['banks'][name]] for name in ('fit','dev')}
   canonical={name:[ids[j] for j in policy['canonical'][name]] for name in ('fit','dev')}
   inputs={name:{hashes[i] for i in banks[name]} for name in banks}
   if inputs['fit']&inputs['dev'] or any(inputs[n]&set(h['image_sha256'].values()) for n in inputs):raise ValueError('Byte bank isolation')
   d=dict(category=cat,phase=phase,records=records,offsets=offsets,banks=banks,canonical=canonical,image_sha256=hashes,
    known_positive_sha={json.dumps(k):list(v) for k,v in known.items()},source009_manifest_sha256=file_sha(f/'manifest.json'),gallery005_manifest_sha256=file_sha(gallery_path),heldout_index_sha256=file_sha(OLD/'train_only_candidate_head/official_heldout_byte_index.json.gz'))
  else:
   records=load_fashioniq_queries(DATA,cat,'val');ids=json.loads((DATA/'image_splits'/('split.'+cat+'.val.json')).read_text())
   if len(records)!={'shirt':2038,'toptee':1961}[cat] or len(ids)!={'shirt':6346,'toptee':5373}[cat]:raise ValueError('Original full validation count changed')
   if any(row['candidate'] not in ids or row['target'] not in ids for row in records):raise ValueError('Missing original reference/GT, no insertion allowed')
   d=dict(category=cat,phase=phase,records=records,gallery_ids=ids,image_sha256={i:h['image_sha256'][i] for i in ids},
    caption_sha256=file_sha(DATA/'captions'/('cap.'+cat+'.val.json')),split_sha256=file_sha(DATA/'image_splits'/('split.'+cat+'.val.json')),
    all_fits_complete_before_label_load=True)
  atomic_write_json(target,d)
 print(json.dumps({'phase':phase,'counts':{cat:len(json.loads((metadata/(cat+'_'+phase+'.json')).read_text())['records']) for cat in ('shirt','toptee')}}))

def generation_rows(root,phase):
 result=[]
 for cat in ('shirt','toptee'):
  m=json.loads((root/'metadata'/(cat+'_'+phase+'.json')).read_text());ledger=sqlite3.connect('file:'+str(PRIMARY/cat/'requests.sqlite3')+'?mode=ro',uri=True)
  for i,row in enumerate(m['records']):
   state=ledger.execute('select state from requests where split=? and idx=?',('train' if phase=='train' else 'val',row['source_index'])).fetchall()
   blocked=any(x[0] in ('unknown','reserved','metered') for x in state)
   result.append(dict(category=cat,index=i,phase=phase,source_index=row['source_index'],reference=row['candidate'],instruction=row['modification_text'],
    reference_path=str(DATA/'resized_image'/(row['candidate']+'.jpg')),reference_sha256=m['image_sha256'][row['candidate']],blocked_old_unknown=blocked))
  ledger.close()
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('train','val'),required=True);a=p.parse_args();prepare(a.root,a.phase)
