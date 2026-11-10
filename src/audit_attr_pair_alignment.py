"""Independent018 saved-rank, delta and exact-byte verification; no fitting."""
import argparse,gzip,json,hashlib
from pathlib import Path
import torch
from fashioniq_noise import file_sha

def transition(new,old,k):
 g=sum(a<=k<b for a,b in zip(new,old));l=sum(b<=k<a for a,b in zip(new,old))
 return dict(gains=g,losses=l,net_pp=100*(g-l)/len(new))

def main():
 p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.research
 plan=json.loads((r/'attr_pair_018_control/plan.json').read_text());state=json.loads((r/'attr_pair_018_control/state.json').read_text())
 if state['status']!='complete' or any(j['status']!='complete' or j['returncode']!=0 for j in state['jobs']):raise ValueError('Incomplete018')
 code=Path(__file__).resolve().parents[1]
 for n,h in plan['source_hashes'].items():
  if file_sha(code/n)!=h:raise ValueError('Sealed018 code changed')
 index=r/'train_only_candidate_head/official_heldout_byte_index.json.gz';old=json.loads((index.parent/'byte_partition_audit_v2.json').read_text())
 if file_sha(index)!=old['heldout_byte_index_sha256']:raise ValueError('Heldout byte index changed')
 held=json.loads(gzip.decompress(index.read_bytes()));held_sha=set(held['image_sha256'].values())
 for n,h in held['metadata_sha256'].items():
  if file_sha(a.root/'image_splits'/n)!=h:raise ValueError('Heldout split metadata changed')
 result=dict(status='verified',categories={},api_calls=0,model_loaded=False,official_val_queries_loaded=0)
 for cat in ('shirt','toptee'):
  folder=r/'attr_pair_alignment_018'/cat;d=json.loads((folder/'results.json').read_text());m=json.loads((folder/'manifest.json').read_text())
  gallery=r/'train_gallery_diagnostic'/cat/'manifest.json';summary=json.loads((gallery.parent.parent/'completion_summary.json').read_text())['categories'][cat]
  if file_sha(gallery)!=summary['manifest_sha256']:raise ValueError('Gallery byte manifest changed')
  hashes=json.loads(gallery.read_text())['image_sha256'];sets={k:set() for k in ('fit','dev')}
  for row in m['records']:
   if row['target']!=row['pool_ids'][0] or len(row['pool_ids'])<2:raise ValueError('GT/pool binding')
   sets[row['split']].update(hashes[i] for i in [row['reference']]+row['pool_ids'])
  intersections=dict(fit_dev=len(sets['fit']&sets['dev']),fit_heldout=len(sets['fit']&held_sha),dev_heldout=len(sets['dev']&held_sha))
  if any(intersections.values()):raise ValueError('Byte isolation violation')
  devrows=[m['records'][i] for i in m['offsets']['dev']]
  metrics=[d['baseline']]+[y for v in d['arms'].values() for y in [v['initial'],v['selected']]+[h['metrics'] for h in v['history']]]
  for metric in metrics:
   ranks=metric['ranks']
   if len(ranks)!=32 or any(not isinstance(x,int) or not 1<=x<=len(row['pool_ids']) for x,row in zip(ranks,devrows)):raise ValueError('Illegal ranks')
   for k in (1,5,10):
    if abs(100*sum(x<=k for x in ranks)/32-metric['candidate_pool_r'+str(k)])>1e-8:raise ValueError('Metric/count mismatch')
  exposure=None;arms={}
  for arm,v in d['arms'].items():
   if v['updates']!=50 or v['presentations']!=200 or sum(v['exposure'])!=200:raise ValueError('Missing updates/exposures')
   if exposure is None:exposure=v['exposure']
   elif exposure!=v['exposure']:raise ValueError('Arms did not use matched sampling')
   best=min(v['history'],key=lambda h:h['metrics']['retrieval'])
   if best['step']!=v['selected_step'] or best['metrics']!=v['selected']:raise ValueError('NCE checkpoint selection mismatch')
   if v['frozen_sha256_before']!=v['frozen_sha256_after'] or not v['old_vocabulary_exact'] or not v['vit_frozen']:raise ValueError('Freeze failure')
   weight=folder/(arm+'_delta.pt')
   if file_sha(weight)!=v['delta_sha256']:raise ValueError('Delta SHA mismatch')
   payload=torch.load(weight,map_location='cpu',weights_only=True);s=payload['state'];blocks=s['blocks']
   if list(s['new_token_rows'].shape)!=[4,1024] or not torch.isfinite(s['new_token_rows']).all():raise ValueError('Token delta shape/finite')
   if any(not n.startswith(('transformer.resblocks.22.','transformer.resblocks.23.')) or not torch.isfinite(t).all() for n,t in blocks.items()):raise ValueError('Unexpected trainable text layer')
   if (arm=='tokens_only' and blocks) or (arm=='tokens_last2' and len(blocks)!=24):raise ValueError('Unexpected block delta')
   arms[arm]=dict(selected_step=v['selected_step'],selected=v['selected'],initial=v['initial'],last_two_block_parameters=sum(t.numel() for t in blocks.values()),
    versus_baseline={str(k):transition(v['selected']['ranks'],d['baseline']['ranks'],k) for k in (1,5,10)},weight_sha256=v['delta_sha256'])
  result['categories'][cat]=dict(baseline=d['baseline'],arms=arms,byte_intersections=intersections,input_queries=len(m['records']),results_sha256=file_sha(folder/'results.json'))
 path=r/'attr_pair_alignment_018/verification.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({c:{'intersections':x['byte_intersections'],'arms':{n:v['last_two_block_parameters'] for n,v in x['arms'].items()}} for c,x in result['categories'].items()}))

if __name__=='__main__':main()
