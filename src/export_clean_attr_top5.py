"""Read-only019 top5 export from sealed weights and cached candidate geometry."""
import argparse,json,shutil,time
from pathlib import Path
import torch
from attr_token_pair_alignment import PairAlignedText,pair_query
from train_clean_attr_alignment import forward
from train_attr_pair_alignment import old
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

SELECTION={'shirt':[1,128],'toptee':[1,129]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError('Inspect existing export; no overwrite')
 a.output.mkdir();assets=a.output/'images';assets.mkdir();rows=[];evidence={};image_sha={}
 for cat,indices in SELECTION.items():
  folder=a.research/'attr_clean_alignment_019'/cat;result=json.loads((folder/'results.json').read_text());cache=Path('/dev/shm/fiq_attr_clean_019_'+cat+'.pt')
  if result['status']!='complete' or file_sha(cache)!=result['new_cache_sha256'] or file_sha(result['weight_path'])!=result['weight_sha256']:raise ValueError('019 cache/weight provenance changed')
  c=torch.load(cache,map_location='cpu',weights_only=True);payload=torch.load(result['weight_path'],map_location='cpu',weights_only=True)
  base=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_shirt_toptee_20261022')/cat/'checkpoints'/(cat+'_dqu_original_seed42_best.pt')
  if file_sha(base)!=c['evidence']['base_sha256']:raise ValueError('Original CLIP changed')
  clip,_,tok=old.load_clip(base);model=PairAlignedText(clip,tok,2).cuda();model.load_delta(payload['state']);model.eval()
  c={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in c.items()};traces={x['index']:x for x in json.loads((folder/'process_examples.json').read_text())['examples']}
  source=json.loads((a.research/'train_gallery_diagnostic'/cat/'manifest.json').read_text())['image_sha256']
  with torch.no_grad():
   for i in indices:
    ix=torch.tensor([i],device='cuda');record=c['records'][i];trace=traces[i]
    with torch.autocast('cuda',dtype=torch.bfloat16):out,mod=forward(model,c,ix);q=pair_query(out,c['v_ref'][ix],mod)
    scores=torch.einsum('bd,bmd->bm',q,c['targets'][ix])[0];n=len(record['pool_ids']);order=sorted(range(n),key=lambda k:(-float(scores[k]),record['original_order'][k]))
    rank=order.index(0)+1
    if rank!=trace['after']['gt_rank'] or [record['pool_ids'][k] for k in order[:3]]!=[x['image_id'] for x in trace['after']['top3']]:raise ValueError('Saved019 rank/top3 did not reproduce')
    for j,k in enumerate(order[:3]):
     if abs(float(scores[k])-trace['after']['top3'][j]['cosine'])>1e-6:raise ValueError('Saved019 score did not reproduce')
    row=dict(category=cat,split=record['split'],source_index=record['source_index'],index=i,reference=record['reference'],target=record['target'],original_modification=record['view']['mod_text'],
     generated_attributes=record['view']['target_attrs'],actual_packed_text=trace['actual_input'],gt_rank=rank,pool_size=n,selected_step=result['selected_step'],
     scope='019 inherited legal SMALL candidate pool; NOT full FashionIQ gallery. GT outside top5 is shown separately, never inserted.',
     top5=[dict(rank=j+1,image_id=record['pool_ids'][k],cosine=float(scores[k]),is_gt=k==0) for j,k in enumerate(order[:5])])
    if len(row['top5'])!=5:raise ValueError('Selected display requires five real candidates')
    rows.append(row)
    for image_id in [row['reference'],row['target']]+[x['image_id'] for x in row['top5']]:
     path=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/resized_image')/(image_id+'.jpg')
     if file_sha(path)!=source[image_id]:raise ValueError('Display JPEG changed')
     if image_id not in image_sha:shutil.copy2(path,assets/path.name);image_sha[image_id]=source[image_id]
  evidence[cat]=dict(weight_sha256=result['weight_sha256'],cache_sha256=result['new_cache_sha256'],results_sha256=file_sha(folder/'results.json'),rank_and_top3_reproduced=True)
  del model,clip,c,payload;torch.cuda.empty_cache()
 atomic_write_json(a.output/'top5_examples.json',dict(status='complete',selection=SELECTION,rows=rows,evidence=evidence,image_sha256=image_sha,api_calls=0,optimizer_updates=0,new_image_encodings=0,elapsed_seconds=time.monotonic()-start,source_sha256=file_sha(__file__)))
 print(json.dumps({'status':'complete','examples':len(rows),'images':len(image_sha),'optimizer_updates':0,'api_calls':0}),flush=True)

if __name__=='__main__':main()
