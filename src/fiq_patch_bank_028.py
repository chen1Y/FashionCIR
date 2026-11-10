"""028 all-candidate projected patch shards, with original batch shapes preserved."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from fiq_cache_batch_geometry_026 import schedule
from train_attr_slot_pilot import load_clip
from expanded_attr_assets import base_path
from fashioniq_noise import file_sha
from experiment_storage import require_space,atomic_torch_save
from generate_structured_edits import atomic_write_json

CACHE=Path('/dev/shm/fiq_gate_patch_028');PARENT=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023');DATA=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/resized_image')


def build(root,cat):
    start=time.monotonic();plan=json.loads((root/'plan.json').read_text());meta=json.loads((root/'metadata'/(cat+'_val.json')).read_text());report=json.loads((PARENT/('assets_'+cat+'_val.json')).read_text());output=CACHE/cat
    if output.exists() or (root/('patches_'+cat+'.json')).exists():raise FileExistsError('028 patch extraction is never blindly restarted')
    require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',plan['estimated_patch_bytes']/2**30+2);output.mkdir(parents=True)
    if file_sha(report['cache_path'])!=report['cache_sha256'] or file_sha(base_path(cat))!=report['base_sha256']:raise ValueError('028 original global bank/model changed')
    old=torch.load(report['cache_path'],map_location='cpu',weights_only=True,mmap=True);ids=old['ids'];index={x:i for i,x in enumerate(ids)}
    if ids!=meta['gallery_ids'] or old['metadata_sha256']!=file_sha(root/'metadata'/(cat+'_val.json')):raise ValueError('028 original gallery order/metadata mismatch')
    if report['counts']['new_image_forwards']!=len(ids) or report['counts']['reused_global_images']!=0:raise ValueError('Original per-ID batch geometry is not established as fresh')
    existing=old['target_patches'];geometry=schedule(ids,ids,set(ids),set(existing));expected=set(ids);shards=[];buffer=[];buffer_ids=[];forward_count=0;max_delta=0.;checks=0
    def guard():
        require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',2)
        if (root/'cancel.request').exists():raise KeyboardInterrupt('028 cancelled')
        if time.monotonic()-start>3300:raise TimeoutError('028 patch job exceeds3300s')
        for n,h in plan['source_hashes'].items():
            if file_sha(Path(__file__).resolve().parents[1]/n)!=h:raise ValueError('028 sealed source changed')
    def flush():
        if not buffer:return
        path=output/('patch_'+str(len(shards)).zfill(4)+'.pt');positions=[index[i] for i in buffer_ids];atomic_torch_save(dict(ids=list(buffer_ids),positions=positions,patches=torch.stack(buffer),plan_sha256=file_sha(root/'plan.json'),category=cat),path);shards.append(dict(path=str(path),sha256=file_sha(path),ids=list(buffer_ids),positions=positions));buffer.clear();buffer_ids.clear()
    def append(i,value):
        if tuple(value.shape)!=(256,1024) or not torch.isfinite(value).all():raise ValueError('Invalid projected local patch geometry')
        buffer_ids.append(i);buffer.append(value.half().clone())
        if len(buffer)>=64:flush()
    try:
        for i in ids:
            if i in existing:append(i,existing[i])
        if geometry['batches']:
            torch.set_num_threads(4);clip,transform,_=load_clip(base_path(cat));clip.visual.output_tokens=True
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for batch_index,batch in enumerate(geometry['batches']):
                    guard();ims=[]
                    for i in batch['ids']:
                        path=DATA/(i+'.jpg')
                        if file_sha(path)!=meta['image_sha256'][i]:raise ValueError('028 source JPEG changed')
                        with Image.open(path) as im:ims.append(transform(im.convert('RGB')))
                    global_,tokens=clip.visual(torch.stack(ims).cuda());global_=F.normalize(global_.float(),dim=-1).cpu();forward_count+=len(batch['ids'])
                    for pos,i in enumerate(batch['ids']):
                        delta=float((global_[pos]-old['gallery'][index[i]]).abs().max());max_delta=max(max_delta,delta);checks+=1
                        if not torch.isfinite(global_[pos]).all() or delta>1e-4:
                            original=index[i]//8*8;failure=dict(image_id=i,jpeg_sha256=meta['image_sha256'][i],max_abs_delta=delta,threshold=1e-4,new_batch=batch['ids'],new_position=pos,original_batch=ids[original:original+8],original_position=index[i]%8,mode=batch['mode'],physical_forwards=forward_count,source_cache_sha256=report['cache_sha256'])
                            atomic_write_json(root/('patch_failure_'+cat+'.json'),failure);raise ValueError('028 original global reproduction failed; exact image and batch evidence preserved')
                    if clip.visual.final_ln_after_pool:tokens=clip.visual.ln_post(tokens)
                    patches=F.normalize((tokens@clip.visual.proj).float(),dim=-1).half().cpu()
                    for i,pos in zip(batch['real_ids'],batch['real_positions']):append(i,patches[pos])
                    if batch_index%8==0:atomic_write_json(root/('patch_progress_'+cat+'.json'),dict(batch=batch_index,physical_forwards=forward_count,planned=geometry['counts']['image_forwards'],completed_published=sum(len(s['ids']) for s in shards)+len(buffer),max_reused_global_delta=max_delta))
            clip.visual.output_tokens=False
        flush();published=[i for s in shards for i in s['ids']]
        if len(published)!=len(set(published)) or set(published)!=expected:raise ValueError('028 patches do not cover every gallery candidate exactly once')
        if forward_count!=geometry['counts']['image_forwards']:raise ValueError('028 physical-forward accounting mismatch')
        if file_sha(report['cache_path'])!=report['cache_sha256']:raise ValueError('028 original global bank changed')
        result=dict(status='complete',category=cat,gallery=len(ids),ids=ids,patches_per_image=256,dimension=1024,shards=shards,base_sha256=report['base_sha256'],original_global_cache_sha256=report['cache_sha256'],metadata_sha256=file_sha(root/'metadata'/(cat+'_val.json')),geometry_counts=geometry['counts'],physical_forwards=forward_count,global_checks=checks,max_reused_global_delta=max_delta,global_threshold_unchanged=1e-4,reused_patch_images=len(existing),all_candidate_images=True,gt_inserted=False,api_calls=0,fit_updates=0,seconds=time.monotonic()-start)
        atomic_write_json(root/('patches_'+cat+'.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ('ids','shards')}))
    except BaseException:
        flush();atomic_write_json(root/('patch_partial_'+cat+'.json'),dict(status='failed',shards=shards,physical_forwards=forward_count,max_reused_global_delta=max_delta));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',choices=('shirt','toptee'),required=True);a=p.parse_args();build(a.root,a.category)
