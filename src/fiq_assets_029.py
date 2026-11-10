"""Fresh029 frozen banks required after migration; never changes legacy caches."""
import argparse,json,time,gc
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from expanded_attr_assets import base_path
from train_attr_slot_pilot import load_clip
from dquDataset import FashionIQ
from experiment_storage import require_space,atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

CACHE=Path('/dev/shm/fiq_fusion_029');DATA=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ');PARENT=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023')


def guard(root,start):
    require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',2)
    if (root/'cancel.request').exists():raise KeyboardInterrupt('029 cancelled')
    if time.monotonic()-start>3500:raise TimeoutError('029 small job timeout')


def build(root,cat,phase):
    start=time.monotonic();guard(root,start);CACHE.mkdir(exist_ok=True);path=CACHE/(cat+'_'+phase+'.pt')
    if path.exists():raise FileExistsError('Inspect existing029 bank instead of rebuilding')
    meta=json.loads((root/'metadata'/(cat+'_'+phase+'.json')).read_text());samples=json.loads((root/'texts'/(cat+'_'+phase+'.json')).read_text())['samples'];rows=meta['records'];ids=list(dict.fromkeys(meta['banks']['fit']+meta['banks']['dev'])) if phase=='train' else meta['gallery_ids'];patch_ids={r['candidate'] for r in rows}|({r['target'] for r in rows} if phase=='train' else set());features=[];patches={};torch.set_num_threads(4)
    if len(samples)!=len(rows):raise ValueError('029 no query dropping')
    clip,transform,tok=load_clip(base_path(cat));clip.visual.output_tokens=True
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for j in range(0,len(ids),8):
            guard(root,start);batch=ids[j:j+8];ims=[]
            for i in batch:
                p=DATA/'resized_image'/(i+'.jpg')
                if file_sha(p)!=meta['image_sha256'][i]:raise ValueError('029 source JPEG changed')
                with Image.open(p) as im:ims.append(transform(im.convert('RGB')))
            glob,pt=clip.visual(torch.stack(ims).cuda());features.append(F.normalize(glob.float(),dim=-1).cpu())
            if any(i in patch_ids for i in batch):
                if clip.visual.final_ln_after_pool:pt=clip.visual.ln_post(pt)
                projected=F.normalize((pt@clip.visual.proj).float(),dim=-1).half().cpu()
                for k,i in enumerate(batch):
                    if i in patch_ids:patches[i]=projected[k].clone()
            if j%512==0:atomic_write_json(root/('asset_progress_'+cat+'_'+phase+'.json'),dict(stage='gallery',done=j+len(batch),total=len(ids)))
        clip.visual.output_tokens=False;gallery=torch.cat(features);index={i:j for j,i in enumerate(ids)};reference=torch.stack([gallery[index[r['candidate']]] for r in rows]);fake=object.__new__(FashionIQ);fake.category=cat;fake.image_dir=DATA/'resized_image';fake.transform=(transform,transform);keypath=DATA/'captions'/('keywords_in_mods_'+cat+'.json');fake.key_words=json.loads(keypath.read_text());written=[];keywords=[]
        for j in range(0,len(rows),8):
            guard(root,start);batch=rows[j:j+8];ims=[]
            for r in batch:
                im,_=fake.get_written_img(r['candidate'],r['target'],stage=0);ims.append(im);keywords.append(fake.key_words[r['candidate']+'_'+r['target']][-1])
            written.append(F.normalize(clip.encode_image(torch.stack(ims).cuda()).float(),dim=-1).cpu())
            if j%256==0:atomic_write_json(root/('asset_progress_'+cat+'_'+phase+'.json'),dict(stage='native_dqu_written_queries',done=j+len(batch),total=len(rows)))
        teacher=None
        if phase=='train':
            phrases=[s['target_attrs'][k] or 'unspecified' for s in samples for k in ('color','shape','detail','texture')];f=[]
            for j in range(0,len(phrases),32):f.append(F.normalize(clip.encode_text(tok(phrases[j:j+32]).cuda()).float(),dim=-1).cpu())
            f=torch.cat(f).reshape(len(rows),4,-1);teacher=torch.stack([F.normalize(torch.softmax(f[i]@patches[r['target']].float().T/.07,dim=-1)@patches[r['target']].float(),dim=-1) for i,r in enumerate(rows)])
    if set(patches)!=patch_ids or not torch.isfinite(gallery).all():raise ValueError('029 missing/nonfinite bank')
    saved=dict(ids=ids,gallery=gallery,reference=reference,written=torch.cat(written),patches=patches,teacher=teacher,keywords=keywords,base_sha256=file_sha(base_path(cat)),metadata_sha256=file_sha(root/'metadata'/(cat+'_'+phase+'.json')),text_sha256=file_sha(root/'texts'/(cat+'_'+phase+'.json')),keywords_sha256=file_sha(keypath),api_calls=0,visual_trainable=False)
    atomic_torch_save(saved,path);atomic_write_json(root/('assets_'+cat+'_'+phase+'.json'),dict(status='complete',path=str(path),sha256=file_sha(path),gallery=len(ids),queries=len(rows),patch_images=len(patches),image_forwards=len(ids)+len(rows),cached_old_features_existed=False,seconds=time.monotonic()-start,**{k:v for k,v in saved.items() if k.endswith('sha256')}));print(json.dumps(dict(status='complete',cat=cat,phase=phase,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',required=True);p.add_argument('--phase',choices=('train','val'),required=True);a=p.parse_args();build(a.root,a.category,a.phase)
