"""Full024 compact frozen image banks with strict023 cache reuse."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from fiq_full_metadata_024 import OLD,PRIMARY,DATA,BASES,CATS
from train_attr_slot_pilot import load_clip
from fashioniq_noise import file_sha
from experiment_storage import require_space,atomic_torch_save
from generate_structured_edits import atomic_write_json

CACHE=Path('/dev/shm/fiq_full_024')

def base_path(cat):return BASES[cat]

def reuse018(cat,sha,phase):
 features={};patches={};evidence={};parent=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023')
 for source_phase in (phase,):
  report=parent/('assets_'+cat+'_'+source_phase+'.json')
  if not report.exists():continue
  info=json.loads(report.read_text());path=Path(info['cache_path'])
  if file_sha(path)!=info['cache_sha256']:raise ValueError('Validation feature source changed')
  d=torch.load(path,map_location='cpu',weights_only=True)
  if d['base_sha256']!=sha:raise ValueError('Different original CLIP bank')
  source_meta=parent/'metadata'/(cat+'_'+source_phase+'.json')
  if file_sha(source_meta)!=d['metadata_sha256']:raise ValueError('Validation bank metadata changed')
  for i,image_id in enumerate(d['ids']):
   v=d['gallery'][i].clone()
   if image_id in features and not torch.equal(features[image_id],v):raise ValueError('Conflicting reusable image vector')
   features[image_id]=v
  for image_id,v in d['target_patches'].items():patches[image_id]=v.clone()
  evidence['source023_'+source_phase+'_sha256']=info['cache_sha256']
 return features,patches,evidence


def prepare_images(root,cat,phase):
 start=time.monotonic();require_space(OLD);require_space('/home/lcy',2);require_space('/dev/shm',2);CACHE.mkdir(exist_ok=True)
 destination=CACHE/(cat+'_'+phase+'_images.pt')
 if destination.exists():raise FileExistsError('No duplicate bank encoding')
 m=json.loads((root/'metadata'/(cat+'_'+phase+'.json')).read_text());base=base_path(cat);sha=file_sha(base)
 expected=m['base_sha256']
 if sha!=expected:raise ValueError('Original category CLIP changed')
 ids=list(dict.fromkeys(m['banks']['fit']+m['banks']['dev'])) if phase=='train' else m['gallery_ids']
 needed_patches=set(r['target'] for r in m['records']) if phase=='train' else set()
 features,patches,reuse=reuse018(cat,sha,phase);requested=set(ids);features={i:v for i,v in features.items() if i in requested};patches={i:v for i,v in patches.items() if i in needed_patches}
 reused_globals=len(features);reused_patches=len(patches);todo=[i for i in ids if i not in features or (i in needed_patches and i not in patches)]
 max_reuse_error=0.
 if todo:
  clip,transform,_=load_clip(base);clip.visual.output_tokens=True
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for j in range(0,len(todo),8):
    require_space(OLD)
    if (root/'cancel.request').exists():raise KeyboardInterrupt('User stopped020')
    if time.monotonic()-start>3300:raise TimeoutError('020 visual cache exceeds1h')
    batch=todo[j:j+8];ims=[]
    for image_id in batch:
     p=DATA/'resized_image'/(image_id+'.jpg')
     if file_sha(p)!=m['image_sha256'][image_id]:raise ValueError('JPEG version changed')
     with Image.open(p) as im:ims.append(transform(im.convert('RGB')))
    global_,tokens=clip.visual(torch.stack(ims).cuda());global_=F.normalize(global_.float(),dim=-1).cpu()
    if clip.visual.final_ln_after_pool:tokens=clip.visual.ln_post(tokens)
    p_=F.normalize((tokens@clip.visual.proj).float(),dim=-1).half().cpu()
    for k,image_id in enumerate(batch):
     if image_id in features:
      delta=float((features[image_id]-global_[k]).abs().max());max_reuse_error=max(max_reuse_error,delta)
      if delta>1e-4:raise ValueError('Reused global did not reproduce within1e-4')
     else:features[image_id]=global_[k]
     if image_id in needed_patches:patches[image_id]=p_[k].clone()
    if j%800==0:atomic_write_json(root/('assets_'+cat+'_'+phase+'_progress.json'),dict(encoded=j+len(batch),pending_total=len(todo),reused_globals=reused_globals))
  clip.visual.output_tokens=False
 gallery=torch.stack([features[i] for i in ids])
 if not torch.isfinite(gallery).all():raise FloatingPointError('Nonfinite gallery')
 saved=dict(ids=ids,gallery=gallery,target_patches=patches,base_sha256=sha,metadata_sha256=file_sha(root/'metadata'/(cat+'_'+phase+'.json')),reused=reuse,
  counts=dict(gallery=len(ids),new_image_forwards=len(todo),reused_global_images=reused_globals,reused_patch_images=reused_patches,max_reused_global_error=max_reuse_error),seconds=time.monotonic()-start)
 atomic_torch_save(saved,destination);atomic_write_json(root/('assets_'+cat+'_'+phase+'.json'),dict(status='complete',cache_path=str(destination),cache_sha256=file_sha(destination),**{k:v for k,v in saved.items() if k in ('base_sha256','metadata_sha256','counts','seconds')}))
 print(json.dumps({'status':'complete','category':cat,'phase':phase,'counts':saved['counts']}),flush=True)

def prepare_teacher(root,cat):
 p=CACHE/(cat+'_train_images.pt');image=torch.load(p,map_location='cpu',weights_only=True);m=json.loads((root/'metadata'/(cat+'_train.json')).read_text());generated=json.loads((root/'texts'/(cat+'_train.json')).read_text());rows=m['records'];samples=generated['samples']
 if len(rows)!=len(samples):raise ValueError('Never drop a generated failure')
 for row,s in zip(rows,samples):
  if s['source_index']!=row['source_index'] or s['reference']!=row['candidate'] or s['instruction']!=row['modification_text']:raise ValueError('Text/query binding changed')
 clip,_,tok=load_clip(base_path(cat));attrs=[s['target_attrs'] for s in samples]
 texts=[a[k] or 'unspecified' for a in attrs for k in ('color','shape','detail','texture')];out=[]
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for j in range(0,len(texts),32):out.append(F.normalize(clip.encode_text(tok(texts[j:j+32]).cuda()).float(),dim=-1).cpu())
 feature=torch.cat(out).reshape(len(rows),4,-1);teachers=[]
 for i,row in enumerate(rows):
  patch=image['target_patches'][row['target']].float();teachers.append(F.normalize(torch.softmax(feature[i]@patch.T/.07,dim=-1)@patch,dim=-1))
 dest=CACHE/(cat+'_teacher.pt')
 if dest.exists():raise FileExistsError('Do not overwrite teacher cache')
 atomic_torch_save(dict(teachers=torch.stack(teachers),attributes=attrs,text_sha256=file_sha(root/'texts'/(cat+'_train.json')),image_cache_sha256=file_sha(p)),dest)
 atomic_write_json(root/('teacher_'+cat+'.json'),dict(status='complete',cache_path=str(dest),cache_sha256=file_sha(dest),expert_attribute_gt=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('train','val','teacher'),required=True);p.add_argument('--category',choices=(*CATS,'both'),required=True);a=p.parse_args();torch.set_num_threads(4)
 for cat in CATS if a.category=='both' else (a.category,):
  if a.phase=='teacher':prepare_teacher(a.root,cat)
  else:prepare_images(a.root,cat,a.phase)
