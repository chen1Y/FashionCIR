"""Identical022 model/loss/optimizer; whole-train metadata and compact output routing."""
import argparse,json
from pathlib import Path
import torch
import train_expanded_attr as engine
import native_expanded_attr_022 as native
from fiq_full_assets_024 import CACHE,base_path
from fiq_full_metadata_024 import CATS
from experiment_storage import require_space,atomic_torch_save
from fashioniq_noise import file_sha


def original_context(root,cat,phase):
    if phase!='train':raise ValueError('Original-only shortcut is train-only')
    m=json.loads((root/'metadata'/(cat+'_train.json')).read_text());p=CACHE/(cat+'_train_images.pt');d=torch.load(p,map_location='cpu',weights_only=True);report=json.loads((root/('assets_'+cat+'_train.json')).read_text())
    if file_sha(p)!=report['cache_sha256'] or file_sha(root/'metadata'/(cat+'_train.json'))!=d['metadata_sha256'] or file_sha(base_path(cat))!=d['base_sha256']:raise ValueError('Original-only context provenance mismatch')
    index={i:j for j,i in enumerate(d['ids'])};rows=[dict(r,target_attrs={k:'' for k in ('color','shape','detail','texture')},generation_status='not_used_by_original_arm') for r in m['records']];gallery=d['gallery'];ref=torch.stack([gallery[index[r['candidate']]] for r in rows]).cuda()
    c=dict(metadata=m,rows=rows,reference=ref,banks={},image_cache_sha256=report['cache_sha256'],text_sha256=m['caption_sha256'],teacher=torch.zeros(len(rows),4,1024,device='cuda'))
    for split in ('fit','dev'):
        ids=m['banks'][split];bankindex={i:j for j,i in enumerate(ids)};indices=m['offsets'][split];mask,labels=engine.legal_masks(m,split)
        c['banks'][split]=dict(ids=ids,gallery=torch.stack([gallery[index[i]] for i in ids]).cuda(),indices=indices,labels=labels.cuda(),references=torch.tensor([bankindex[rows[i]['candidate']] for i in indices],device='cuda'),mask=mask.cuda())
    return c


def configure(root,arm,official):
    plan=json.loads((root/'plan.json').read_text());weights=Path(plan['weight_directory']);weights.mkdir(parents=True,exist_ok=True)
    engine.CACHE=CACHE;engine.base_path=base_path;native.base_path=base_path
    if arm=='original' and not official:engine.load_context=original_context
    native.load_context=engine.load_context
    original_guard=engine.guard
    def guard(output,start):
        original_guard(output,start);require_space('/dev/shm',2)
    def save(state,path):
        actual=weights/path.relative_to(root)
        if path.exists() or actual.exists():raise FileExistsError('Never replace a full-study checkpoint')
        actual.parent.mkdir(parents=True,exist_ok=True)
        atomic_torch_save(state,actual);path.symlink_to(actual)
    engine.guard=guard;engine.atomic_torch_save=save


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',choices=CATS,required=True);p.add_argument('--arm',choices=('original','slots','dual'));p.add_argument('--smoke',action='store_true');p.add_argument('--official',action='store_true');a=p.parse_args();configure(a.root,a.arm,a.official)
    if a.official:
        for cat in CATS:
            for arm in ('original','slots','dual'):
                if not (a.root/'fits'/cat/arm/'fit_complete.json').exists():raise ValueError('All nine complete fits required')
        engine.official(a.root,a.category)
    elif a.smoke:native.smoke(a.root,a.category,a.arm)
    else:engine.fit(a.root,a.category,a.arm)
