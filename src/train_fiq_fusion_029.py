"""029 full legal-bank matched training and complete official retrieval."""
import argparse,gc,json,math,time
from pathlib import Path
import torch
from fiq_fusion_029 import FusionText029,ARMS,encode_parts,fuse_parts,objective
from fiq_assets_029 import CACHE,guard
from expanded_attr_model import legal_masks,ranks_from_scores,metrics
from expanded_attr_assets import base_path
from train_attr_slot_pilot import load_clip
from train_attr_pair_alignment import frozen_digest
from experiment_storage import atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json


def context(root,cat,phase):
    meta=json.loads((root/'metadata'/(cat+'_'+phase+'.json')).read_text());text=json.loads((root/'texts'/(cat+'_'+phase+'.json')).read_text());report=json.loads((root/('assets_'+cat+'_'+phase+'.json')).read_text());p=Path(report['path'])
    if file_sha(p)!=report['sha256']:raise ValueError('029 image cache changed')
    bank=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    if bank['metadata_sha256']!=file_sha(root/'metadata'/(cat+'_'+phase+'.json')) or bank['text_sha256']!=file_sha(root/'texts'/(cat+'_'+phase+'.json')) or bank['base_sha256']!=file_sha(base_path(cat)):raise ValueError('029 source binding changed')
    rows=[]
    for r,s in zip(meta['records'],text['samples']):
        if r['source_index']!=s['source_index'] or r['candidate']!=s['reference'] or r['modification_text']!=s['instruction']:raise ValueError('029 text/query mismatch')
        rows.append(dict(r,target_attrs=s['target_attrs']))
    if len(rows)!=len(meta['records']):raise ValueError('029 dropped query')
    out=dict(meta=meta,rows=rows,reference=bank['reference'].cuda(),written=bank['written'].cuda(),teacher=bank['teacher'].cuda() if phase=='train' else None,banks={},asset_sha256=report['sha256']);index={i:j for j,i in enumerate(bank['ids'])}
    for split in (('fit','dev') if phase=='train' else ('val',)):
        ids=meta['banks'][split] if phase=='train' else meta['gallery_ids'];ix=meta['offsets'][split] if phase=='train' else list(range(len(rows)));mapping={i:j for j,i in enumerate(ids)};mask,labels=legal_masks(meta,split) if phase=='train' else (None,torch.tensor([mapping[r['target']] for r in rows]));out['banks'][split]=dict(ids=ids,indices=ix,gallery=bank['gallery'][[index[i] for i in ids]].cuda(),labels=labels.cuda(),references=torch.tensor([mapping[rows[i]['candidate']] for i in ix],device='cuda'),mask=mask.cuda() if mask is not None else None)
    return out


def make_model(cat,arm):
    clip,_,tok=load_clip(base_path(cat));model=FusionText029(clip,tok,2).cuda();model.image_fusion.from_dqu(base_path(cat));model.configure_arm(arm);return model


def evaluate(model,c,split,arm,include_parts=False):
    model.eval();b=c['banks'][split];ranks=[];nce=[];alphas=[];betas=[];weights=[];saved=[]
    with torch.no_grad():
        for j in range(0,len(b['indices']),8):
            chosen=b['indices'][j:j+8];ix=torch.tensor(chosen,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):parts=encode_parts(model,[c['rows'][i] for i in chosen],arm);out=fuse_parts(model,parts,c['reference'][ix],c['written'][ix],arm)
            scores=out['q']@b['gallery'].T;labels=b['labels'][j:j+len(ix)];ref=b['references'][j:j+len(ix)];ranks+=ranks_from_scores(scores,labels,ref).tolist();logits=scores.clone()
            if b['mask'] is None:logits[torch.arange(len(ix),device='cuda'),ref]=-torch.inf
            else:logits=logits.masked_fill(~b['mask'][j:j+len(ix)],-torch.inf)
            nce+=torch.nn.functional.cross_entropy(logits/.07,labels,reduction='none').tolist();alphas+=out['alpha'].tolist();betas+=out['beta'].tolist()
            if out['slot_weights'] is not None:weights+=out['slot_weights'].tolist()
            if include_parts:saved.append({k:v.detach().cpu() if v is not None else None for k,v in parts.items()})
    value=dict(**metrics(ranks),retrieval_nce=sum(nce)/len(nce),image_text_alpha=alphas,attribute_text_beta=betas,slot_weights=weights)
    if include_parts:value['parts']={k:torch.cat([s[k] for s in saved]) if saved[0][k] is not None else None for k in saved[0]}
    return value


def save_delta(root,cat,arm,epoch,value):
    target=Path(json.loads((root/'plan.json').read_text())['weight_directory'])/cat/arm/('epoch_'+str(epoch)+'_delta.pt');target.parent.mkdir(parents=True,exist_ok=True);link=root/'fits'/cat/arm/target.name
    if target.exists() or link.is_symlink():raise FileExistsError('029 checkpoint exists')
    atomic_torch_save(value,target);link.symlink_to(target);return str(link),file_sha(target)


def fit(root,cat,arm):
    start=time.monotonic();torch.manual_seed(42);torch.set_num_threads(4);guard(root,start);output=root/'fits'/cat/arm
    if output.exists():raise FileExistsError('029 no blind fit restart')
    output.mkdir(parents=True);c=context(root,cat,'train');model=make_model(cat,arm);frozen=frozen_digest(model.clip);optimizer=torch.optim.AdamW(model.parameter_groups());params=[p for g in optimizer.param_groups for p in g['params']];bank=c['banks']['fit'];n=len(bank['indices']);exposure=torch.zeros(n,dtype=torch.long);rng=torch.Generator().manual_seed(42);history=[];steps=0
    # Real full-bank backward, no optimizer update, so no exposure or momentum leak.
    model.eval();gpu=torch.arange(8,device='cuda');ix=torch.tensor(bank['indices'][:8],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):parts=encode_parts(model,[c['rows'][i] for i in ix.tolist()],arm);out=fuse_parts(model,parts,c['reference'][ix],c['written'][ix],arm);ls=objective(out,bank['gallery'],bank['labels'][gpu],bank['mask'][gpu],c['teacher'][ix],arm)
    ls['total'].backward()
    if model.embedding.weight.grad[:49408].count_nonzero() or any(p.grad is not None for p in model.clip.visual.parameters()):raise ValueError('029 hook/ViT smoke failed')
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('029 smoke gradient nonfinite')
    atomic_write_json(output/'smoke.json',dict(status='passed',batch=8,optimizer_updates=0,old_vocabulary_gradient_zero=True,vit_gradients=False,fusion_gradient_norm=float(model.image_fusion.combiner_fc[0].weight.grad.norm()),loss={k:float(v.detach()) for k,v in ls.items()}));optimizer.zero_grad(set_to_none=True)
    def record(epoch):
        value=evaluate(model,c,'dev',arm);path,sha=save_delta(root,cat,arm,epoch,dict(state=model.delta_state(),category=cat,arm=arm,epoch=epoch,asset_sha256=c['asset_sha256']));history.append(dict(epoch=epoch,steps=steps,metrics=value,weight_path=path,weight_sha256=sha));selected=max(history,key=lambda h:((h['metrics']['r10']+h['metrics']['r50'])/2,-h['metrics']['retrieval_nce'],-h['epoch']))['epoch'];atomic_write_json(output/'progress.json',dict(status='training',updates=steps,history=history,selected_epoch=selected,exposure=exposure.tolist()))
    record(0)
    for epoch in range(1,4):
        model.train();order=torch.randperm(n,generator=rng)
        for j in range(0,n,8):
            guard(root,start);local=order[j:j+8];exposure[local]+=1;gpu=local.cuda();ix=torch.tensor([bank['indices'][i] for i in local.tolist()],device='cuda');optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):parts=encode_parts(model,[c['rows'][i] for i in ix.tolist()],arm);out=fuse_parts(model,parts,c['reference'][ix],c['written'][ix],arm);ls=objective(out,bank['gallery'],bank['labels'][gpu],bank['mask'][gpu],c['teacher'][ix],arm)
            if not torch.isfinite(ls['total']):raise FloatingPointError('029 nonfinite loss')
            ls['total'].backward()
            if model.embedding.weight.grad[:49408].count_nonzero():raise ValueError('029 old vocabulary gradient leak')
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise FloatingPointError('029 nonfinite gradient')
            optimizer.step();steps+=1
            if not torch.equal(model.embedding.weight[:49408],model.clip.token_embedding.weight):raise ValueError('029 old vocabulary drift')
            if steps%16==0:
                with (output/'training.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,step=steps,presentations=int(exposure.sum()),loss={k:float(v.detach()) for k,v in ls.items()}))+'\n')
        record(epoch)
    if exposure.tolist()!=[3]*n or frozen_digest(model.clip)!=frozen:raise ValueError('029 exposure/frozen SHA failed')
    chosen=max(history,key=lambda h:((h['metrics']['r10']+h['metrics']['r50'])/2,-h['metrics']['retrieval_nce'],-h['epoch']));result=dict(status='complete',category=cat,arm=arm,history=history,selected_epoch=chosen['epoch'],selected_weight=chosen['weight_path'],selected_weight_sha256=chosen['weight_sha256'],updates=steps,presentations=int(exposure.sum()),fit_queries=n,dev_queries=len(c['banks']['dev']['indices']),per_fit_row_exposures=exposure.tolist(),frozen_sha256_before=frozen,frozen_sha256_after=frozen,old_vocabulary_exact=True,official_validation_loaded=False,seconds=time.monotonic()-start)
    atomic_write_json(output/'results.json',result);atomic_write_json(output/'fit_complete.json',dict(status='complete',weight_sha256=chosen['weight_sha256']));print(json.dumps(dict(status='complete',cat=cat,arm=arm,epoch=chosen['epoch'],updates=steps)),flush=True)


def official(root,cat):
    for category in ('shirt','toptee'):
        for arm in ARMS:
            if not (root/'fits'/category/arm/'fit_complete.json').exists():raise ValueError('029 eight fits before official evaluation')
    start=time.monotonic();torch.set_num_threads(4);c=context(root,cat,'val');bank=c['banks']['val'];model=make_model(cat,'original_written');model.eval();rows={};meta=c['meta'];data=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/captions');captions=json.loads((data/('image_captions_'+cat+'_val.json')).read_text());original=json.loads((data/('cap.'+cat+'.val.json')).read_text());correction=json.loads((data/('correction_dict_'+cat+'.json')).read_text());from fiq_text_branch_math_027 import corrected_modification
    native=[]
    with torch.no_grad():
        for j in range(0,len(c['rows']),8):
            batch=c['rows'][j:j+8];texts=[captions[r['candidate']]+', but '+corrected_modification(original[r['source_index']]['captions'],correction) for r in batch]
            with torch.autocast('cuda',dtype=torch.bfloat16):text=torch.nn.functional.normalize(model.clip.encode_text(model.tokenizer(texts).cuda()).float(),dim=-1);q,_=model.image_fusion(text,c['written'][j:j+len(batch)])
            native+=ranks_from_scores(q@bank['gallery'].T,bank['labels'][j:j+len(batch)],bank['references'][j:j+len(batch)]).tolist()
    rows['frozen_native_dqu_same_bank']=metrics(native);rows['frozen_raw_mod_written']=evaluate(model,c,'val','original_written')
    for arm in ARMS:
        guard(root,start);report=json.loads((root/'fits'/cat/arm/'results.json').read_text())
        if file_sha(report['selected_weight'])!=report['selected_weight_sha256']:raise ValueError('029 selected SHA changed')
        model.load_delta(torch.load(report['selected_weight'],map_location='cpu',weights_only=True)['state']);rows[arm]=evaluate(model,c,'val',arm)
    atomic_write_json(root/('official_'+cat+'.json'),dict(status='complete',category=cat,queries=len(c['rows']),gallery=len(bank['ids']),rows=rows,precision='BF16 forwards, FP32 normalized scoring; contemporaneous nativeDQU baseline, not exact historical FP16 pipeline',reference_id_excluded=True,all_queries_retained=True,gt_inserted=False,seconds=time.monotonic()-start));print(json.dumps(dict(status='complete',cat=cat,metrics={n:{k:r[k] for k in ('r1','r10','r50')} for n,r in rows.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--official',action='store_true');a=p.parse_args();official(a.root,a.category) if a.official else fit(a.root,a.category,a.arm)
