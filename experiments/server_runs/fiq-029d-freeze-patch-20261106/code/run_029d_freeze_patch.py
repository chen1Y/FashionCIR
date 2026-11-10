"""Matched 029D text freezing and independent 028-style patch scoring."""

import argparse
import hashlib
import io
import time
from pathlib import Path

from attr_transfer_data import official_inputs
from fiq_native_caption_math_031 import metrics, transitions
from run_029d_gate_refresh import ARM, SHIRT_TOPTEE, inputs_029, load_training_sources, restore
from run_029d_spelling_pair import cached_bank, load_validation
from run_attr_transfer_029d import encode_bank, fit, load_dataset, training_context
from run_fiq_native_caption_031 import identity, read, sha, write


CATEGORIES = ('dress', 'shoes', 'shirt', 'toptee')
ARMS = ('last2', 'frozen_text')
WEIGHTS = (0.0, 0.1, 0.2)


def effective_parameters(model):
    clip = sum(p.numel() for p in model.clip.parameters() if p.requires_grad)
    token = model.embedding.weight.shape[1] * 4
    modules = {'attribute_gate':model.attribute_gate,'text_gate':model.text_mixture,'image_fusion':model.image_fusion}
    result = {name:sum(p.numel() for p in mod.parameters() if p.requires_grad) for name,mod in modules.items()}
    result.update(text_transformer=clip,added_tokens=token)
    result['total'] = sum(result.values())
    return result


def ranks_for(scores, bank):
    import torch
    from expanded_attr_model import ranks_from_scores

    scores = scores.cpu()
    targets,refs = bank['labels'].cpu(),bank['references'].cpu()
    valid = targets!=refs
    result = torch.full_like(targets,scores.shape[1])
    result[valid] = ranks_from_scores(scores[valid],targets[valid],refs[valid])
    return result.tolist()


def eval_global(model, bank, use_dqu, progress):
    import torch
    import torch.nn.functional as F
    from fiq_fusion_029 import encode_parts, fuse_parts
    from train_clean_attr_alignment import plain_text

    model.eval()
    rows = bank['rows']
    batches = {'native':[]} if use_dqu else {'raw':[],'native':[]}
    slots, valid = [], []
    with torch.inference_mode():
        for j in range(0,len(rows),8):
            batch = rows[j:j+8]
            stop = j+len(batch)
            w,v = bank['written'][j:stop],bank['reference'][j:stop]
            native = [row['native_text'] for row in batch]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if use_dqu:
                    m = F.normalize(model.clip.encode_text(model.tokenizer(native).cuda()).float(),dim=-1)
                    q,_ = model.image_fusion(m,w)
                    queries = {'native':q}
                else:
                    parts = encode_parts(model,batch,ARM)
                    raw = fuse_parts(model,parts,v,w,ARM)
                    cm = plain_text(model,native)
                    combined = fuse_parts(model,dict(parts,mod=cm),v,w,ARM)
                    queries = {'raw':raw['q'],'native':combined['q']}
                    slots.append(parts['slots'].float())
                    valid.append(parts['valid'])
            for name,q in queries.items():
                batches[name].append((q.float() @ bank['gallery'].T).cpu())
            if j%128==0:
                progress(stop,len(rows))
    result = {name:torch.cat(value) for name,value in batches.items()}
    if any(not torch.isfinite(x).all() for x in result.values()):
        raise FloatingPointError('Nonfinite global score')
    return result,(torch.cat(slots),torch.cat(valid)) if slots else None


def patch_scores(clip, transform, category, bank, slots, valid, progress):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from fiq_gate_patch_model_028 import local_scores

    if category in SHIRT_TOPTEE:
        meta,_=inputs_029(category,'val')
        ids,hashes=meta['gallery_ids'],meta['image_sha256']
    else:
        _,ids,_,_=official_inputs(category)
        hashes=bank['image_sha256']
    if len(ids)!=len(bank['gallery']) or len(set(ids))!=len(ids):
        raise ValueError('Patch/gallery geometry mismatch')
    path_for,_=load_dataset(category,transform)
    result=torch.full((len(slots),len(ids)),torch.nan)
    max_diff=0.0
    divisor=valid.sum(1).clamp_min(1)
    clip.eval()
    with torch.inference_mode():
        for begin in range(0,len(ids),64):
            finish=min(begin+64,len(ids)); projected=[]
            for j in range(begin,finish,8):
                names=ids[j:min(j+8,finish)]
                images=[]
                for name in names:
                    blob=path_for(name).read_bytes()
                    if hashlib.sha256(blob).hexdigest()!=hashes[name]:
                        raise ValueError(f'Gallery JPEG changed: {name}')
                    with Image.open(io.BytesIO(blob)) as image:
                        images.append(transform(image.convert('RGB')))
                clip.visual.output_tokens=True
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    global_,patch=clip.visual(torch.stack(images).cuda())
                    if clip.visual.final_ln_after_pool:
                        patch=clip.visual.ln_post(patch)
                    patch=F.normalize((patch @ clip.visual.proj).float(),dim=-1).half()
                clip.visual.output_tokens=False
                if tuple(patch.shape[1:])!=(256,1024) or not torch.isfinite(patch).all():
                    raise ValueError('Projected patch geometry or finite check failed')
                diff=(F.normalize(global_.float(),dim=-1)-bank['gallery'][j:j+len(names)]).abs().amax(1)
                peak=float(diff.max())
                max_diff=max(max_diff,peak)
                if not torch.isfinite(diff).all() or peak>1e-4:
                    index=int(diff.argmax()); name=names[index]
                    raise ValueError(f'Patch/global reproduction failed for {name}: {peak:g} in batch {names}')
                projected.append(patch)
            patches=torch.cat(projected)
            for j in range(0,len(slots),64):
                stop=min(j+64,len(slots))
                local,_,_=local_scores(slots[j:stop],valid[j:stop],patches)
                result[j:stop,begin:finish]=(local/divisor[j:stop,None]).cpu()
            progress(finish,len(ids))
    if not torch.isfinite(result).all() or result[~valid.any(1).cpu()].count_nonzero():
        raise ValueError('Invalid patch scores or invalid-slot fallback')
    return result,dict(images=len(ids),patches_per_image=256,global_max_abs=max_diff,
                       all_slots_invalid=int((~valid.any(1)).sum()))


def run(root, category, seed):
    import cv2
    import torch
    from fiq_fusion_029 import FusionText029
    from train_attr_slot_pilot import load_clip

    status=root/'state.json'
    if status.exists():
        raise FileExistsError(status)
    started=time.monotonic()
    state=dict(status='running',phase='loading',category=category,seed=seed,identity=identity(),api_calls=0)
    write(status,state)

    def progress(phase,completed,total):
        if time.monotonic()-started>3600:
            raise TimeoutError('One-hour category limit')
        if (root.parent/'cancel.request').exists():
            raise InterruptedError('Queue cancelled')
        state.update(phase=phase,completed=completed,total=total)
        write(status,state)

    try:
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(.4)
        meta,old,old_sha,base,base_sha=load_training_sources(category)
        if sha(old)!=old_sha or sha(base)!=base_sha:
            raise ValueError('Base or selected 029D checkpoint checksum mismatch')
        bank=cached_bank(category,'train',dict(base=base_sha,metadata=meta))
        ctx=training_context(bank,meta)
        del bank
        clip,transform,tokenizer=load_clip(base)
        model=FusionText029(clip,tokenizer,2).cuda()
        model.image_fusion.from_dqu(base)
        model.configure_arm(ARM)
        init=model.delta_state()
        states={'native_dqu':init,'saved_029d':torch.load(old,map_location='cpu',weights_only=True)['state']}
        params={}
        for arm in ARMS:
            restore(model,init)
            if arm=='frozen_text':
                model.clip.transformer.requires_grad_(False)
            params[arm]=effective_parameters(model)
            if arm=='frozen_text' and params[arm]['text_transformer']!=0:
                raise ValueError('Transformer freeze did not take effect')
            torch.manual_seed(seed)
            folder=root/arm
            folder.mkdir()
            choice=fit(folder,model,ctx,lambda stage,done,total:progress(arm+'_'+stage,done,total),seed=seed)
            path=folder/choice['weight']
            if sha(path)!=choice['sha256']:
                raise ValueError('Selected checkpoint checksum changed')
            states[arm]=torch.load(path,map_location='cpu',weights_only=True)['state']
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        del ctx
        bank=load_validation(category,base_sha)
        ranks, patch_info = {}, None
        for name in ('native_dqu',*ARMS,'saved_029d'):
            restore(model,init)
            model.load_delta(states[name])
            scores,attrs=eval_global(model,bank,name=='native_dqu',lambda done,total:progress('eval_'+name,done,total))
            for text_mode,value in scores.items():
                ranks[f'{name}/{text_mode}']=ranks_for(value,bank)
            if name=='saved_029d':
                local,patch_info=patch_scores(clip,transform,category,bank,*attrs,
                                               lambda done,total:progress('patch_full_gallery',done,total))
                for text_mode,value in scores.items():
                    for weight in WEIGHTS[1:]:
                        ranks[f'saved_029d/{text_mode}/local{weight}']=ranks_for(value+weight*local,bank)
        comparisons={}
        for mode in ('raw','native'):
            comparisons['frozen_minus_last2/'+mode]={str(k):transitions(ranks['frozen_text/'+mode],ranks['last2/'+mode],k) for k in (1,10,50)}
            for weight in WEIGHTS[1:]:
                name=f'saved_029d/{mode}/local{weight}'
                comparisons['patch_minus_saved/'+mode+f'/{weight}']={str(k):transitions(ranks[name],ranks['saved_029d/'+mode],k) for k in (1,10,50)}
        write(root/'official_results.json',dict(category=category,queries=len(bank['rows']),gallery=len(bank['gallery']),
              rows={name:dict(metrics(rr),ranks=rr) for name,rr in ranks.items()},comparisons=comparisons,
              params=params,patch=patch_info,patch_checkpoint_sha256=old_sha,patch_fit_updates=0))
        state.update(status='complete',phase='finished',completed=len(bank['rows']),total=len(bank['rows']),
                     seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/(1<<30))
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(status,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--category',choices=CATEGORIES,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    run(args.root,args.category,args.seed)
