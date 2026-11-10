"""029D paired training and 031 inference controls for FashionIQ spell lookup."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from attr_transfer_data import official_inputs
from fiq_native_caption_math_031 import metrics, transitions
from fiq_text_branch_math_027 import corrected_modification, textual_query
from run_029d_gate_refresh import ARM, SHIRT_TOPTEE, load_training_sources, restore, reuse_031_bank
from run_attr_transfer_029d import fit, training_context
from run_fiq_native_caption_031 import BASE, DATA, identity, read, sha, write


CATEGORIES = ('dress', 'shirt', 'toptee')
SPELLINGS = ('no_dict', 'official_dict')
TRAINED = ('fit_no_dict', 'fit_official_dict')
CACHE = BASE / 'runs/fiq-029d-gate-seeds-20261106/feature_cache'


def text_views(category, phase, rows):
    captions = DATA/'captions'
    source = read(captions/f'cap.{category}.{phase}.json')
    correction = read(captions/f'correction_dict_{category}.json')
    ref_captions = read(captions/f'image_captions_{category}_{phase}.json') if phase=='val' else None
    mods = {key:[] for key in SPELLINGS}
    native = {key:[] for key in SPELLINGS}
    for row in rows:
        original = source[row['source_index']]
        if any(row[k] != original[k] for k in ('candidate','target')):
            raise ValueError('Modification source/query ID mismatch')
        for key, mapping in zip(SPELLINGS, ({}, correction)):
            mod = corrected_modification(original['captions'], mapping)
            mods[key].append(mod)
            if ref_captions is not None:
                native[key].append(textual_query(ref_captions[row['candidate']], mod))
    return mods, native


def text_manifest(category):
    names = [f'cap.{category}.train.json', f'cap.{category}.val.json',
             f'correction_dict_{category}.json', f'image_captions_{category}_val.json']
    return {str(DATA/'captions'/name):sha(DATA/'captions'/name) for name in names}


def cached_bank(category, phase, binding, cuda=True):
    path = CACHE/f'{category}_{phase}.pt'
    info = read(path.with_suffix('.json'))
    key = hashlib.sha256(json.dumps(binding,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if info['key'] != key or sha(path) != info['sha256']:
        raise ValueError('Saved bank failed its data/checkpoint checksum')
    if not cuda:
        return info['sha256']
    import torch
    bank = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    return {key:value.cuda() if isinstance(value,torch.Tensor) else value for key,value in bank.items()}


def load_validation(category, base_sha):
    if category in SHIRT_TOPTEE:
        return reuse_031_bank(category)[0]
    rows, ids, hashes, _ = official_inputs(category)
    return cached_bank(category,'val',dict(base=base_sha,rows=rows,ids=ids,hashes=hashes))


def evaluate(root, category, model, bank, states, report):
    import torch
    import torch.nn.functional as F
    from expanded_attr_model import ranks_from_scores
    from fiq_fusion_029 import fuse_parts
    from train_clean_attr_alignment import plain_text

    rows = bank['rows']
    mods, native = text_views(category,'val',rows)
    if any(text != row['native_text'] for text,row in zip(native['official_dict'],rows)):
        raise ValueError('Official native caption input did not reproduce 031')
    gallery, written, reference = (bank[key] for key in ('gallery','written','reference'))
    n = len(rows)
    ranks, comparisons, native_m = {}, {}, {key:[] for key in SPELLINGS}

    def collect(name, query, j):
        stop = j + len(query)
        scores = query.float() @ gallery.T
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite official similarity')
        ranks.setdefault(name,[]).extend(ranks_from_scores(scores,bank['labels'][j:stop],
                                                           bank['references'][j:stop]).tolist())

    restore(model,states['dqu'])
    model.eval()
    native_head = copy.deepcopy(model.image_fusion).eval()
    with torch.inference_mode():
        for j in range(0,n,8):
            stop=min(j+8,n); w=written[j:stop]
            for spell in SPELLINGS:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    m=F.normalize(model.clip.encode_text(model.tokenizer(mods[spell][j:stop]).cuda()).float(),dim=-1)
                    cm=F.normalize(model.clip.encode_text(model.tokenizer(native[spell][j:stop]).cuda()).float(),dim=-1)
                    q,_=native_head(m,w)
                    qc,_=native_head(cm,w)
                native_m[spell].append(cm)
                collect(f'dqu/mod/{spell}',q,j)
                collect(f'dqu/caption/{spell}',qc,j)
    for variant in ('saved_029d', *TRAINED):
        restore(model,states[variant])
        model.eval()
        anchor = SimpleNamespace(attribute_gate=model.attribute_gate,text_mixture=model.text_mixture,image_fusion=native_head)
        with torch.inference_mode():
            for j in range(0,n,8):
                stop=min(j+8,n); w=written[j:stop]; v=reference[j:stop]
                batch=rows[j:stop]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    attrs=model([r['target_attrs'] for r in batch])
                    parts=dict(eos=attrs['f_global_text'],slots=attrs['f_attr_text'],valid=attrs['attr_valid'])
                    for spell in SPELLINGS:
                        m=plain_text(model,mods[spell][j:stop])
                        cm=plain_text(model,native[spell][j:stop])
                        original=fuse_parts(model,dict(parts,mod=m),v,w,ARM)
                        combined=fuse_parts(model,dict(parts,mod=cm),v,w,ARM)
                        q0,_=model.image_fusion(cm,w)
                        anchored=fuse_parts(anchor,dict(parts,mod=native_m[spell][j//8]),v,w,ARM)
                        for mode,q in (('mod',original['q']),('caption',combined['q']),
                                       ('caption_beta0',q0),('native_anchor',anchored['q'])):
                            with torch.autocast('cuda',enabled=False):
                                collect(f'{variant}/{mode}/{spell}',q,j)
                if j%128==0:
                    report('official_'+variant,stop,n)

    def pair(name, left, right):
        comparisons[name]={str(k):transitions(ranks[left],ranks[right],k) for k in (1,10,50)}

    for name in ranks:
        if name.endswith('/official_dict'):
            prefix=name.rsplit('/',1)[0]
            pair('eval_dictionary/'+prefix,name,prefix+'/no_dict')
    for mode in ('mod','caption','caption_beta0','native_anchor'):
        for spelling in SPELLINGS:
            pair(f'train_dictionary/{mode}/{spelling}',f'fit_official_dict/{mode}/{spelling}',
                 f'fit_no_dict/{mode}/{spelling}')
        pair('train_and_eval_dictionary/'+mode,f'fit_official_dict/{mode}/official_dict',f'fit_no_dict/{mode}/no_dict')
    write(root/'official_results.json',dict(category=category,queries=n,gallery=len(gallery),
          dictionary_changed_queries=sum(a!=b for a,b in zip(mods['no_dict'],mods['official_dict'])),
          rows={name:dict(metrics(rr),ranks=rr) for name,rr in ranks.items()},comparisons=comparisons))


def run(root, category, seed):
    import torch
    from fiq_fusion_029 import FusionText029
    from train_attr_slot_pilot import load_clip

    status = root/'state.json'
    if status.exists():
        raise FileExistsError(status)
    started = time.monotonic()
    state = dict(status='running',category=category,seed=seed,phase='loading',identity=identity(),api_calls=0)
    write(status,state)

    def progress(phase, completed, total):
        if time.monotonic()-started > 3600:
            raise TimeoutError('One-hour task limit')
        if (root.parent/'cancel.request').exists():
            raise InterruptedError('Queue cancelled')
        state.update(phase=phase,completed=completed,total=total)
        write(status,state)

    try:
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(.4)
        if text_manifest(category) != read(root.parent/'plan.json')['text_manifest'][category]:
            raise ValueError('Sealed correction text sources changed')
        meta, existing, existing_sha, base, base_sha = load_training_sources(category)
        if sha(existing)!=existing_sha or sha(base)!=base_sha:
            raise ValueError('Checkpoint provenance mismatch')
        bank=cached_bank(category,'train',dict(base=base_sha,metadata=meta))
        context=training_context(bank,meta)
        del bank
        mods,_=text_views(category,'train',meta['records'])
        clip,_,tokenizer=load_clip(base)
        model=FusionText029(clip,tokenizer,2).cuda()
        model.image_fusion.from_dqu(base)
        model.configure_arm(ARM)
        states=dict(dqu=model.delta_state(),saved_029d=torch.load(existing,map_location='cpu',weights_only=True)['state'])
        for arm, spelling in zip(TRAINED,SPELLINGS):
            restore(model,states['dqu'])
            torch.manual_seed(seed)
            folder=root/arm
            folder.mkdir()
            text_rows=[dict(row,modification_text=mod) for row,mod in zip(meta['records'],mods[spelling])]
            branch=dict(context,rows=text_rows)
            report=lambda phase,done,total:progress(arm+'_'+phase,done,total)
            report('smoke',0,1)
            choice=fit(folder,model,branch,report,seed=seed)
            path=folder/choice['weight']
            if sha(path)!=choice['sha256']:
                raise ValueError('Selected epoch weight SHA changed')
            states[arm]=torch.load(path,map_location='cpu',weights_only=True)['state']
            model.zero_grad(set_to_none=True)
            del branch,text_rows
            torch.cuda.empty_cache()
        del context
        progress('load_official_bank',0,1)
        bank=load_validation(category,base_sha)
        evaluate(root,category,model,bank,states,progress)
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
