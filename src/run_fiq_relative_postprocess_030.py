"""030-P zero-new-API semantic postprocessing and frozen-parent adaptation."""
import argparse,gc,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from fiq_relative_postprocess_030 import SLOTS,AXES,derive
from fiq_relative_adapter_030 import RelativeAdapter030,ARMS
from fiq_fusion_029 import fuse_parts,encode_parts
from train_fiq_fusion_029 import context,make_model
from fiq_assets_029 import CACHE as CACHE029,guard
from expanded_attr_model import ranks_from_scores,metrics
from train_clean_attr_alignment import plain_text
from fashioniq_noise import file_sha
from experiment_storage import atomic_torch_save
from generate_structured_edits import atomic_write_json

CACHE=Path('/dev/shm/fiq_relative_030_post');PARENT=Path('/data/8T1/lcy/CIRData/runs/fiq-fusion-029');DATA=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/captions')


def build(root,cat,phase):
    start=time.monotonic();guard(root,start);torch.set_num_threads(4);CACHE.mkdir(exist_ok=True);path=CACHE/(cat+'_'+phase+'.pt')
    if path.exists():raise FileExistsError('030-P existing query cache')
    if json.loads((PARENT/'state.json').read_text())['status']!='complete' or json.loads((PARENT/'verification.json').read_text())['status']!='verified':raise ValueError('030-P verified029 required')
    if phase=='val' and not (root/'fit_complete.json').exists():raise ValueError('030-P train fits before val')
    c=context(PARENT,cat,phase);report=json.loads((PARENT/'fits'/cat/'full_learned_written/results.json').read_text())
    if file_sha(report['selected_weight'])!=report['selected_weight_sha256']:raise ValueError('030-P parent selected SHA changed')
    state=torch.load(report['selected_weight'],map_location='cpu',weights_only=True)['state'];model=make_model(cat,'full_learned_written');model.load_delta(state);model.requires_grad_(False);model.eval();forbid=model.clip.visual.register_forward_pre_hook(lambda *x:(_ for _ in ()).throw(RuntimeError('030-P may not reencode images')))
    caption_path=DATA/('image_captions_'+cat+'_'+phase+'.json');captions=json.loads(caption_path.read_text());old=json.loads((PARENT/'texts'/(cat+'_'+phase+'.json')).read_text())['samples'];rows=c['rows'];sem=[derive(r['modification_text'],r['target_attrs'],captions.get(r['candidate'],''),s['status'],model.tokenizer.encode) for r,s in zip(rows,old)];saved=[];base_ranks={};parts_lists=[]
    with torch.no_grad():
        # Reproduce the parent's split-specific batch geometry, including tails.
        # Encoding all1221 rows sequentially would change the dev tail from B1.
        parts={};seen=set()
        for bank in c['banks'].values():
            for j in range(0,len(bank['indices']),8):
                guard(root,start);chosen=bank['indices'][j:j+8]
                if seen.intersection(chosen):raise ValueError('030-P duplicate split row')
                seen.update(chosen)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=encode_parts(model,[rows[i] for i in chosen],'full_learned_written')
                for k,v in out.items():
                    if k not in parts:parts[k]=torch.empty((len(rows),*v.shape[1:]),dtype=v.dtype)
                    parts[k][chosen]=v.cpu()
        if seen!=set(range(len(rows))):raise ValueError('030-P incomplete split feature coverage')
        garment='shirt' if cat=='shirt' else 'top';texts={}
        for axis,(negative,positive) in AXES.items():
            for side,pole in [('negative',negative),('positive',positive)]:texts[(axis,side)]=[f'a photo of a {garment} with {pole}',f'a {garment} with {pole}',f'the garment has {pole}']
        unique=set(p for ts in texts.values() for p in ts);source_keys={};neutral={}
        neutral_words={'color':'body color','shape':'sleeve and garment shape','detail':'construction details','texture':'pattern and material texture'}
        for k in SLOTS:neutral[k]=f'a photo of a {garment} showing {neutral_words[k]}';unique.add(neutral[k])
        for i,s in enumerate(sem):
            for k,p in s['source_phrases'].items():
                if p:source_keys[(i,k)]=f'a photo of a {garment} with {p}';unique.add(source_keys[(i,k)])
        unique=sorted(unique);encoded={}
        for j in range(0,len(unique),32):
            with torch.autocast('cuda',dtype=torch.bfloat16):value=plain_text(model,unique[j:j+32])
            encoded.update({s:v.cpu() for s,v in zip(unique[j:j+32],value)})
        axes={axis:F.normalize(torch.stack([encoded[p] for p in texts[(axis,'positive')]]).mean(0)-torch.stack([encoded[p] for p in texts[(axis,'negative')]]).mean(0),dim=-1) for axis in AXES};raw=torch.load(CACHE029/(cat+'_'+phase+'.pt'),map_location='cpu',weights_only=True,mmap=True);n=len(rows);anchors_all=torch.zeros(n,4,1024);anchors_source=torch.zeros_like(anchors_all);source=torch.zeros_like(anchors_all);directions=torch.zeros_like(anchors_all);active_all=torch.zeros(n,4,dtype=torch.bool);active_source=torch.zeros_like(active_all)
        for i,(r,s) in enumerate(zip(rows,sem)):
            patch=F.normalize(raw['patches'][r['candidate']].float(),dim=-1)
            for k,slot in enumerate(SLOTS):
                rel=[x for x in s['relations'] if x['slot']==slot]
                if not rel or not bool(parts['valid'][i,k]):continue
                active_all[i,k]=True;q=encoded[neutral[slot]];anchors_all[i,k]=F.normalize(torch.softmax(q@patch.T/.07,dim=-1)@patch,dim=-1);directions[i,k]=torch.stack([x['sign']*axes[x['axis']] for x in rel]).mean(0)
                if (i,slot) in source_keys:
                    active_source[i,k]=True;q=encoded[source_keys[(i,slot)]];source[i,k]=q;anchors_source[i,k]=F.normalize(torch.softmax(q@patch.T/.07,dim=-1)@patch,dim=-1)
        # Each alternative is independently encoded; never concatenate 'or' colors.
        all_h=[(i,h) for i,s in enumerate(sem) for h in s['hypotheses']];hyp={i:[] for i,s in enumerate(sem) if s['hypotheses']}
        for j in range(0,len(all_h),8):
            batch=all_h[j:j+8];attrs=[h for _,h in batch];padded=attrs+[attrs[0]]*(8-len(attrs))
            with torch.autocast('cuda',dtype=torch.bfloat16):o=model(padded)
            for p,(i,_) in enumerate(batch):hyp[i].append(dict(slots=o['f_attr_text'][p].cpu(),eos=o['f_global_text'][p].cpu(),valid=o['attr_valid'][p].cpu()))
    banks={name:{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in bank.items()} for name,bank in c['banks'].items()};payload=dict(parts=parts,reference=c['reference'].cpu(),written=c['written'].cpu(),anchors_all=anchors_all,anchors_source=anchors_source,source=source,directions=directions,active_all=active_all,active_source=active_source,banks=banks,hypotheses=hyp,semantics=sem,parent_state={name:{k:v.cpu() for k,v in state[name].items()} for name in ('attribute_gate','text_mixture','image_fusion')},parent_weight_sha256=report['selected_weight_sha256'],parent_asset_sha256=c['asset_sha256'],caption_sha256=file_sha(caption_path),axis_templates={str(k):v for k,v in texts.items()},axes=axes,metadata_records=c['meta']['records'])
    atomic_torch_save(payload,path);atomic_write_json(root/('data_'+cat+'_'+phase+'.json'),dict(status='complete',path=str(path),sha256=file_sha(path),queries=n,hypothesis_queries=len(hyp),active_axis_queries=int(active_all.any(1).sum()),source_supported_queries=int(active_source.any(1).sum()),parent_weight_sha256=report['selected_weight_sha256'],caption_sha256=file_sha(caption_path),failed_online_outputs_used=False,new_api_calls=0,new_image_forwards=0,seconds=time.monotonic()-start));atomic_write_json(root/('semantic_examples_'+cat+'_'+phase+'.json'),dict(samples=[dict(source_index=r['source_index'],reference=r['candidate'],instruction=r['modification_text'],**s) for r,s in zip(rows,sem)]));forbid.remove();print(json.dumps(dict(status='complete',category=cat,phase=phase,hypothesis_queries=len(hyp),active_axis_queries=int(active_all.any(1).sum()),source_queries=int(active_source.any(1).sum()))),flush=True)


def load(root,cat,phase):
    report=json.loads((root/('data_'+cat+'_'+phase+'.json')).read_text())
    if file_sha(report['path'])!=report['sha256']:raise ValueError('030-P query cache changed')
    return torch.load(report['path'],map_location='cpu',weights_only=True,mmap=True)


def subset(data,indices):
    ix=torch.tensor(indices,dtype=torch.long);device='cuda'
    return dict(parts={k:v[ix].to(device) for k,v in data['parts'].items()},reference=data['reference'][ix].to(device),written=data['written'][ix].to(device),anchors_all=data['anchors_all'][ix].to(device),anchors_source=data['anchors_source'][ix].to(device),source=data['source'][ix].to(device),axis=data['directions'][ix].to(device),active_all=data['active_all'][ix].to(device),active_source=data['active_source'][ix].to(device))


def predict(model,x):return model(x['parts'],x['reference'],x['written'],x['anchors_source'] if model.arm in ('anchor_source','target_minus_source') else x['anchors_all'],x['source'],x['axis'],x['active_all'],x['active_source'])


def evaluate(model,data,split,baseline=False):
    model.eval();b=data['banks'][split];gallery=b['gallery'].cuda();ranks=[];nce=[]
    with torch.no_grad():
        for j in range(0,len(b['indices']),8):
            chosen=b['indices'][j:j+8];x=subset(data,chosen)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=fuse_parts(model,x['parts'],x['reference'],x['written'],'full_learned_written') if baseline else predict(model,x)
            score=out['q']@gallery.T;labels=b['labels'][j:j+len(chosen)].cuda();ref=b['references'][j:j+len(chosen)].cuda();ranks+=ranks_from_scores(score,labels,ref).tolist();logits=score.clone()
            if b['mask'] is None:logits[torch.arange(len(chosen),device='cuda'),ref]=-torch.inf
            else:logits=logits.masked_fill(~b['mask'][j:j+len(chosen)].cuda(),-torch.inf)
            nce+=F.cross_entropy(logits/.07,labels,reduction='none').tolist()
    return dict(**metrics(ranks),retrieval_nce=sum(nce)/len(nce))


def fit(root,cat,arm):
    start=time.monotonic();guard(root,start);torch.manual_seed(42);torch.set_num_threads(4);data=load(root,cat,'train');model=RelativeAdapter030(data['parent_state'],arm).cuda();model.eval();output=root/'fits'/cat/arm
    if output.exists():raise FileExistsError('030-P no blind fit restart')
    output.mkdir(parents=True);b=data['banks']['fit'];gallery=b['gallery'].cuda();n=len(b['indices']);parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=.005,weight_decay=0.);rng=torch.Generator().manual_seed(42);exposure=torch.zeros(n,dtype=torch.long);history=[];steps=0;parent=json.loads((PARENT/'fits'/cat/'full_learned_written/results.json').read_text());expected=next(h['metrics']['ranks'] for h in parent['history'] if h['epoch']==parent['selected_epoch'])
    initial=evaluate(model,data,'dev')
    if initial['ranks']!=expected:raise ValueError('030-P zero initialization does not reproduce029 dev ranks')
    def record(epoch,value=None):
        value=value or evaluate(model,data,'dev');p=output/('epoch_'+str(epoch)+'.pt');atomic_torch_save(dict(blend=model.blend.detach().cpu(),gamma=model.gamma.detach().cpu(),arm=arm),p);history.append(dict(epoch=epoch,updates=steps,metrics=value,path=str(p),sha256=file_sha(p),blend=model.blend.tolist(),gamma=model.gamma.tolist()));atomic_write_json(output/'progress.json',dict(status='training',history=history,exposure=exposure.tolist(),updates=steps))
    record(0,initial)
    for epoch in range(1,4):
        order=torch.randperm(n,generator=rng)
        for j in range(0,n,8):
            guard(root,start);ix=order[j:j+8];exposure[ix]+=1;chosen=[b['indices'][i] for i in ix.tolist()];x=subset(data,chosen);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=predict(model,x)
            logits=(out['q']@gallery.T).masked_fill(~b['mask'][ix].cuda(),-torch.inf)/.07;ret=F.cross_entropy(logits,b['labels'][ix].cuda());loss=ret+.01*sum(p.square().sum() for p in parameters)
            if not torch.isfinite(loss):raise FloatingPointError('030-P nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise FloatingPointError('030-P nonfinite gradient')
            optimizer.step();model.project();steps+=1
        record(epoch)
    if exposure.tolist()!=[3]*n or any(p.grad is not None for module in (model.attribute_gate,model.text_mixture,model.image_fusion) for p in module.parameters()):raise ValueError('030-P exposure or frozen fusion failed')
    chosen=max(history,key=lambda h:((h['metrics']['r10']+h['metrics']['r50'])/2,-h['metrics']['retrieval_nce'],-h['epoch']));atomic_write_json(output/'results.json',dict(status='complete',arm=arm,category=cat,history=history,selected_epoch=chosen['epoch'],selected_weight=chosen['path'],selected_weight_sha256=chosen['sha256'],fit_queries=n,updates=steps,presentations=int(exposure.sum()),exposure=exposure.tolist(),parent_frozen=True,initial029_ranks_exact=True,seconds=time.monotonic()-start));print(json.dumps(dict(status='complete',cat=cat,arm=arm,selected_epoch=chosen['epoch'])),flush=True)


def enum_evaluate(model,data,split,shuffle=False):
    b=data['banks'][split];gallery=b['gallery'].cuda();ranks=[];indices=b['indices'];mapping={i:i for i in indices};groups={}
    if shuffle:
        for i in indices:
            h=len(data['hypotheses'].get(i,[]))
            if h:groups.setdefault(h,[]).append(i)
        rng=torch.Generator().manual_seed(42)
        for group in groups.values():
            perm=torch.randperm(len(group),generator=rng).tolist();mapping.update({i:group[k] for i,k in zip(group,perm)})
    with torch.no_grad():
        for j in range(0,len(indices),8):
            batch=indices[j:j+8];x=subset(data,batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):base=fuse_parts(model,x['parts'],x['reference'],x['written'],'full_learned_written')
            scores=base['q']@gallery.T
            for pos,i in enumerate(batch):
                hypotheses=data['hypotheses'].get(mapping[i],[])
                if not hypotheses:continue
                k=len(hypotheses);parts={key:x['parts'][key][pos:pos+1].expand(8,*x['parts'][key].shape[1:]).clone() for key in x['parts']}
                for h in range(8):
                    entry=hypotheses[min(h,k-1)]
                    for key in ('slots','eos','valid'):parts[key][h]=entry[key].cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):out=fuse_parts(model,parts,x['reference'][pos:pos+1].expand(8,-1),x['written'][pos:pos+1].expand(8,-1),'full_learned_written')
                s=out['q'][:k]@gallery.T;scores[pos]=.07*(torch.logsumexp(s/.07,0)-math.log(k))
            ranks+=ranks_from_scores(scores,b['labels'][j:j+len(batch)].cuda(),b['references'][j:j+len(batch)].cuda()).tolist()
    return dict(**metrics(ranks),hypothesis_queries=sum(bool(data['hypotheses'].get(i)) for i in indices),shuffle=shuffle)


def official(root,cat):
    data=load(root,cat,'val');model=RelativeAdapter030(data['parent_state'],'anchor_all').cuda();rows={'baseline029':evaluate(model,data,'val',True)};parent=json.loads((PARENT/('official_'+cat+'.json')).read_text())['rows']['full_learned_written']
    if rows['baseline029']['ranks']!=parent['ranks']:raise ValueError('030-P official baseline rank mismatch')
    rows['enumeration']=enum_evaluate(model,data,'val');rows['enumeration_shuffled']=enum_evaluate(model,data,'val',True)
    for arm in ARMS:
        report=json.loads((root/'fits'/cat/arm/'results.json').read_text())
        if file_sha(report['selected_weight'])!=report['selected_weight_sha256']:raise ValueError('030-P selected weight changed')
        state=torch.load(report['selected_weight'],map_location='cpu',weights_only=True);model=RelativeAdapter030(data['parent_state'],arm).cuda()
        with torch.no_grad():model.blend.copy_(state['blend'].cuda());model.gamma.copy_(state['gamma'].cuda())
        rows[arm]=evaluate(model,data,'val')
    strata={}
    for name,mask in [('all_queries',torch.ones(len(data['semantics']),dtype=torch.bool)),('rule_axis',data['active_all'].any(1)),('caption_source',data['active_source'].any(1)),('enumerated',torch.tensor([i in data['hypotheses'] for i in range(len(data['semantics']))]))]:
        ix=mask.nonzero().flatten().tolist();strata[name]=dict(queries=len(ix),indices=ix,metrics={method:{'r'+str(k):100*sum(value['ranks'][i]<=k for i in ix)/len(ix) if ix else None for k in (1,10,50)} for method,value in rows.items()})
    comparisons=[]
    for a,b in [('enumeration','baseline029'),('enumeration','enumeration_shuffled'),('polarity_axis','anchor_all'),('target_minus_source','anchor_source'),('polarity_axis','baseline029'),('target_minus_source','baseline029')]:
        for k in (1,10,50):
            x=rows[a]['ranks'];y=rows[b]['ranks'];g=[i for i,(u,v) in enumerate(zip(x,y)) if u<=k<v];l=[i for i,(u,v) in enumerate(zip(x,y)) if v<=k<u];comparisons.append(dict(a=a,b=b,k=k,queries=len(x),gains=len(g),losses=len(l),gain_indices=g,loss_indices=l,delta_pp=100*(len(g)-len(l))/len(x)))
    atomic_write_json(root/('official_'+cat+'.json'),dict(status='complete',category=cat,queries=len(data['semantics']),gallery=len(data['banks']['val']['ids']),rows=rows,strata=strata,comparisons=comparisons,baseline029_all_ranks_exact=True,new_api_calls=0,failed_online_responses_used=False,reference_id_excluded=True,gt_inserted=False));print(json.dumps(dict(status='complete',cat=cat,metrics={n:{k:v[k] for k in ('r1','r10','r50')} for n,v in rows.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',required=True);p.add_argument('--build',choices=('train','val'));p.add_argument('--fit',choices=ARMS);p.add_argument('--official',action='store_true');a=p.parse_args()
    if a.build:build(a.root,a.category,a.build)
    elif a.fit:fit(a.root,a.category,a.fit)
    elif a.official:official(a.root,a.category)
    else:p.error('build/fit/official required')
