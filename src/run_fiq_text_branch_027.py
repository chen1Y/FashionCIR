"""Zero-API027: DQU textual branch and matched current query-side text ablations."""
import argparse,json,math,time,gc,subprocess,os
from pathlib import Path
from fiq_text_branch_math_027 import ROWS,corrected_modification,textual_query,metrics,transition,first_valid_index
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

CODE=Path(__file__).resolve().parents[1];PARENT=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023');DATA=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ');PRIMARY=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_shirt_toptee_20261022');CATS=('shirt','toptee')
FILES=('src/run_fiq_text_branch_027.py','src/fiq_text_branch_math_027.py','tests/test_fiq_text_branch_027.py','src/dquDataset.py','src/dquModel.py','src/dquTest.py','src/expanded_attr_model.py','src/attr_token_pair_alignment.py','src/attr_slot_fusion.py','src/attr_slot_data.py','src/train_clean_attr_alignment.py','src/train_attr_pair_alignment.py','src/train_attr_slot_pilot.py','src/experiment_storage.py','src/fashioniq_noise.py','src/generate_structured_edits.py')


def text_only_query(output,arm):
    from torch.nn import functional as F
    if arm not in ('original','slots','dual'):raise ValueError('Unknown fixed arm')
    return F.normalize((output['t_mod'] if arm=='original' else output['text']).float(),dim=-1)


def prepare(root,tests):
    from experiment_storage import require_space
    from dquDataset import FashionIQ
    if root.exists():raise FileExistsError('027 is independent and immutable')
    if not tests.read_text().rstrip().endswith('OK'):raise ValueError('027 tests failed')
    require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8)
    if json.loads((PARENT/'audit/verification.json').read_text())['status']!='verified':raise ValueError('023 verification required')
    root.mkdir();protected={};categories={}
    def bind(path,expected=None):
        path=Path(path);h=file_sha(path)
        if expected and h!=expected:raise ValueError('Parent provenance changed: '+str(path))
        protected[str(path)]=h;return h
    for cat in CATS:
        m=json.loads((PARENT/'metadata'/(cat+'_val.json')).read_text());s=json.loads((PARENT/'texts'/(cat+'_val.json')).read_text())['samples'];rawpath=DATA/'captions'/('cap.'+cat+'.val.json');raw=json.loads(rawpath.read_text());cap_path=DATA/'captions'/('image_captions_'+cat+'_val.json');caps=json.loads(cap_path.read_text());cor_path=DATA/'captions'/('correction_dict_'+cat+'.json');correction=json.loads(cor_path.read_text());fake=object.__new__(FashionIQ);fake.correction_dict=correction
        records=[]
        for r in m['records']:
            original=raw[r['source_index']]
            if original['candidate']!=r['candidate'] or original['target']!=r['target']:raise ValueError('Original val ID binding changed')
            mod=corrected_modification(original['captions'],correction)
            if mod!=fake._concat_text(original['captions']):raise ValueError('Official DQU textual input did not reproduce')
            records.append(dict(r,dqu_corrected_mod=mod,dqu_textual_query=textual_query(caps[r['candidate']],mod),reference_caption=caps[r['candidate']]))
        asset=json.loads((PARENT/('assets_'+cat+'_val.json')).read_text());base=PRIMARY/cat/'checkpoints'/(cat+'_dqu_original_seed42_best.pt');bind(base,asset['base_sha256']);bind(asset['cache_path'],asset['cache_sha256']);bind(PARENT/'metadata'/(cat+'_val.json'),asset['metadata_sha256'])
        for p in (rawpath,cap_path,cor_path,PARENT/'texts'/(cat+'_val.json'),PARENT/'official'/cat/'results.json'):bind(p)
        for arm in ('original','slots','dual'):
            p=PARENT/'fits'/cat/arm/'results.json';d=json.loads(p.read_text());bind(p);bind(d['selected_weight'],d['selected_weight_sha256'])
        example=first_valid_index(records,s,list(range(len(records))))
        categories[cat]=dict(queries=len(records),gallery=len(m['gallery_ids']),base=str(base),cache_path=asset['cache_path'],metadata_path=str(PARENT/'metadata'/(cat+'_val.json')),text_path=str(PARENT/'texts'/(cat+'_val.json')),val_example_index=example)
        atomic_write_json(root/(cat+'_queries.json'),dict(records=records));bind(root/(cat+'_queries.json'))
    cat='toptee'
    for p in (PARENT/'metadata'/('toptee_train.json'),PARENT/'texts/toptee_train.json',PARENT/'teacher_toptee.json',PARENT/'assets_toptee_train.json'):bind(p)
    for name in ('teacher_toptee.json','assets_toptee_train.json'):
        d=json.loads((PARENT/name).read_text());bind(d['cache_path'],d['cache_sha256'])
    plan=dict(id='027_query_text_branch_ablation',source_hashes={n:file_sha(CODE/n) for n in FILES},protected_sha256=protected,categories=categories,rows=list(ROWS),precision='BF16 text forward, normalized FP32 scores; same immutable023 BF16 image gallery for every row',batch_size=8,api_calls=0,fit_updates=0,image_forwards=0,gallery_reencoded=False,reference_query_image_in_text_only_rows=False,reference_caption_in_dqu_caption_row=True,current_checkpoint_selection_unchanged=True,posthoc_inference_ablation_not_matched_text_only_retraining=True,max_category_seconds=3600,max_total_seconds=7200,sample_rule='lowest original source_index among existing status=ok; never ranks')
    atomic_write_json(root/'plan.json',plan);(root/'tests.log').write_text(tests.read_text());print(json.dumps(dict(status='prepared',categories=categories,rows=list(ROWS),api_calls=0,fit_updates=0,image_forwards=0)))


def run(root):
    import torch
    from torch.nn import functional as F
    from train_attr_slot_pilot import load_clip
    from attr_token_pair_alignment import PairAlignedText,packed_audit
    from expanded_attr_model import branch_forward,compose,ranks_from_scores,losses,legal_masks
    from experiment_storage import require_space
    plan=json.loads((root/'plan.json').read_text());started=time.monotonic()
    if (root/'state.json').exists():raise FileExistsError('Inspect existing027, no blind retry')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    p=Path('/proc')/str(os.getpid());stat=(p/'stat').read_text().split(') ',1)[1].split();ident=dict(pid=os.getpid(),ppid=os.getppid(),start_ticks=int(stat[19]),cwd=str((p/'cwd').resolve()),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode());state=dict(status='running',identity=ident,categories={},api_calls=0,fit_updates=0,image_forwards=0)
    def save():atomic_write_json(root/'state.json',state)
    def guard(begin):
        require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',2)
        if (root/'cancel.request').exists():raise KeyboardInterrupt('027 cancelled')
        if time.monotonic()-begin>3300 or time.monotonic()-started>plan['max_total_seconds']:raise TimeoutError('027 deadline exceeded')
        for n,h in plan['source_hashes'].items():
            if file_sha(CODE/n)!=h:raise ValueError('027 sealed source changed')
    def top5(score,ids,label,reference):
        score=score.clone();score[reference]=-torch.inf;order=torch.argsort(score,descending=True,stable=True);return dict(target_rank=int(ranks_from_scores(score[None],torch.tensor([label],device=score.device),torch.tensor([reference],device=score.device))[0]),top5=[dict(image_id=ids[int(i)],score=float(score[i]),is_actual_target=int(i)==label) for i in order[:5]])
    try:
        save();guard(started)
        for p,h in plan['protected_sha256'].items():
            if file_sha(p)!=h:raise ValueError('027 protected input changed')
        for cat,config in plan['categories'].items():
            begin=time.monotonic();state.update(status='evaluating_'+cat);save();torch.manual_seed(42);torch.set_num_threads(4)
            metadata=json.loads(Path(config['metadata_path']).read_text());records=json.loads((root/(cat+'_queries.json')).read_text())['records'];samples=json.loads(Path(config['text_path']).read_text())['samples'];cache=torch.load(config['cache_path'],map_location='cpu',weights_only=True,mmap=True);ids=cache['ids'];index={i:j for j,i in enumerate(ids)}
            if ids!=metadata['gallery_ids']:raise ValueError('Gallery ordering changed')
            gallery=cache['gallery'].cuda();del cache;labels=torch.tensor([index[r['target']] for r in records],device='cuda');references=torch.tensor([index[r['candidate']] for r in records],device='cuda');clip,_,tok=load_clip(Path(config['base']));model=PairAlignedText(clip,tok,2).cuda();model.eval();model.requires_grad_(False);initial=model.embedding.weight[-4:].detach().clone()
            def forbid_visual_forward(*args):raise RuntimeError('027 must not call the image encoder')
            visual_guard=clip.visual.register_forward_pre_hook(forbid_visual_forward)
            bound=[dict(r,target_attrs=s['target_attrs'],generation_status=s['status']) for r,s in zip(records,samples)];example_i=config['val_example_index'];example=dict(query=records[example_i],generated=samples[example_i],rows={});ranks={};token_updates={}
            with torch.inference_mode():
                for name in ROWS:
                    guard(begin);state['active_row']=name;save();arm={'current_original_text':'original','current_slots_text':'slots','current_dual_text':'dual'}.get(name)
                    if arm:
                        fit=json.loads((PARENT/'fits'/cat/arm/'results.json').read_text());delta=torch.load(fit['selected_weight'],map_location='cpu',weights_only=True);model.load_delta(delta['state'])
                        displacement=(model.embedding.weight[-4:]-initial).float();token_updates[arm]=dict(embedding_dimension=initial.shape[1],registered_rows=4,trainable_block_parameters=sum(v.numel() for v in delta['state']['blocks'].values()),new_row_l2=displacement.norm(dim=1).tolist(),new_row_max_abs=displacement.abs().max(dim=1).values.tolist(),selected_epoch=fit['selected_epoch'],updates=fit['updates'],presentations=fit['sample_presentations'],selected_weight_sha256=fit['selected_weight_sha256']);del delta
                    if not torch.equal(model.embedding.weight[:-4],clip.token_embedding.weight):raise ValueError('Old vocabulary unexpectedly changed')
                    rank=[]
                    for j in range(0,len(records),8):
                        guard(begin);batch=records[j:j+8]
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            if name.startswith('dqu_'):
                                key={'dqu_caption_mod_native':'dqu_textual_query','dqu_corrected_mod_native':'dqu_corrected_mod','dqu_raw_mod_native':'modification_text'}[name];q=F.normalize(clip.encode_text(tok([r[key] for r in batch]).cuda()).float(),dim=-1);out=None
                            else:
                                active=arm or 'original';out=branch_forward(model,bound[j:j+8],active);q=text_only_query(out,active)
                        score=q@gallery.T;rank.extend(ranks_from_scores(score,labels[j:j+len(batch)],references[j:j+len(batch)]).tolist())
                        if j<=example_i<j+len(batch):
                            pos=example_i-j;example['rows'][name]=top5(score[pos],ids,int(labels[example_i]),int(references[example_i]))
                            if name=='current_dual_text':
                                q_full=compose(out,gallery[references[j:j+len(batch)]],'dual');fullscore=q_full@gallery.T;demo=top5(fullscore[pos],ids,int(labels[example_i]),int(references[example_i]));saved=json.loads((PARENT/'official'/cat/'results.json').read_text())['rows']['dual']['ranks'][example_i]
                                if demo['target_rank']!=saved:raise ValueError('Original multimodal example did not reproduce saved rank')
                                example['original_multimodal_dual']=demo;example['packed']=packed_audit(model,[samples[example_i]['target_attrs']])[0]
                    ranks[name]=rank
                    state['rows_completed']=list(ranks);save()
                if cat=='toptee':
                    tm=json.loads((PARENT/'metadata/toptee_train.json').read_text());ts=json.loads((PARENT/'texts/toptee_train.json').read_text())['samples'];ti=first_valid_index(tm['records'],ts,tm['offsets']['fit']);row=tm['records'][ti];ar=json.loads((PARENT/'assets_toptee_train.json').read_text());data=torch.load(ar['cache_path'],map_location='cpu',weights_only=True,mmap=True);gi={x:i for i,x in enumerate(data['ids'])};bank_ids=tm['banks']['fit'];bank_index={x:i for i,x in enumerate(bank_ids)};train_gallery=torch.stack([data['gallery'][gi[x]] for x in bank_ids]).cuda();ref=data['gallery'][gi[row['candidate']]][None].cuda();del data
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=branch_forward(model,[dict(row,target_attrs=ts[ti]['target_attrs'])],'dual');query=compose(out,ref,'dual')
                    one=dict(tm,offsets={**tm['offsets'],'fit':[ti]});mask,target=legal_masks(one,'fit');teacher_info=json.loads((PARENT/'teacher_toptee.json').read_text());teacher=torch.load(teacher_info['cache_path'],map_location='cpu',weights_only=True,mmap=True)['teachers'][ti:ti+1].cuda();loss=losses(out,query,train_gallery,target.cuda(),mask.cuda(),teacher,'dual');training_example=dict(query=row,generated=ts[ti],packed=packed_audit(model,[ts[ti]['target_attrs']])[0],training_forward_demo_no_update=True,demo_batch_size=1,actual_training_batch_size=8,loss_at_selected_checkpoint={k:float(v) for k,v in loss.items()},gallery=len(bank_ids),retrieval_demo_training_split=True,retrieval=top5((query@train_gallery.T)[0],bank_ids,int(target[0]),bank_index[row['candidate']]))
                    atomic_write_json(root/'training_example_toptee.json',training_example);del train_gallery,ref,teacher,out,query
                    if not all(math.isfinite(v) for v in training_example['loss_at_selected_checkpoint'].values()):raise FloatingPointError('Nonfinite demo loss')
            if any(p.grad is not None for p in model.parameters()):raise ValueError('027 unexpectedly produced gradients')
            result=dict(status='complete',category=cat,queries=len(records),gallery=len(ids),rows={n:dict(**metrics(rr),ranks=rr) for n,rr in ranks.items()},comparisons={a+'_minus_'+b:{str(k):transition(ranks[a],ranks[b],k) for k in (1,10,50)} for a,b in [('current_original_text','dqu_caption_mod_native'),('current_slots_text','dqu_caption_mod_native'),('current_dual_text','dqu_caption_mod_native'),('current_slots_text','current_original_text'),('current_dual_text','current_original_text'),('dqu_caption_mod_native','dqu_corrected_mod_native'),('dqu_corrected_mod_native','dqu_raw_mod_native')]},token_updates=token_updates,reference_query_image_used=False,gallery_remains_images=True,reference_id_excluded=True,all_queries_retained=True,api_calls=0,fit_updates=0,image_forwards=0,seconds=time.monotonic()-begin,scope='Frozen selected checkpoints with query-side image contribution removed at inference; not retrained text-only models')
            atomic_write_json(root/(cat+'_results.json'),result);atomic_write_json(root/(cat+'_example.json'),example);state['categories'][cat]='complete';save();visual_guard.remove();del model,clip,gallery;gc.collect();torch.cuda.empty_cache()
        for p,h in plan['protected_sha256'].items():
            if file_sha(p)!=h:raise ValueError('027 changed a protected input')
        state.update(status='complete',seconds=time.monotonic()-started);save();print(json.dumps(dict(status='complete',api_calls=0,fit_updates=0,image_forwards=0)))
    except BaseException as exc:state.update(status='failed',error_type=type(exc).__name__,error=str(exc));save();raise


if __name__=='__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--tests',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.root,a.tests)
    elif a.run:run(a.root)
    else:raise ValueError('Explicit prepare or run required')
