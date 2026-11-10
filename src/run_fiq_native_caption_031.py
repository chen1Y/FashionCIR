"""031: sealed, zero-fit/API native-text transplant on complete official val.

No old root/cache/checkpoint is modified. This runner deliberately does not
import a training entry point, create an optimizer, or access API credentials.
"""
import argparse
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from fiq_native_caption_math_031 import METHODS, COMPARISONS, native_queries, metrics, transitions

CODE = Path(__file__).resolve().parents[1]
BASE = Path('/data/8T1/lcy/CIRData')
PARENT = BASE / 'runs/fiq-fusion-029'
DATA = BASE / 'autodl-tmp/data/FashionIQ'
CACHE = Path('/dev/shm/fiq_native_caption_031')
RUNTIME = Path('/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python')
EXPECTED = {'shirt': (2038, 6346, '6690822bf59dbc4c6c9e45df9d2a0026da033b327afef595bbc17829d2150866'),
            'toptee': (1961, 5373, 'cfd39cda844b3453a6460997dacda886c08c07e3f74b9249cff2d9af8c3c437e')}
FILES = ('src/run_fiq_native_caption_031.py', 'src/fiq_native_caption_math_031.py',
         'tests/test_fiq_native_caption_031.py', 'src/fiq_text_branch_math_027.py',
         'src/fiq_fusion_029.py', 'src/fiq_gate_patch_model_028.py', 'src/attr_token_pair_alignment.py',
         'src/attr_slot_fusion.py', 'src/attr_slot_data.py', 'src/expanded_attr_model.py',
         'src/train_clean_attr_alignment.py', 'src/train_attr_pair_alignment.py',
         'src/train_attr_slot_pilot.py', 'src/dquDataset.py', 'src/fashioniq_noise.py',
         'src/experiment_storage.py', 'src/attr_slot_rank_math.py', 'src/generate_structured_edits.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n'); os.replace(temp, path)


def resolve(old):
    p = str(old)
    if p.startswith('/root/autodl-tmp/'): return BASE / p[len('/root/'):]
    if p.startswith('/root/fiq-'): return BASE / 'runs' / p[len('/root/'):]
    return Path(p)


def identity():
    p = Path('/proc') / str(os.getpid()); fields = (p/'stat').read_text().split(') ', 1)[1].split()
    return dict(pid=os.getpid(), ppid=os.getppid(), start_ticks=int(fields[19]),
                cwd=str((p/'cwd').resolve()), command=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode())


def snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                     '--format=csv,noheader,nounits'], text=True)
    devices = []
    for line in raw.strip().splitlines():
        ix, uid, util, free = [s.strip() for s in line.split(',')]
        devices.append(dict(index=int(ix), uuid=uid, utilization=int(util), free_mib=int(free)))
    return dict(time=time.time(), devices=devices, compute_apps=apps.strip().splitlines())


def choose(first, second):
    previous = {d['uuid']: d for d in first['devices']}; busy = set()
    for s in (first, second): busy.update(line.split(',')[0].strip() for line in s['compute_apps'])
    choices = [d for d in second['devices'] if d['uuid'] in previous
               and d['utilization'] <= 10 and previous[d['uuid']]['utilization'] <= 10
               and min(d['free_mib'], previous[d['uuid']]['free_mib']) >= 12288]
    return sorted(choices, key=lambda d: (d['uuid'] in busy, -d['free_mib'], d['index']))[0] if choices else None


def prepare(root, tests):
    if root.exists(): raise FileExistsError('031 root already exists: inspect, never replay')
    log = Path(tests).read_text()
    if not log.rstrip().endswith('OK') or 'skipped=' in log: raise ValueError('Seven real server tests required')
    protected = {}; categories = {}
    def bind(path, expected=None):
        h = sha(path)
        if expected and h != expected: raise ValueError('Parent SHA mismatch: ' + str(path))
        protected[str(path)] = h
        return h
    for cat, (n, g, base_sha) in EXPECTED.items():
        mp, tp = PARENT/'metadata'/f'{cat}_val.json', PARENT/'texts'/f'{cat}_val.json'
        m, t = read(mp), read(tp); a = read(PARENT/f'assets_{cat}_val.json')
        bind(mp, a['metadata_sha256']); bind(tp, a['text_sha256'])
        if len(m['records']) != n or len(t['samples']) != n or len(m['gallery_ids']) != g: raise ValueError('Full denominator changed')
        if len(set(m['gallery_ids'])) != g: raise ValueError('Duplicate original gallery ID')
        files = {name: DATA/'captions'/pattern.format(cat=cat) for name, pattern in (
            ('raw', 'cap.{cat}.val.json'), ('captions', 'image_captions_{cat}_val.json'),
            ('correction', 'correction_dict_{cat}.json'), ('keywords', 'keywords_in_mods_{cat}.json'))}
        for name, p in files.items(): bind(p, a['keywords_sha256'] if name == 'keywords' else None)
        if sha(files['raw']) != m['caption_sha256']: raise ValueError('Original captions SHA mismatch')
        original = read(files['raw']); queries = native_queries(m['records'], original, read(files['captions']), read(files['correction']))
        keywords = read(files['keywords']); gallery = set(m['gallery_ids'])
        for r, s in zip(m['records'], t['samples']):
            if (r['source_index'] != s['source_index'] or r['candidate'] != s['reference']
                    or r['modification_text'] != s['instruction']
                    or m['image_sha256'][r['candidate']] != s['reference_sha256']): raise ValueError('029 response/query SHA binding changed')
            if r['candidate'] == r['target'] or not {r['candidate'], r['target']} <= gallery: raise ValueError('Illegal official IDs')
            if r['candidate']+'_'+r['target'] not in keywords: raise ValueError('Missing original DQU keyword')
        fit_path = PARENT/'fits'/cat/'full_learned_written/results.json'; f = read(fit_path); bind(fit_path)
        delta = resolve(f['selected_weight']); bind(delta, f['selected_weight_sha256'])
        base = BASE/'autodl-tmp/results/fiq_shirt_toptee_20261022'/cat/'checkpoints'/f'{cat}_dqu_original_seed42_best.pt'
        if a['base_sha256'] != base_sha or not base.is_file(): raise ValueError('Unrecognized native checkpoint')
        # Base SHA is verified while copying the existing weight to tmpfs before load.
        categories[cat] = dict(queries=n, gallery=g, base=str(base), base_sha256=base_sha,
                               base_bytes=base.stat().st_size, delta=str(delta), delta_sha256=f['selected_weight_sha256'],
                               selected_epoch=f['selected_epoch'], metadata=str(mp), text=str(tp),
                               files={k:str(p) for k,p in files.items()}, native_queries=queries)
    if CACHE.exists(): raise FileExistsError('Inspect existing 031 tmpfs; do not overwrite')
    root.mkdir(parents=True)
    for cat, config in categories.items():
        queries = config.pop('native_queries'); write(root/f'{cat}_native_inputs.json', queries)
        protected[str(root/f'{cat}_native_inputs.json')] = sha(root/f'{cat}_native_inputs.json')
    plan = dict(id='031_native_caption_transplant', methods=METHODS, comparisons=COMPARISONS,
                categories=categories, protected_sha256=protected, sources={n:sha(CODE/n) for n in FILES},
                precision='BF16 forwards, normalized FP32 gallery and scoring', batch_size=8, seed=42,
                api_calls=0, fit_updates=0, original_selected_checkpoints_unchanged=True,
                beta_fixed_zero_control=True, no_val_parameter_selection=True,
                cached_features_missing_after_migration=True, new_gallery_patch_features=False,
                limits=dict(total_seconds=7200, category_seconds=3600, gpu_memory_fraction=.4),
                limitation='Inference input transplant and anchored control; gates have not been retrained on caption input')
    write(root/'plan.json', plan); (root/'tests.log').write_text(log)
    write(root/'preflight.json', dict(status='passed', categories={c:{k:v[k] for k in ('queries','gallery','selected_epoch')} for c,v in categories.items()},
                                     response_bindings_verified=3999, api_calls=0, fit_updates=0,
                                     source_files=len(FILES), base_sha_validation='required before actual load'))
    print('031 prepared: all 3999 metadata/response bindings passed', flush=True)


def launch(root):
    if (root/'state.json').exists() or (root/'launch.json').exists(): raise FileExistsError('031 already launched')
    first = snapshot(); time.sleep(5); second = snapshot(); selected = choose(first, second)
    write(root/'gpu_selection.json', dict(snapshots=[first, second], selected=selected))
    if selected is None:
        write(root/'state.json', dict(status='abandoned_no_suitable_gpu', api_calls=0, fit_updates=0)); return
    env = {k:os.environ[k] for k in ('HOME','PATH','LANG','LD_LIBRARY_PATH') if k in os.environ}
    env.update(CUDA_VISIBLE_DEVICES=selected['uuid'], PYTHONPATH=str(CODE/'src'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    with (root/'run.log').open('xb') as log:
        child = subprocess.Popen([str(RUNTIME),str(CODE/'src/run_fiq_native_caption_031.py'),
                                  '--root',str(root),'--run','--gpu-uuid',selected['uuid']],
                                 cwd=CODE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    write(root/'launch.json',dict(pid=child.pid, gpu=selected, plan_sha256=sha(root/'plan.json'), time=time.time()))
    print('031 worker PID '+str(child.pid),flush=True)


def run(root, uid):
    import fcntl
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock = (root/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'state.json').exists(): raise FileExistsError('031 state exists')
    plan = read(root/'plan.json'); start=time.monotonic()
    state=dict(status='starting',identity=identity(),gpu_uuid=uid,api_calls=0,fit_updates=0,categories={})
    def save(): write(root/'state.json',state)
    def verify_small():
        for mapping in (plan['sources'],plan['protected_sha256']):
            for name, h in mapping.items():
                p=CODE/name if name in plan['sources'] else Path(name)
                if sha(p)!=h: raise ValueError('031 protected input/source changed: '+str(p))
    def guard(begin=start):
        if time.monotonic()-start>7200 or time.monotonic()-begin>3500: raise TimeoutError('031 time limit')
        if (root/'cancel.request').exists(): raise RuntimeError('031 cancelled')
        for path,gib in ((CODE,30),(BASE,8),(CACHE,2)):
            if shutil.disk_usage(path).free < gib*(1<<30): raise OSError('031 insufficient storage '+str(path))
    try:
        save(); CACHE.mkdir();guard();verify_small()
        current=next(d for d in snapshot()['devices'] if d['uuid']==uid)
        if current['utilization']>10 or current['free_mib']<12288:
            state['status']='abandoned_no_suitable_gpu';save();return
        import torch
        from torch.nn import functional as F
        from PIL import Image
        from train_attr_slot_pilot import load_clip
        from fiq_fusion_029 import FusionText029, encode_parts, fuse_parts
        from train_clean_attr_alignment import plain_text
        from expanded_attr_model import ranks_from_scores
        from dquDataset import FashionIQ
        from attr_token_pair_alignment import packed_audit
        from types import SimpleNamespace
        torch.set_num_threads(4);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        torch.cuda.set_per_process_memory_fraction(.4)
        state['versions']=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name())
        save()
        for cat,config in plan['categories'].items():
            begin=time.monotonic();guard(begin);state.update(status='loading_'+cat,phase='copy_base');save()
            cached=CACHE/f'{cat}_base.pt';h=hashlib.sha256()
            with Path(config['base']).open('rb') as source,cached.open('xb') as target:
                for b in iter(lambda:source.read(8<<20),b''):
                    target.write(b);h.update(b);guard(begin)
            if h.hexdigest()!=config['base_sha256'] or cached.stat().st_size!=config['base_bytes']: raise ValueError('Native base SHA changed')
            m,t=read(config['metadata']),read(config['text']);native=read(root/f'{cat}_native_inputs.json')
            rows=[dict(r,target_attrs=s['target_attrs']) for r,s in zip(m['records'],t['samples'])]
            ids=m['gallery_ids'];index={name:i for i,name in enumerate(ids)}
            labels=torch.tensor([index[r['target']] for r in rows],device='cuda')
            refs=torch.tensor([index[r['candidate']] for r in rows],device='cuda')
            clip,transform,tokenizer=load_clip(cached);clip.eval()
            model=FusionText029(clip,tokenizer,2).cuda();model.image_fusion.from_dqu(cached)
            model.eval();model.requires_grad_(False)
            native_head=copy.deepcopy(model.image_fusion).eval().requires_grad_(False)
            fake=object.__new__(FashionIQ);fake.category=cat;fake.image_dir=DATA/'resized_image'
            fake.transform=(transform,transform);fake.key_words=read(config['files']['keywords'])
            state.update(status='encoding_'+cat,phase='gallery',completed=0,total=len(ids));save()
            gallery=[];written=[];checked={}
            with torch.inference_mode():
                for j in range(0,len(ids),8):
                    guard(begin);batch=ids[j:j+8];images=[]
                    for name in batch:
                        path=fake._image_path(name);blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
                        if digest!=m['image_sha256'][name]: raise ValueError('JPEG SHA changed: '+name)
                        checked[name]=digest
                        with Image.open(io.BytesIO(blob)) as im: images.append(transform(im.convert('RGB')))
                    with torch.autocast('cuda',dtype=torch.bfloat16):feat=clip.encode_image(torch.stack(images).cuda())
                    if not torch.isfinite(feat).all():raise FloatingPointError('031 nonfinite gallery')
                    gallery.append(F.normalize(feat.float(),dim=-1).cpu())
                    if j%128==0:state['completed']=j+len(batch);save()
                state.update(phase='written_reference',completed=0,total=len(rows));save()
                for j in range(0,len(rows),8):
                    guard(begin);batch=rows[j:j+8]
                    ims=torch.stack([fake.get_written_img(r['candidate'],r['target'],0)[0] for r in batch]).cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16):feat=clip.encode_image(ims)
                    if not torch.isfinite(feat).all():raise FloatingPointError('031 nonfinite written reference')
                    written.append(F.normalize(feat.float(),dim=-1).cpu())
                    if j%128==0:state['completed']=j+len(batch);save()
                gallery_cpu=torch.cat(gallery);written_cpu=torch.cat(written)
                # References read again by OpenCV must retain the same bytes.
                for name in set(r['candidate'] for r in rows):
                    if sha(fake._image_path(name))!=checked[name]:raise ValueError('Written reference JPEG changed')
                bankpath=CACHE/f'{cat}_val.pt'
                torch.save(dict(ids=ids,gallery=gallery_cpu,written=written_cpu,checked_image_sha256=checked,
                                metadata_sha256=sha(config['metadata']),base_sha256=config['base_sha256']),bankpath)
                write(root/f'{cat}_assets.json',dict(path=str(bankpath),sha256=sha(bankpath),gallery=len(ids),queries=len(rows),
                      gallery_forwards=len(ids),written_forwards=len(rows),patch_forwards=0,old_cache_missing=True,
                      base_sha256=config['base_sha256'],metadata_sha256=sha(config['metadata']),text_sha256=sha(config['text'])))
                gallery=gallery_cpu.cuda();written=written_cpu.cuda();reference=gallery[refs]
                del gallery_cpu,written_cpu,ims,feat
                results={name:dict(ranks=[],alpha=[],beta=[],slot_weights=[]) for name in METHODS}
                examples={name:{} for name in METHODS};native_m=[];lengths=[];zero_max=0.
                def collect(name,out,j):
                    q=out['q'];score=q.float()@gallery.T
                    if not torch.isfinite(q).all() or not torch.isfinite(score).all():raise FloatingPointError('031 nonfinite query/score')
                    n=len(q);rr=ranks_from_scores(score,labels[j:j+n],refs[j:j+n]).tolist()
                    dst=results[name];dst['ranks'].extend(rr)
                    for key in ('alpha','beta','slot_weights'):
                        if key in out:dst[key].extend(out[key].float().cpu().tolist())
                    if j==0:
                        s=score[0].clone();s[refs[0]]=-torch.inf;order=torch.argsort(s,descending=True,stable=True)[:5]
                        examples[name]=dict(rank=rr[0],top5=[dict(image_id=ids[k],score=float(s[k]),is_target=ids[k]==rows[0]['target']) for k in order.tolist()])
                state.update(status='evaluating_'+cat,phase='native_baselines',completed=0,total=len(rows));save()
                for j in range(0,len(rows),8):
                    guard(begin);batch=rows[j:j+8];texts=[r['native_text'] for r in native[j:j+8]]
                    lengths.extend(dict(source_index=r['source_index'],native_bpe=len(tokenizer.encode(r['native_text']))+2,
                                        raw_bpe=len(tokenizer.encode(rows[j+i]['modification_text']))+2) for i,r in enumerate(native[j:j+8]))
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        mt=F.normalize(clip.encode_text(tokenizer(texts).cuda()).float(),dim=-1)
                        mr=F.normalize(clip.encode_text(tokenizer([r['modification_text'] for r in batch]).cuda()).float(),dim=-1)
                        q,a=native_head(mt,written[j:j+len(batch)]);qr,ar=native_head(mr,written[j:j+len(batch)])
                    native_m.append(mt.cpu());collect(METHODS[0],dict(q=q,alpha=a),j);collect(METHODS[1],dict(q=qr,alpha=ar),j)
                    if j%128==0:state['completed']=j+len(batch);save()
                native_m=torch.cat(native_m).cuda()
                delta=torch.load(config['delta'],map_location='cpu',weights_only=True)['state']
                if set(delta)!=set(model.delta_state()):raise ValueError('Strict 029D delta component mismatch')
                # requires_grad_(False) empties delta_state.blocks; compare fixed last-two block names instead.
                names=set(n for n,_ in model.clip.named_parameters() if n.startswith(tuple('transformer.resblocks.'+str(k)+'.' for k in range(len(model.clip.transformer.resblocks)-2,len(model.clip.transformer.resblocks)))))
                if set(delta['blocks'])!=names or tuple(delta['new_token_rows'].shape)!=(4,1024):raise ValueError('029D delta block/row mismatch')
                model.load_delta(delta);del delta;model.eval();model.requires_grad_(False)
                if not torch.equal(model.embedding.weight[:-4],model.clip.token_embedding.weight):raise ValueError('029D old vocabulary changed')
                proxy=SimpleNamespace(attribute_gate=model.attribute_gate,text_mixture=model.text_mixture,image_fusion=native_head)
                state.update(phase='four_029d_controls',completed=0);save();packed=[]
                for j in range(0,len(rows),8):
                    guard(begin);batch=rows[j:j+8];n=len(batch);v=reference[j:j+n];w=written[j:j+n]
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        parts=encode_parts(model,batch,'full_learned_written')
                        mcap=plain_text(model,[r['native_text'] for r in native[j:j+n]])
                        old=fuse_parts(model,parts,v,w,'full_learned_written')
                        new=fuse_parts(model,dict(parts,mod=mcap),v,w,'full_learned_written')
                        q0,a0=model.image_fusion(mcap,w)
                        anchor=fuse_parts(proxy,dict(parts,mod=native_m[j:j+n]),v,w,'full_learned_written')
                        qzero,_=native_head(F.normalize(native_m[j:j+n],dim=-1),w)
                        qnative,_=native_head(native_m[j:j+n],w)
                    err=float((qzero-qnative).abs().max());zero_max=max(zero_max,err)
                    if err>1e-6 or not torch.equal(ranks_from_scores(qzero@gallery.T,labels[j:j+n],refs[j:j+n]),
                                                  torch.tensor(results[METHODS[0]]['ranks'][j:j+n],device='cuda')):raise ValueError('Native-anchor beta-zero identity failed')
                    collect(METHODS[2],old,j);collect(METHODS[3],dict(q=q0,alpha=a0,beta=torch.zeros(n,device='cuda')),j)
                    collect(METHODS[4],new,j);collect(METHODS[5],anchor,j)
                    audits=packed_audit(model,[r['target_attrs'] for r in batch])
                    packed.extend(dict(source_index=batch[i]['source_index'],valid=a['valid'],truncated=a['truncated'],eos_position=a['eos_position']) for i,a in enumerate(audits))
                    if j%128==0:state['completed']=j+n;save()
                if any(p.grad is not None for p in model.parameters()):raise ValueError('Unexpected gradients')
                if any(len(r['ranks'])!=len(rows) for r in results.values()):raise ValueError('Dropped official query')
                report=dict(status='complete',category=cat,queries=len(rows),gallery=len(ids),
                            rows={name:dict(metrics(r['ranks']),**r) for name,r in results.items()},
                            comparisons={a+'_minus_'+b:{str(k):transitions(results[a]['ranks'],results[b]['ranks'],k) for k in (1,10,50)} for a,b in COMPARISONS},
                            native_anchor_beta0_max_abs=zero_max,all_queries_retained=True,reference_id_excluded=True,
                            gt_inserted=False,api_calls=0,fit_updates=0,seconds=time.monotonic()-begin,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/(1<<30),
                            truncation=dict(native=sum(v['native_bpe']>77 for v in lengths),raw=sum(v['raw_bpe']>77 for v in lengths)))
                write(root/f'{cat}_results.json',report);write(root/f'{cat}_token_audit.json',dict(lengths=lengths,attributes=packed))
                write(root/f'{cat}_example.json',dict(query=rows[0],native_input=native[0],rows=examples,rule='original source_index zero, no rank selection'))
                state['categories'][cat]=dict(status='complete',seconds=report['seconds']);save()
                print(cat+' complete '+json.dumps({name:metrics(r['ranks']) for name,r in results.items()}),flush=True)
            del model,clip,native_head,proxy,gallery,reference,written,native_m,parts,old,new,anchor,q0,qzero,qnative,mt,mr,qr,q
            gc.collect();torch.cuda.empty_cache()
        verify_small();state.update(status='complete',seconds=time.monotonic()-start);save()
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--tests',type=Path)
    p.add_argument('--launch',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--gpu-uuid')
    args=p.parse_args()
    if args.prepare:prepare(args.root,args.tests)
    elif args.launch:launch(args.root)
    elif args.run:run(args.root,args.gpu_uuid)
    else:raise ValueError('Explicit 031 action required')
