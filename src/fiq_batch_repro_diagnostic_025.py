"""Bounded025 numerical diagnostic; no fit, API, retrieval or cache publication."""
import argparse,json,math,time
from pathlib import Path
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

SOURCE=('src/fiq_batch_repro_diagnostic_025.py','src/train_attr_slot_pilot.py','src/fashioniq_noise.py','src/experiment_storage.py','src/generate_structured_edits.py','tests/test_fiq_batch_repro_025.py')


def fixed_batches(conflicts):
    if len(conflicts)!=1:raise ValueError('Exactly one metadata-defined batch-size conflict is required; no scanning')
    c=conflicts[0];old=c['old_batch_ids'];new=c['new_batch_ids'];image=c['image_id']
    if len(old)!=3 or len(new)!=8 or image not in old or image not in new:raise ValueError('Unexpected fixed diagnostic geometry')
    return image,[dict(name='original_batch3',ids=old),dict(name='failed_window_batch8',ids=new),dict(name='original_batch3_repeat',ids=old),dict(name='original_batch3_padded8',ids=old+[old[0]]*5)]


def comparison(a,b):
    if len(a)!=len(b) or not a or any(not math.isfinite(x) for x in a+b):raise ValueError('Invalid diagnostic vectors')
    differences=[x-y for x,y in zip(a,b)];maximum=max(abs(x) for x in differences)
    return dict(max_abs_delta=maximum,l2_delta=math.sqrt(sum(x*x for x in differences)),within_existing_1e4=maximum<=1e-4,exact=all(x==0 for x in differences))


def prepare(root,tests,code):
    p=root/'plan.json'
    if p.exists():raise FileExistsError('025 diagnostic plan is immutable')
    if not tests.read_text().rstrip().endswith('OK'):raise ValueError('025 tests failed')
    audit=json.loads((root/'metadata_audit.json').read_text())
    if audit['status']!='metadata_verified_no_forward':raise ValueError('Metadata preflight missing')
    image,batches=fixed_batches(audit['old_new_batch_size_conflict']);validation=Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023');report=json.loads((validation/'assets_shirt_train.json').read_text());metadata=json.loads((validation/'metadata/shirt_train.json').read_text());data=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/resized_image');base=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_shirt_toptee_20261022/shirt/checkpoints/shirt_dqu_original_seed42_best.pt')
    if file_sha(base)!=report['base_sha256'] or file_sha(validation/'metadata/shirt_train.json')!=report['metadata_sha256']:raise ValueError('Original evidence changed')
    ids=list(dict.fromkeys(i for b in batches for i in b['ids']));hashes={i:file_sha(data/(i+'.jpg')) for i in ids}
    if any(hashes[i]!=metadata['image_sha256'][i] for i in ids):raise ValueError('JPEG changed')
    plan=dict(id='025_fixed_batch_repro_diagnostic',source_hashes={n:file_sha(code/n) for n in SOURCE},metadata_audit_sha256=file_sha(root/'metadata_audit.json'),test_sha256=file_sha(tests),base_path=str(base),base_sha256=report['base_sha256'],cache_path=report['cache_path'],cache_sha256=report['cache_sha256'],image_sha256=hashes,probe_image=image,batches=batches,max_image_forwards=22,unique_images=len(ids),max_seconds=3600,api_calls=0,fit_updates=0,official_queries=0,threshold_unchanged=1e-4,precision='Existing BF16 autocast, FP32-normalized global; identical frozen ViT-H/14',no_cache_or_checkpoint_writes=True,no_automatic_full_continuation=True)
    atomic_write_json(p,plan);(root/'tests.log').write_text(tests.read_text());print(json.dumps(dict(status='sealed',probe=image,image_forwards=22,unique_images=len(ids),api_calls=0)))


def run(root,code):
    import os,subprocess
    import torch
    from PIL import Image
    from torch.nn import functional as F
    from experiment_storage import require_space
    from train_attr_slot_pilot import load_clip
    from datetime import datetime,timezone
    start=time.monotonic();plan=json.loads((root/'plan.json').read_text())
    if (root/'results.json').exists() or (root/'state.json').exists():raise FileExistsError('No blind diagnostic restart')
    def guard():
        require_space('/home/lcy',2);require_space('/data/8T1/lcy/CIRData/autodl-tmp',8);require_space('/dev/shm',2)
        if (root/'cancel.request').exists():raise KeyboardInterrupt('025 cancelled')
        if time.monotonic()-start>plan['max_seconds']:raise TimeoutError('025 exceeded1h')
        if file_sha(root/'metadata_audit.json')!=plan['metadata_audit_sha256']:raise ValueError('025 metadata preflight changed')
        for n,h in plan['source_hashes'].items():
            if file_sha(code/n)!=h:raise ValueError('025 sealed source changed')
    guard()
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied; do not interfere')
    p=Path('/proc')/str(os.getpid());fields=(p/'stat').read_text().split(') ',1)[1].split();identity=dict(pid=os.getpid(),ppid=os.getppid(),cwd=str((p/'cwd').resolve()),start_ticks=int(fields[19]),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    atomic_write_json(root/'state.json',dict(status='running',identity=identity,api_calls=0,fit_updates=0))
    try:
        assert file_sha(plan['base_path'])==plan['base_sha256'] and file_sha(plan['cache_path'])==plan['cache_sha256']
        cache=torch.load(plan['cache_path'],map_location='cpu',weights_only=True,mmap=True);index=cache['ids'].index(plan['probe_image']);cached=cache['gallery'][index].clone().tolist();del cache
        torch.set_num_threads(4);clip,transform,_=load_clip(Path(plan['base_path']));clip.visual.output_tokens=True
        if any(v.requires_grad for v in clip.parameters()) or clip.training:raise ValueError('025 model must be fully frozen/eval')
        images={}
        for i,h in plan['image_sha256'].items():
            path=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/resized_image')/(i+'.jpg')
            if file_sha(path)!=h:raise ValueError('025 image source changed')
            with Image.open(path) as im:images[i]=transform(im.convert('RGB'))
        vectors={'cached':cached};forwards=0;shapes=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for batch in plan['batches']:
                guard();x=torch.stack([images[i] for i in batch['ids']]).cuda();global_,_=clip.visual(x);global_=F.normalize(global_.float(),dim=-1).cpu();position=batch['ids'].index(plan['probe_image']);vectors[batch['name']]=global_[position].tolist();forwards+=len(batch['ids']);shapes.append(dict(name=batch['name'],shape=list(x.shape),position=position))
        assert forwards==plan['max_image_forwards']==22
        pairs=[('original_batch3','cached'),('failed_window_batch8','cached'),('original_batch3_repeat','original_batch3'),('original_batch3_padded8','cached'),('original_batch3_padded8','original_batch3'),('failed_window_batch8','original_batch3_padded8')]
        comparisons={a+'_vs_'+b:comparison(vectors[a],vectors[b]) for a,b in pairs}
        guard();assert file_sha(plan['cache_path'])==plan['cache_sha256'] and file_sha(plan['base_path'])==plan['base_sha256']
        result=dict(status='complete',plan_sha256=file_sha(root/'plan.json'),completed_utc=datetime.now(timezone.utc).isoformat(),probe_image=plan['probe_image'],image_forwards=forwards,api_calls=0,fit_updates=0,official_queries=0,model_fully_frozen=True,old_cache_sha_unchanged=True,base_sha_unchanged=True,comparisons=comparisons,vectors=vectors,batch_shapes=shapes,backend=dict(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic),seconds=time.monotonic()-start,scope='One metadata-defined batch-size conflict; not a complete replay of the failed024 run and no retrieval result')
        atomic_write_json(root/'results.json',result);atomic_write_json(root/'state.json',dict(status='complete',identity=identity,api_calls=0,fit_updates=0,image_forwards=forwards));print(json.dumps({k:v for k,v in result.items() if k not in ('vectors','backend')}))
    except BaseException as exc:
        atomic_write_json(root/'state.json',dict(status='failed',identity=identity,error_type=type(exc).__name__,error=str(exc),api_calls=0,fit_updates=0));raise


if __name__=='__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--tests',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args();code=Path(__file__).resolve().parents[1]
    if a.prepare:prepare(a.root,a.tests,code)
    elif a.run:run(a.root,code)
    else:raise ValueError('Explicit prepare or run required')
