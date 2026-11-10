"""Matched three-epoch image+slot training with145 new attribute-gate parameters."""
import argparse,json,gc,time
from pathlib import Path
import torch
import train_expanded_attr as engine
import native_expanded_attr_022 as native
from fiq_gate_patch_model_028 import GatedAttributeText,branch_forward,compose,original_compose
from train_attr_slot_pilot import load_clip
from experiment_storage import require_space,atomic_torch_save
from generate_structured_edits import atomic_write_json


def configure(root):
    engine.PairAlignedText=GatedAttributeText;engine.branch_forward=branch_forward;engine.compose=compose;native.PairAlignedText=GatedAttributeText;native.branch_forward=branch_forward;native.compose=compose
    weights=Path(json.loads((root/'plan.json').read_text())['weight_directory']);weights.mkdir(exist_ok=True)
    old_guard=engine.guard
    def guard(r,start):old_guard(r,start);require_space('/dev/shm',2)
    def save(state,path):
        target=weights/path.relative_to(root)
        if path.exists() or target.exists():raise FileExistsError('028 never overwrites a checkpoint')
        target.parent.mkdir(parents=True,exist_ok=True);atomic_torch_save(state,target);path.symlink_to(target)
    engine.guard=guard;engine.atomic_torch_save=save


def preflight(root,cat):
    torch.manual_seed(42);torch.set_num_threads(4);context=engine.load_context(root,cat,'train');clip,_,tok=load_clip(engine.base_path(cat));model=GatedAttributeText(clip,tok,2).cuda();model.eval();rows=context['rows'];b=context['banks']['fit'];ix=torch.tensor(b['indices'][:8],device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=branch_forward(model,[rows[i] for i in ix.tolist()],'slots');original=original_compose(out,context['reference'][ix],'slots');gated=compose(out,context['reference'][ix],'slots')
    delta=float((original-gated).abs().max())
    if delta>1e-6:raise ValueError('Uniform gate failed to reproduce baseline query')
    result=engine.evaluate(model,context,'dev','slots');baseline=json.loads((Path('/data/8T1/lcy/CIRData/runs/fiq-expanded-023')/'fits'/cat/'slots/results.json').read_text())['history'][0]['metrics']
    if result['ranks']!=baseline['ranks']:raise ValueError('028 initial uniform gate dev ranks differ from old initial baseline')
    atomic_write_json(root/('initial_gate_'+cat+'.json'),dict(status='passed',batch=8,max_query_delta=delta,all_initial_dev_ranks_exact=True,gate_parameters=145,weights_discarded=True,api_calls=0));del model,clip,context;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',choices=('shirt','toptee'),required=True);a=p.parse_args();configure(a.root);preflight(a.root,a.category);native.smoke(a.root,a.category,'slots');gc.collect();torch.cuda.empty_cache();engine.fit(a.root,a.category,'slots')
