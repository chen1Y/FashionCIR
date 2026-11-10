"""API-only scoring fix around sealed017; reuse unchanged prepared features/settings."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import train_attr_slot_pilot as pilot
from attr_slot_fusion import AttrSlotCIR
from attr_slot_rank_math import candidate_ranks
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json


def evaluate(model,cache,indices,baseline=False):
    ranks=[];losses=[]
    if model is not None:model.eval()
    with torch.no_grad():
        for start in range(0,len(indices),4):
            ix=torch.tensor(indices[start:start+4],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):q=F.normalize(cache['v_cls'][ix]+cache['t_mod'][ix],dim=-1) if baseline else pilot.forward(model,cache,ix)['q_final']
            scores=torch.einsum('bd,bmd->bm',q.float(),cache['targets'][ix]);ranks.extend(candidate_ranks(scores,cache['mask'][ix],cache['original_order'][ix]).tolist())
            logits=scores.masked_fill(~cache['mask'][ix],-torch.inf)/.07
            losses.extend(F.cross_entropy(logits,torch.zeros(len(ix),device='cuda',dtype=torch.long),reduction='none').tolist())
    return dict(queries=len(indices),candidate_pool_r1=100*sum(r<=1 for r in ranks)/len(ranks),candidate_pool_r5=100*sum(r<=5 for r in ranks)/len(ranks),candidate_pool_r10=100*sum(r<=10 for r in ranks)/len(ranks),retrieval_nce=sum(losses)/len(losses),ranks=ranks)


def trace():
    return dict(reason='Torch2.9 stable argsort overload requires dim as keyword; original017 failed in baseline scoring before updates.',
        original_runner_sha256=file_sha(Path(pilot.__file__)),compatibility_runner_sha256=file_sha(__file__),rank_math_sha256=file_sha(Path(__file__).with_name('attr_slot_rank_math.py')),
        scientific_configuration_changed=False,prepared_feature_encoding_repeated=False)


def main():
    import sys
    pilot.evaluate=evaluate
    if '--compatibility-preflight' not in sys.argv:
        pilot.main();output=Path(sys.argv[sys.argv.index('--output')+1]);d=json.loads((output/'results.json').read_text());d['compatibility_evidence']=trace();atomic_write_json(output/'results.json',d);atomic_write_json(output/'compatibility_provenance.json',trace());return
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    for name in ('root','primary','research','output','cache'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task-sha256',required=True);p.add_argument('--compatibility-preflight',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    rows,offsets,hashes,base,evidence=pilot.metadata(a);cache=torch.load(a.cache,map_location='cpu',weights_only=True)
    if cache['evidence']!=evidence or cache['records']!=rows:raise ValueError('Unchanged prepared cache does not reproduce source binding')
    clip,_,tokenizer=pilot.load_clip(base);cache={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cache.items()};cache['patches']=cache['patches'].float()
    model=AttrSlotCIR(clip,tokenizer).cuda();baseline=evaluate(None,cache,offsets['dev'],baseline=True);initial=evaluate(model,cache,offsets['dev']);dev=pilot.dev_objective(model,cache,offsets['dev'])
    ix=torch.tensor(offsets['fit'][:4],device='cuda');o=pilot.forward(model,cache,ix);o['q_final'].mul(torch.randn_like(o['q_final'])).sum().backward()
    if not torch.isfinite(model.text.embedding.extra.grad).all() or any(p.grad is not None for p in clip.parameters()):raise ValueError('Gradient/frozen base check failed')
    report=dict(status='compatibility_preflight_passed',trace=trace(),cache_sha256=file_sha(a.cache),baseline=baseline,initial=initial,dev_nce=dev,optimizer_updates=0,api_calls=0)
    atomic_write_json(a.research/'attr_slot_017_preflight'/(a.dataset+'_compatibility.json'),report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
