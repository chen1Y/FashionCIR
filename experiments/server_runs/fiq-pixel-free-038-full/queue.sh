#!/usr/bin/env bash
set -Eeuo pipefail
r=$1
cd /home/lcy/fashionCIR
py=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-3632a70e-2697-e833-7188-067e874f3e63
export PYTHONPATH="$r/code:/home/lcy/fashionCIR/src"
trap 'rc=$?; printf "%s queue stopped exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
timeout 120s "$py" - "$r" >"$r/cuda_smoke.log" 2>&1 <<'PY'
import pathlib,sys,torch
from torch.nn import functional as F
from fiq_pixel_free_038 import PixelFreeVisual, loss_for
from run_fiq_native_caption_031 import write
torch.manual_seed(42)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
device='cuda'
visual=PixelFreeVisual(1024,True,True).cuda()
gallery=F.normalize(torch.randn(32,1024,device=device),dim=-1)
labels=torch.arange(16,device=device)
references=gallery[16:]
text=F.normalize(torch.randn_like(references),dim=-1).requires_grad_()
drops=F.normalize(torch.randn_like(text),dim=-1)
active=torch.arange(16,device=device)<8
valid=torch.ones((16,4),device=device,dtype=torch.bool)
mask=torch.ones((16,32),device=device,dtype=torch.bool)
mask[labels,labels+16]=False
teacher=gallery[labels][:,None].expand(-1,4,-1)
optimizer=torch.optim.AdamW(visual.parameters(),lr=1e-4)
with torch.autocast('cuda',dtype=torch.bfloat16):
 query,proxy=visual(references,text,drops*active[:,None],active)
 out=dict(q=query,proxy=proxy,global_text=text,slots=text[:,None].expand(-1,4,-1),valid=valid)
 losses=loss_for(out,gallery,labels,mask,teacher,references,'full_learned_written',lambda_rec=.2,lambda_kl=.5,reference_weight=.2)
assert all(torch.isfinite(x) for x in losses.values())
losses['total'].backward()
assert torch.isfinite(text.grad).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in visual.parameters())
optimizer.step()
write(pathlib.Path(sys.argv[1])/'cuda_smoke.json',dict(status='passed',synthetic=True,batch=16,dimension=1024,
      peak_mib=torch.cuda.max_memory_allocated()/(1<<20),losses={k:float(v.detach()) for k,v in losses.items()}))
PY
for cat in dress shirt toptee; do
  printf '%s starting %s\n' "$(date -Is)" "$cat"
  timeout 12h nice -n 8 "$py" -u "$r/code/run_fiq_pixel_free_038.py" --root "$r" --category "$cat" --suite --gpu "$CUDA_VISIBLE_DEVICES" >"$r/$cat.log" 2>&1
  printf '%s finished %s\n' "$(date -Is)" "$cat"
done
date -Is >"$r/queue.complete"
