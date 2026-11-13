#!/usr/bin/env bash
set -Eeuo pipefail
r=$1
cd /home/lcy/fashionCIR
py=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-d776047c-8853-5dd8-e1f1-cee8065e553b
export PYTHONPATH="$r/code:/home/lcy/fashionCIR/src"
trap 'rc=$?; printf "%s queue stopped exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
timeout 120s "$py" - "$r" >"$r/cuda_smoke.log" 2>&1 <<'PY'
import pathlib,sys,torch
from torch.nn import functional as F
from fiq_pixel_free_038 import gallery_kl
from run_fiq_native_caption_031 import write
torch.manual_seed(42)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.cuda.set_per_process_memory_fraction(.42)
gallery=F.normalize(torch.randn(17908,1024,device='cuda'),dim=-1)
query=F.normalize(torch.randn(16,1024,device='cuda'),dim=-1).requires_grad_()
labels=torch.arange(16,device='cuda')
mask=torch.ones(16,len(gallery),device='cuda',dtype=torch.bool)
mask[:,32:128]=False
with torch.autocast('cuda',dtype=torch.bfloat16):
 value=gallery_kl(query,gallery,labels,mask)
value.backward()
assert value.dtype==torch.float32 and torch.isfinite(value) and torch.isfinite(query.grad).all()
write(pathlib.Path(sys.argv[1])/'cuda_smoke.json',dict(status='passed',synthetic=True,batch=16,
      candidates=len(gallery),kl=float(value.detach()),peak_mib=torch.cuda.max_memory_allocated()/2**20))
PY
for cat in toptee; do
  printf '%s starting 038E %s\n' "$(date -Is)" "$cat"
  timeout 12h nice -n 8 "$py" -u "$r/code/run_fiq_pixel_free_038.py" --root "$r" --category "$cat" --variant 038E --gpu "$CUDA_VISIBLE_DEVICES" >"$r/038E_$cat.log" 2>&1
  printf '%s finished 038E %s\n' "$(date -Is)" "$cat"
done
printf '%s starting 038F joint\n' "$(date -Is)"
timeout 12h nice -n 8 "$py" -u "$r/code/run_fiq_pixel_free_038.py" --root "$r" --joint_categories dress,shirt,toptee --variant 038F --gpu "$CUDA_VISIBLE_DEVICES" >"$r/038F_joint.log" 2>&1
printf '%s finished 038F joint\n' "$(date -Is)"
date -Is >"$r/queue.complete"
