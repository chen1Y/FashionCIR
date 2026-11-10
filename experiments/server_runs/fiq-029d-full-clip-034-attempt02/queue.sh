#!/usr/bin/env bash
set -Eeuo pipefail
root=$1
cd /home/lcy/fashionCIR
runtime=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-eecc8534-68d6-406e-ebf0-638ceb761cc6
export PYTHONPATH="$root/code:/home/lcy/fashionCIR/src"
trap 'rc=$?; printf "%s queue failed exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
python3 - "$root" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
plan=json.loads((p/'plan.json').read_text())
s=json.loads((p/'dress_smoke/smoke.json').read_text())
assert s['status']=='passed' and s['updates']==1
assert s['vision_conv_change']>0 and s['original_vocab_gradient_norm']>0
for name,digest in plan['source_files'].items():
    assert hashlib.sha256((p/'code'/name).read_bytes()).hexdigest()==digest,name
PY
for category in dress shirt toptee; do
  printf '%s starting %s\n' "$(date -Is)" "$category"
  timeout 4h "$runtime" -u "$root/code/run_029d_full_clip_034.py" --root "$root" --category "$category" --gpu "$CUDA_VISIBLE_DEVICES" >"$root/$category.log" 2>&1
  printf '%s finished %s\n' "$(date -Is)" "$category"
done
printf '%s\n' "$(date -Is)" > "$root/queue.complete"
