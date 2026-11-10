#!/usr/bin/env bash
set -Eeuo pipefail
r=$1
cd /home/lcy/fashionCIR
py=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-eecc8534-68d6-406e-ebf0-638ceb761cc6
export PYTHONPATH="$r/code:/home/lcy/fashionCIR/src"
trap 'rc=$?; printf "%s queue stopped exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
printf '%s batch32 smoke\n' "$(date -Is)"
timeout 20m "$py" -u "$r/code/run_029d_inbatch_035.py" --root "$r" --category dress --gpu "$CUDA_VISIBLE_DEVICES" --smoke >"$r/smoke.log" 2>&1
test -s "$r/dress_smoke/smoke.json"
for cat in dress shirt toptee; do
  printf '%s starting %s\n' "$(date -Is)" "$cat"
  timeout 2h "$py" -u "$r/code/run_029d_inbatch_035.py" --root "$r" --category "$cat" --gpu "$CUDA_VISIBLE_DEVICES" >"$r/$cat.log" 2>&1
  printf '%s finished %s\n' "$(date -Is)" "$cat"
done
printf '%s\n' "$(date -Is)" > "$r/queue.complete"
