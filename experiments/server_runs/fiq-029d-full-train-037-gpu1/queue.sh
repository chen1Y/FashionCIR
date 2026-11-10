#!/usr/bin/env bash
set -Eeuo pipefail
r=$1
cd /home/lcy/fashionCIR
py=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-3632a70e-2697-e833-7188-067e874f3e63
export PYTHONPATH="$r/code:/home/lcy/fashionCIR/src"
trap 'rc=$?; printf "%s queue stopped exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
for cat in dress shirt toptee; do
  printf '%s starting %s\n' "$(date -Is)" "$cat"
  timeout 12h nice -n 8 "$py" -u "$r/code/run_029d_full_train_037.py" --root "$r" --category "$cat" --gpu "$CUDA_VISIBLE_DEVICES" >"$r/$cat.log" 2>&1
  printf '%s finished %s\n' "$(date -Is)" "$cat"
done
printf '%s\n' "$(date -Is)" > "$r/queue.complete"
