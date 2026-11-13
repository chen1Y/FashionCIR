#!/usr/bin/env bash
set -Eeuo pipefail
r=$1
cd /home/lcy/fashionCIR
py=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-a690c04f-a547-1ca5-5288-5dc8ff6097ea
export PYTHONPATH="$r/code:/home/lcy/fashionCIR/src"
trap 'rc=$?; printf "%s 039 restart stopped exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
printf '%s starting 039 restart, A then B\n' "$(date -Is)"
timeout 24h nice -n 8 "$py" -u "$r/code/run_fiq_vit_lora_039.py" --root "$r" --gpu "$CUDA_VISIBLE_DEVICES" >"$r/039.log" 2>&1
date -Is >"$r/queue.complete"
