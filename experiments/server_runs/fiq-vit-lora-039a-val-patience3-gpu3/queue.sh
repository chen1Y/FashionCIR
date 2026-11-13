#!/usr/bin/env bash
set -Eeuo pipefail
r=$1
cd /home/lcy/fashionCIR
py=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-eecc8534-68d6-406e-ebf0-638ceb761cc6
export PYTHONPATH="$r/code:/home/lcy/fashionCIR/src"
trap 'rc=$?; printf "%s 039A stopped exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
printf '%s starting 039A on GPU 3 with per-epoch official val and patience 3\n' "$(date -Is)"
nice -n 8 "$py" -u "$r/code/run_fiq_vit_lora_039a_earlystop.py" --root "$r" --gpu "$CUDA_VISIBLE_DEVICES" >"$r/039a.log" 2>&1
date -Is >"$r/queue.complete"
