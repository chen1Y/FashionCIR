#!/usr/bin/env bash
set -Eeuo pipefail
r=$1
gpu=$2
cd /home/lcy/fashionCIR
py=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$r/code:/home/lcy/fashionCIR/src"
trap 'rc=$?; printf "%s caption-only queue stopped exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
printf '%s starting caption-only training on %s\n' "$(date -Is)" "$gpu"
nice -n 8 "$py" -u "$r/code/run_fiq_vit_lora_039a_earlystop.py" --root "$r" --gpu "$gpu" >"$r/train.log" 2>&1
date -Is >"$r/queue.complete"
