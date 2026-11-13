#!/usr/bin/env bash
set -Eeuo pipefail
r=$1
cd /home/lcy/fashionCIR
py=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-a690c04f-a547-1ca5-5288-5dc8ff6097ea
export PYTHONPATH="$r/code:/home/lcy/fashionCIR/src"
trap 'rc=$?; printf "%s queue stopped exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
printf '%s starting 038G joint and Dress-weight ablation\n' "$(date -Is)"
timeout 12h nice -n 8 "$py" -u "$r/code/run_fiq_pixel_free_038.py" --root "$r" --joint_categories dress,shirt,toptee --variant 038G --gpu "$CUDA_VISIBLE_DEVICES" >"$r/038G_joint.log" 2>&1
printf '%s finished 038G joint and Dress-weight ablation\n' "$(date -Is)"
date -Is >"$r/queue.complete"
