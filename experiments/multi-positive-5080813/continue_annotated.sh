#!/usr/bin/env bash
set -euo pipefail
cd /home/lcy/fashionCIR/experiments/multi-positive-5080813
while kill -0 2011 2>/dev/null; do sleep 5; done
grep -q 'INFO epoch=6 metrics=' /data/8T1/lcy/CIRData/autodl-tmp/results/multi_positive_pilot_20261014/off_seed42.log
exec python -u scripts/run_multi_positive_pilot.py --fashioniq-path /data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ --clip-checkpoint /home/lcy/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_pytorch_model.bin --dqu-checkpoint /data/8T1/lcy/CIRData/autodl-tmp/checkpoints/dress_dqu_official_original_seed42_best.pt --output-dir /data/8T1/lcy/CIRData/autodl-tmp/results/multi_positive_pilot_20261014/annotated_stage --num-epochs 6 --modes annotated
