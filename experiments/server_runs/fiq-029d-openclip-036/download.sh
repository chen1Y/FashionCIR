#!/usr/bin/env bash
set -Eeuo pipefail
r=$1
cd "$r/weights"
w=open_clip_pytorch_model.bin
part="$w.partial"
url=https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_pytorch_model.bin
test ! -e "$w"
trap 'rc=$?; printf "%s download failed exit=%s\n" "$(date -Is)" "$rc" >&2; exit "$rc"' ERR
curl --fail --location --connect-timeout 15 --max-time 2700 --retry 2 --retry-connrefused --continue-at - --output "$part" "$url"
test "$(stat -c %s "$part")" = 3944692325
printf '%s  %s\n' 9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4 "$part" | sha256sum --check -
mv "$part" "$w"
printf '%s\n' "$(date -Is)" > "$r/download.complete"
