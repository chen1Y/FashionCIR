#!/usr/bin/env bash
set -Eeuo pipefail
r=$PWD
py=/home/lcy/.cache/fashioncir_runtime_20261105/venv/bin/python
test ! -e "$r/state.json"
test ! -e "$r/queue.pid"
test ! -e "$r/cancel.request"
grep -q '^PASS:' "$r/dry_preflight.log"
export PYTHONPATH="$r/code:/home/lcy/fashionCIR/src"
"$py" - <<'PY'
from pathlib import Path
from run_fiq_native_caption_031 import read,sha,write
from fiq_context_subset_040c import load_pilot
root=Path.cwd()
plan=read(root/'plan.json')
for name,d in plan['source_sha256'].items():
    assert sha(root/'code'/name)==d,name
plan['pilot']['generation_sha256']=sha(root/'generation.json')
responses=load_pilot(root,plan)
write(root/'plan.json',plan)
write(root/'generation_verified.json',{'status':'verified','queries':len(responses),'sha256':plan['pilot']['generation_sha256']})
print('040C generation verified',len(responses),flush=True)
PY
nohup bash "$r/queue.sh" "$r" </dev/null >queue.log 2>&1 &
printf '%s\n' "$!" >queue.pid
sleep 1
ps -o pid,ppid,stat,etime,args -p "$(cat queue.pid)"
