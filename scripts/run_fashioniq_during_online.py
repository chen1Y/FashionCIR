"""Spend the existing FashionIQ pair budget while Shoes uses only the online API."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cir_gpu_window import hold_online_dispatcher, check_online_queue
from experiment_storage import require_space
from fashioniq_augmentation import validate, build_overlay, write_json
from fashioniq_noise import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shoes-results',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--limit',type=int,default=120)
    parser.add_argument('--min-free-gib',type=float,default=8)
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    if args.limit!=120 or args.min_free_gib<8:parser.error('Preserve the existing 120-pair budget and >=8 GiB reserve')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepared=args.output/'prepared_proposals.json';reviewed=args.output/'qwen_reviewed.json'
    manifest=json.loads(prepared.read_text());previous=json.loads((args.output/'followup_state.json').read_text())
    if (previous['status'] not in ('prepared','waiting_for_shoes','waiting_for_gpu')
            or previous['new_attempt_budget']!=args.limit or previous['model']!=args.model
            or Counter(x['decision'] for x in manifest['records'])!={'pending':120,'accept':20,'reject':4}):
        raise RuntimeError('Existing prepared state or prior sample counts changed; refusing a replacement batch')
    if any(p.exists() for p in (reviewed,args.output/'labeling.log',args.output/'overlay')):
        raise RuntimeError('Prior labeling/export exists; preserve and inspect instead of retrying')
    validate(args.root,manifest)
    require_space(args.output,args.min_free_gib)
    check_online_queue(args.shoes_results)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if gpu or free<22*1024:raise RuntimeError('GPU must be idle with >=22 GiB free before loading the existing 8B model')
    if args.detach:
        sys.stdout.flush();sys.stderr.flush();pid=os.fork()
        if pid:print(json.dumps({'fashioniq_supervisor_pid':pid}),flush=True);return
        os.setsid();signal.signal(signal.SIGHUP,signal.SIG_IGN)
        with open(os.devnull,'rb') as null,(args.output/'followup.log').open('ab',buffering=0) as log:
            os.dup2(null.fileno(),0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    (args.output/'followup.pid').write_text(str(os.getpid())+'\n')
    state=dict(previous,code_root=str(ROOT),scheduling_mode='during_shoes_online_generation',
               prepared_manifest_sha256=file_sha(prepared),gpu_hold=str(args.output/'gpu_hold.json'))
    def save(status,**extra):
        state.update(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        write_json(args.output/'followup_state.json',state);print(json.dumps(state),flush=True)
    def interrupted(signum,frame):raise KeyboardInterrupt('Supervisor termination requested')
    signal.signal(signal.SIGTERM,interrupted)
    child=None
    try:
        with hold_online_dispatcher(args.shoes_results,args.output/'gpu_hold.json'):
            save('labeling',shoes_dispatcher_held=True)
            command=[sys.executable,'-u','src/generate_fashioniq_pair_captions.py','--root',str(args.root),
                     '--manifest',str(prepared),'--output',str(reviewed),'--model',args.model,
                     '--guard-results',str(args.shoes_results),'--gpu-hold',str(args.output/'gpu_hold.json'),
                     '--limit',str(args.limit),'--min-free-gib',str(args.min_free_gib)]
            try:
                with (args.output/'labeling.log').open('x') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    save('labeling',labeling_pid=child.pid)
                    code=child.wait()
                if code:raise RuntimeError(f'FashionIQ labeling exited {code}; see preserved labeling.log')
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            require_space(args.output,args.min_free_gib)
            labeled=json.loads(reviewed.read_text());validate(args.root,labeled)
            if file_sha(prepared)!=state['prepared_manifest_sha256']:raise RuntimeError('Prepared manifest changed during labeling')
            before={ (r['candidate'],r['target']):r for r in manifest['records'] if r['decision']!='pending'}
            after={ (r['candidate'],r['target']):r for r in labeled['records'] if (r['candidate'],r['target']) in before}
            if before!=after:raise RuntimeError('Historical accepted/rejected records changed')
            counts=Counter(r['decision'] for r in labeled['records']);new=counts['accept']-20
            if counts['pending'] or new<1:raise RuntimeError('Incomplete labeling or no new accepted pairs')
            summary=build_overlay(args.root,labeled,args.output/'overlay')
            write_json(args.output/'labeling_summary.json',dict(new_attempts=120,new_accepted=new,new_rejected=counts['reject']-4,
                prior_accepted=20,decisions=dict(counts),summary=summary,
                note='Synthetic labels from two prompts of the same model; no retrieval training'))
        save('complete',new_accepted=new,new_rejected=counts['reject']-4,decisions=dict(counts),summary=summary,
             shoes_dispatcher_held=False,note='Synthetic labels; retrieval improvement untested. No retrieval training started.')
    except BaseException as exc:
        save('stopped',error=f'{type(exc).__name__}: {exc}',shoes_dispatcher_held=False)
        raise
    finally:
        lock.close()


if __name__=='__main__':main()
