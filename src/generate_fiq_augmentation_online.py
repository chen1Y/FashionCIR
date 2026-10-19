"""Metered Qwen3.7-Flash supplement for the fixed 101 augmented Dress queries."""
import argparse
import concurrent.futures
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

from experiment_storage import require_space
from generate_shoes_online import Ledger, RateLimiter
from shoes_online_pilot import Client, BASE_URL, MODEL, reference_data_url
from generate_structured_edits import (SYSTEM_PROMPT, build_user_prompt, load_fashioniq_queries,
    extract_json_object, validate_edit_program, process_semantics, make_output_sample, atomic_write_json)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--background',action='store_true')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records=load_fashioniq_queries(args.root,'dress','train')[5985:]
    if len(records)!=101: raise ValueError('Expected exactly the 101 locked augmentation queries')
    config=dict(model=MODEL,input_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
        prompt_sha256=hashlib.sha256((SYSTEM_PROMPT+build_user_prompt('')).encode()).hexdigest(),
        protocol='fiq-online-101-v1',max_returned_invalid_repairs=1,concurrency=4,rpm=120,
        validation='Legacy FashionIQ schema and deterministic program-consistency checks; not model/human GT')
    key=getpass.getpass('DashScope API key (memory only): ')
    if not key:raise ValueError('Empty API key')
    if args.background:
        sys.stdout.flush();sys.stderr.flush();pid=os.fork()
        if pid:
            print('online generation PID',pid,flush=True);return
        os.setsid()
        fd=os.open(args.output.parent/'online_structures.log',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
        fd=os.open('/dev/null',os.O_RDONLY);os.dup2(fd,0);os.close(fd)
    (args.output.parent/'generation.pid').write_text(str(os.getpid())+'\n')
    ledger=Ledger(args.output.parent/'fiq_online_requests.sqlite3',config,max_cny=1,max_calls=202)
    stop=threading.Event();rate=RateLimiter(120,stop);local=threading.local()
    stored={s['source_index']:s for s in ledger.samples('train')}
    payload=dict(model=MODEL,category='dress',split='train',query_only_generation=True,protocol=config,samples=[],state='running')

    def ask(row,phase,image,prompt):
        cached=ledger.begin('train',row['source_index'],phase)
        if cached is not None:return cached
        if not hasattr(local,'client'):local.client=Client(key,BASE_URL,ledger)
        rate.wait()
        try:
            response=local.client.ask(image,prompt,1024);ledger.finish(response);return response
        except Exception:
            ledger.abandon();stop.set();raise

    def process(row):
        require_space(args.output)
        program=None;errors=[];raw=None;semantic=None;responses=[]
        try:
            image=reference_data_url(row['reference_image'])
            prompt=SYSTEM_PROMPT+'\n'+build_user_prompt(row['modification_text'])
            for attempt in range(2):
                answer=ask(row,'generate' if attempt==0 else 'repair_known_invalid',image,prompt)
                responses.append(answer);raw=answer['text']
                try:program=extract_json_object(raw);errors=validate_edit_program(program)
                except ValueError as e:errors=[str(e)]
                if not errors:
                    program,semantic,_=process_semantics(row,program,SimpleNamespace(semantic_repair_retries=0))
                    errors=semantic['errors']
                if not errors:break
                prompt=(SYSTEM_PROMPT+'\n'+build_user_prompt(row['modification_text'])+
                        '\nCorrect only this returned invalid JSON program: '+json.dumps(program)+
                        '\nValidation errors: '+'; '.join(errors)+'. Return full corrected JSON.')
        except Exception as e:
            errors=[type(e).__name__+': '+(local.client.safe_text(str(e)) if hasattr(local,'client') else 'request stopped')]
        sample=make_output_sample(row,program,raw,errors,False)
        sample['query_index']=row['source_index'];sample['generator_provider']='qwen37-api'
        sample['api_responses']=responses
        if semantic is not None:sample['semantic_validation']=semantic
        ledger.save_sample('train',sample)
        return sample

    def save():
        payload['samples']=[stored[k] for k in sorted(stored)]
        payload['cost']=ledger.summary();payload['completed_queries']=len(stored)
        payload['valid_queries']=sum(s.get('status')=='ok' and s.get('semantic_validation',{}).get('valid') is True for s in stored.values())
        atomic_write_json(args.output,payload)

    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(process,row):row for row in records if row['source_index'] not in stored}
        for future in concurrent.futures.as_completed(futures):
            if future.cancelled():continue
            sample=future.result();stored[sample['source_index']]=sample;save()
            print(json.dumps(dict(done=len(stored),valid=payload['valid_queries'],cost=ledger.summary())),flush=True)
            if stop.is_set():
                for pending in futures:pending.cancel()
    payload['state']='complete' if len(stored)==101 else 'stopped';save()
    print('FINAL',payload['state'],payload['completed_queries'],payload['valid_queries'],flush=True)


if __name__=='__main__':main()
