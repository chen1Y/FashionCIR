"""Temporarily hold only the Shoes dispatcher while its API child keeps running."""
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from fashioniq_augmentation import write_json


def process_info(pid):
    root = Path('/proc') / str(pid)
    try:
        fields = (root/'stat').read_text().split(') ', 1)[1].split()
        args = (root/'cmdline').read_bytes().split(b'\0')
        command = [x.decode() for x in args if x]
        return dict(pid=pid, state=fields[0], start_ticks=fields[19],
                    cwd=os.readlink(root/'cwd'), command=command,
                    command_sha256=hashlib.sha256(json.dumps(command).encode()).hexdigest())
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(current, expected):
    return bool(current) and all(current.get(k)==expected.get(k)
                                for k in ('pid','start_ticks','cwd','command_sha256'))


def check_online_queue(results):
    results = Path(results)
    report = json.loads((results/'experiment_report.json').read_text())
    if (report.get('status') not in ('qwen_train','qwen_val')
            or report.get('config',{}).get('generation_provider')!='online-qwen37'):
        raise RuntimeError('Shoes is not in its online-only generation window')
    pid = int((results/'queue.pid').read_text())
    info = process_info(pid)
    if not info or info['state']=='Z':
        raise RuntimeError('Shoes dispatcher is not live')
    args = info['command']
    expected = json.loads((results/'switch_to_online_20261016/online_start_snapshot.json').read_text())['code_root']
    if (not any(Path(arg).name=='run_cirr_shoes_benchmark.py' for arg in args)
            or '--dataset' not in args or args[args.index('--dataset')+1]!='shoes'
            or '--generation-provider' not in args
            or args[args.index('--generation-provider')+1]!='online-qwen37'
            or info['cwd']!=expected):
        raise RuntimeError('Shoes dispatcher identity mismatch')
    return info


def verify_hold(path, results):
    record = json.loads(Path(path).read_text())
    if record.get('status')!='held' or Path(record['results']).resolve()!=Path(results).resolve():
        raise RuntimeError('Invalid or released GPU-window proof')
    queue = check_online_queue(results)
    if not same_process(queue, record['queue']) or queue['state'] not in ('T','t'):
        raise RuntimeError('Shoes dispatcher is not held with the recorded identity')
    owner = process_info(record['owner']['pid'])
    if (not same_process(owner,record['owner']) or owner['state'] in ('T','t','Z')
            or not any(Path(x).name=='run_fashioniq_during_online.py' for x in owner['command'])):
        raise RuntimeError('GPU-window supervisor is not live')
    return record


@contextmanager
def hold_online_dispatcher(results, proof):
    import fcntl
    results, proof = Path(results).resolve(), Path(proof).resolve()
    lock = (results/'gpu_window.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue = check_online_queue(results)
    if queue['state'] in ('T','t'):
        raise RuntimeError('Dispatcher was already stopped; refusing to claim another hold')
    owner = process_info(os.getpid())
    record = dict(status='requesting',results=str(results),queue=queue,owner=owner,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  purpose='User-authorized FashionIQ labeling during Shoes online generation')
    paused = False
    try:
        write_json(proof,record)
        os.kill(queue['pid'],signal.SIGSTOP)
        paused = True
        for _ in range(50):
            current = process_info(queue['pid'])
            if same_process(current,queue) and current['state'] in ('T','t'):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Dispatcher did not enter held state')
        if not same_process(check_online_queue(results),queue):
            raise RuntimeError('Queue phase/identity changed while acquiring GPU window')
        record['status']='held'
        write_json(proof,record)
        verify_hold(proof,results)
        yield record
    finally:
        current = process_info(queue['pid'])
        if paused and same_process(current,queue) and current['state'] in ('T','t'):
            os.kill(queue['pid'],signal.SIGCONT)
            record['release_result']='same_dispatcher_resumed'
        else:
            record['release_result']='no_matching_stopped_dispatcher'
        record.update(status='released',released_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(proof,record)
        lock.close()
