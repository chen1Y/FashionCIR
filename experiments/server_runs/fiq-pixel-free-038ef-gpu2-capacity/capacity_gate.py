"""One-shot GPU 2 submission after a currently observed compute job exits."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

root = Path(sys.argv[1])
plan = json.loads((root / 'plan.json').read_text())
config = plan['capacity_gate']
deadline = time.monotonic() + config['max_wait_seconds']

def state(status, **values):
    path = root / 'gate_state.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(status=status, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                    gpu_index=2, **values), indent=2) + '\n')
    temp.replace(path)

def query(kind, fields):
    return subprocess.check_output(['nvidia-smi', '-i', '2', f'--query-{kind}={fields}',
                                    '--format=csv,noheader,nounits'], text=True, timeout=8).strip()

def snapshot():
    index, uuid, free, util = (s.strip() for s in query('gpu','index,uuid,memory.free,utilization.gpu').split(','))
    if int(index) != 2 or uuid != plan['gpu']:
        raise RuntimeError('GPU identity changed')
    apps = {int(pid): int(used) for pid, used in (line.split(',') for line in query('compute-apps','pid,used_gpu_memory').splitlines())}
    return dict(free_mib=int(free), utilization=int(util), apps=apps)

def enough(pair):
    first, second = pair
    return (all(x['free_mib'] >= config['min_free_mib'] and x['utilization'] <= config['max_utilization_percent'] for x in pair)
            and all(pid in first['apps'] and mem <= first['apps'][pid] + 512 for pid, mem in second['apps'].items()))

fds = {}
try:
    first = snapshot()
    for pid in first['apps']:
        try:
            fds[os.pidfd_open(pid)] = pid
        except ProcessLookupError:
            pass
    while True:
        if (root / 'cancel.request').exists():
            raise InterruptedError('capacity gate cancelled')
        first = snapshot()
        time.sleep(5)
        pair = [first, snapshot()]
        if enough(pair):
            with (root / 'queue.log').open('x') as log:
                child = subprocess.Popen(['bash', str(root / 'queue.sh'), str(root)],
                                         cwd='/home/lcy/fashionCIR', stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            (root / 'queue.pid').write_text(str(child.pid) + '\n')
            state('launched', queue_pid=child.pid, snapshots=pair)
            break
        if not fds:
            state('capacity_unavailable', snapshots=pair)
            break
        state('waiting_for_gpu_2_capacity', watched_compute_pids=list(fds.values()), snapshots=pair)
        poller = select.poll()
        for fd in fds:
            poller.register(fd, select.POLLIN)
        timeout_ms = max(0, int((deadline - time.monotonic()) * 1000))
        events = poller.poll(timeout_ms)
        if not events:
            state('capacity_timeout', snapshots=pair)
            break
        for fd, _ in events:
            os.close(fd)
            del fds[fd]
except Exception as exc:
    state('gate_failed', error=type(exc).__name__, detail=str(exc))
    raise
finally:
    for fd in fds:
        os.close(fd)
