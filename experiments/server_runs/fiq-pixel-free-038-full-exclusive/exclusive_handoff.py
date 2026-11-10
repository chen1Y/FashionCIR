"""One-shot handoff after the already-running Dress child finishes."""
import hashlib
import json
import os
import pathlib
import select
import signal
import subprocess
import sys
import time

old, root = map(pathlib.Path, sys.argv[1:])
status = root / 'handoff_state.json'

def read(path):
    return json.loads(path.read_text())

def write(**data):
    tmp = status.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(status)

def proc(pid):
    path = pathlib.Path('/proc') / str(pid)
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ppid=int(fields[1]), start_ticks=int(fields[19]), stat=fields[0],
                argv=(path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                cwd=str((path / 'cwd').resolve(strict=True)))

def wait_for_exit(fd, timeout_ms):
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    if not poller.poll(timeout_ms):
        raise TimeoutError('038 Dress boundary did not arrive within the handoff window')

stopped = False
old_fd = None
evidence = {}
try:
    state = read(old / 'dress/state.json')
    assert state['status'] == 'running'
    assert not (old / 'shirt').exists() and not (old / 'toptee').exists()
    ident = state['identity']
    worker = proc(ident['pid'])
    watcher = proc(worker['ppid'])
    scheduler = proc(int((old / 'queue.pid').read_text()))
    assert worker['start_ticks'] == ident['start_ticks'] and worker['ppid'] == ident['ppid']
    assert worker['cwd'] == watcher['cwd'] == scheduler['cwd'] == '/home/lcy/fashionCIR'
    assert worker['argv'][2:] == [str(old / 'code/run_fiq_pixel_free_038.py'), '--root', str(old),
                                  '--category', 'dress', '--suite', '--gpu', state['gpu']]
    assert watcher['argv'][0] == 'timeout' and watcher['argv'][-len(worker['argv']):] == worker['argv']
    assert watcher['ppid'] == scheduler['pid'] and scheduler['argv'] == ['bash', str(old / 'queue.sh'), str(old)]
    old_fd = os.pidfd_open(scheduler['pid'])
    watch_fd = os.pidfd_open(watcher['pid'])
    worker_fd = os.pidfd_open(worker['pid'])
    assert all(proc(p['pid'])['start_ticks'] == p['start_ticks'] for p in (worker, watcher, scheduler))
    evidence = dict(old_scheduler=scheduler, dress_worker=worker, dress_timeout=watcher, root=str(root))
    signal.pidfd_send_signal(old_fd, signal.SIGSTOP)
    stopped = True
    for _ in range(100):
        if proc(scheduler['pid'])['stat'] in ('T', 't'):
            break
        time.sleep(.02)
    else:
        raise RuntimeError('038 scheduler did not suspend')
    assert not (old / 'shirt').exists() and not (old / 'toptee').exists()
    write(status='waiting_dress_boundary', **evidence)
    wait_for_exit(worker_fd, 90 * 60 * 1000)
    wait_for_exit(watch_fd, 30000)
    state = read(old / 'dress/state.json')
    result = read(old / 'dress/dress_results.json')
    assert state['status'] == 'complete' and state['identity']['start_ticks'] == worker['start_ticks']
    assert (result['queries'], result['gallery']) == (2017, 3817)
    assert set(result['rows']) == {'038B_clean', '038C_erase', '038D_proxy', '038D_proxy_patch'}
    assert not (old / 'shirt').exists() and not (old / 'toptee').exists()
    evidence['dress_results_sha256'] = hashlib.sha256((old / 'dress/dress_results.json').read_bytes()).hexdigest()
    (root / 'dress').symlink_to(old / 'dress', target_is_directory=True)
    signal.pidfd_send_signal(old_fd, signal.SIGKILL)
    stopped = False
    wait_for_exit(old_fd, 30000)
    with (root / 'queue.log').open('x') as log:
        child = subprocess.Popen(['bash', str(root / 'queue.sh'), str(root)], cwd='/home/lcy/fashionCIR',
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    (root / 'queue.pid').write_text(str(child.pid) + '\n')
    write(status='exclusive_queue_launched', queue_pid=child.pid, **evidence)
except BaseException as error:
    if stopped:
        try:
            signal.pidfd_send_signal(old_fd, signal.SIGCONT)
        except ProcessLookupError:
            pass
    write(status='failed', error=type(error).__name__, detail=str(error), **evidence)
    raise
