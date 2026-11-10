"""Run only a sealed018 batch, without adopting/replaying the cancelled old queue."""
import argparse,json,os,subprocess,sys,time,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiment_storage import require_space
from generate_structured_edits import atomic_write_json

def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    stat=(p/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,ppid=int(stat[1]),start_ticks=int(stat[19]),cwd=str((p/'cwd').resolve()),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())

def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True);a=parser.parse_args();control=a.control.resolve();plan=json.loads((control/'plan.json').read_text());research=control.parent
    lock=(research/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if plan['api_calls']!=0 or plan['id']!='018_real_pair_alignment':raise ValueError('Wrong plan')
    if (control/'state.json').exists():raise FileExistsError('Inspect existing018 before any continuation')
    state=dict(status='running',identity=identity(os.getpid()),jobs=[],api_calls=0,old_queue_resumed=False)
    atomic_write_json(control/'state.json',state);child=None;child_identity=None
    def guard():
        require_space(research)
        if (control/'cancel.request').exists():raise KeyboardInterrupt('Stop only018')
        for n,h in plan['source_hashes'].items():
            if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise ValueError('Sealed source changed')
    env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ};env.update(HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    try:
        for job in plan['jobs']:
            guard();cmd=job['command'];destination=Path(cmd[cmd.index('--output')+1]).resolve()
            if destination.exists() or not destination.is_relative_to(research/'attr_pair_alignment_018'):raise ValueError('Wrong or existing output')
            if cmd[0]!=sys.executable or cmd[1]!='src/train_attr_pair_alignment.py' or '--preflight' in cmd:raise ValueError('Wrong executable')
            if not 0<job['max_seconds']<=3600:raise ValueError('Timeout outside authorization')
            row=dict(id=job['id'],status='running',command=cmd);state['jobs'].append(row);start=time.monotonic()
            with (control/(job['id']+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);child_identity=identity(child.pid);row['identity']=child_identity;atomic_write_json(control/'state.json',state)
                while child.poll() is None:
                    guard()
                    if time.monotonic()-start>job['max_seconds']:raise TimeoutError('018 timed out')
                    time.sleep(2)
            row.update(status='complete' if child.returncode==0 else 'failed',returncode=child.returncode,seconds=time.monotonic()-start);atomic_write_json(control/'state.json',state)
            if child.returncode:raise RuntimeError('018 failed, inspect outputs; no auto retry')
            result=json.loads((destination/'results.json').read_text())
            if result['status']!='complete' or result['api_calls']!=0:raise ValueError('Invalid completion')
            child=None
        state['status']='complete';atomic_write_json(control/'state.json',state)
    except BaseException as e:
        if child is not None and child.poll() is None:
            current=identity(child.pid)
            if current and child_identity and all(current[k]==child_identity[k] for k in ('pid','ppid','start_ticks','cwd')):
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(status='cancelled' if isinstance(e,KeyboardInterrupt) else 'failed',error_type=type(e).__name__,error=str(e));atomic_write_json(control/'state.json',state);raise

if __name__=='__main__':main()
