"""019 one bounded generation then Shirt/Toptee trials; never resumes old queues."""
import argparse,json,os,subprocess,sys,time,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from experiment_storage import require_space
from generate_structured_edits import atomic_write_json
from run_attr_pair_batch import identity

def main():
 import fcntl
 p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);p.add_argument('--key-fd',type=int,required=True);a=p.parse_args();r=a.research;control=r/'attr_clean_019_control'
 lock=(r/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=json.loads((control/'generation_plan.json').read_text());state=dict(status='starting',identity=identity(os.getpid()),jobs=[],old_queue_resumed=False)
 atomic_write_json(control/'state.json',state);child=None;ci=None
 def guard():
  require_space(r)
  if (control/'cancel.request').exists():raise KeyboardInterrupt('User stopped019')
  for n,h in plan['source_hashes'].items():
   if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise ValueError('019 source changed')
 env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ};env.update(HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
 jobs=[('generation',[sys.executable,'src/generate_clean_attr_slots.py','--research',str(r),'--key-fd',str(a.key_fd)],1200)]+[(cat,[sys.executable,'src/train_clean_attr_alignment.py','--research',str(r),'--dataset',cat],3600) for cat in ('shirt','toptee')]
 try:
  for name,cmd,timeout in jobs:
   guard()
   if name!='generation':
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if gpu:raise RuntimeError('GPU belongs to an existing process; do not interfere')
   row=dict(id=name,status='running',command=cmd);state['jobs'].append(row);state['status']=name;start=time.monotonic()
   with (control/(name+'.log')).open('x') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(a.key_fd,) if name=='generation' else ())
    ci=identity(child.pid);row['identity']=ci;atomic_write_json(control/'state.json',state)
    if name=='generation':os.close(a.key_fd)
    while child.poll() is None:
     guard()
     if time.monotonic()-start>timeout:raise TimeoutError('019 '+name+' timed out')
     time.sleep(2)
   row.update(status='complete' if child.returncode==0 else 'failed',returncode=child.returncode,seconds=time.monotonic()-start);atomic_write_json(control/'state.json',state)
   if child.returncode:raise RuntimeError('019 '+name+' failed; inspect saved state, no automatic retry')
   child=None
  state['status']='complete';atomic_write_json(control/'state.json',state)
 except BaseException as e:
  if child and child.poll() is None:
   current=identity(child.pid)
   if current and ci and all(current[k]==ci[k] for k in ('pid','ppid','start_ticks','cwd')):
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  state.update(status='failed' if not isinstance(e,KeyboardInterrupt) else 'cancelled',error_type=type(e).__name__,error=str(e));atomic_write_json(control/'state.json',state);raise

if __name__=='__main__':main()
