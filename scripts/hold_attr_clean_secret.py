"""One-use hidden credential handoff for019; no secret files/argv/environment."""
import argparse,getpass,os,sys,time,json,hashlib,resource
from pathlib import Path
import subprocess

def main():
 p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);a=p.parse_args()
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 if not sys.stdin.isatty():raise RuntimeError('Interactive hidden input required')
 control=a.research/'attr_clean_019_control';control.mkdir(exist_ok=True)
 if (control/'credential_ready.json').exists():raise FileExistsError('Inspect existing credential broker first')
 key=getpass.getpass('DashScope API key (hidden; memory only): ').strip()
 if not key or len(key)>4000:raise ValueError('Invalid credential length')
 pid=os.fork()
 if pid:print(json.dumps({'credential_broker_pid':pid,'secret_persisted':False}),flush=True);return
 os.setsid()
 with open(os.devnull,'rb') as null,(control/'broker.log').open('xb',buffering=0) as log:
  os.dup2(null.fileno(),0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
 ready=control/'credential_ready.json';ready.write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],'status':'waiting_for_sealed_plan','secret_on_disk':False}))
 deadline=time.monotonic()+3600
 while not (control/'generation_plan.json').exists():
  if time.monotonic()>deadline or (control/'cancel.request').exists():key='';ready.write_text(json.dumps({'status':'expired_or_cancelled','secret_on_disk':False}));os._exit(0)
  time.sleep(2)
 root=Path(__file__).resolve().parents[1];plan=json.loads((control/'generation_plan.json').read_text())
 if plan['id']!='019_clean_attr_slots' or plan['cap_cny']!=1.5:raise ValueError('Wrong or excessive019 plan')
 for n,h in plan['source_hashes'].items():
  if hashlib.sha256((root/n).read_bytes()).hexdigest()!=h:raise ValueError('Sealed source changed')
 readfd,writefd=os.pipe();os.write(writefd,key.encode());os.close(writefd);key=''
 env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ};env.update(HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
 with (control/'pipeline.log').open('xb',buffering=0) as log:
  child=subprocess.Popen([sys.executable,'scripts/run_clean_attr_pipeline.py','--research',str(a.research),'--key-fd',str(readfd)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(readfd,),start_new_session=True)
 os.close(readfd);ready.write_text(json.dumps({'status':'handed_off_once','pipeline_pid':child.pid,'secret_on_disk':False}));os._exit(0)

if __name__=='__main__':main()
