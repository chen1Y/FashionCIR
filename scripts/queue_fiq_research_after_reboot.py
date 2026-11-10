"""Audited adoption of an idle, entirely completed ledger after a server reboot."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from queue_fiq_research import identity,now,stop
from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json


def check_completed_ledger(state,plans):
    if state['status']!='awaiting_next_research_plan':raise ValueError('Adopt only idle completed ledger')
    completed=state['completed_plans'];jobs=state['jobs']
    if len(completed)!=len(set(completed)) or set(completed)!=set(plans):raise ValueError('Pending/duplicate plans require separate investigation')
    ids=[j['id'] for j in jobs]
    if len(ids)!=len(set(ids)) or any(j['status']!='complete' or j.get('returncode')!=0 for j in jobs):raise ValueError('Unfinished/duplicate/failed historical job')
    actual={(j['plan'],j['id']) for j in jobs};expected={(p['id'],j['id']) for p in plans.values() for j in p['jobs']}
    if actual!=expected:raise ValueError('Completed plan/job history mismatch')


def inspect_idle(output,primary):
    if (output/'cancel.request').exists():raise ValueError('User cancellation present')
    state=json.loads((output/'research_state.json').read_text())
    pid=int((output/'queue.pid').read_text())
    if identity(pid) is not None:raise ValueError('Old PID still exists; inspect its identity, never assume ownership')
    plans={p.stem:json.loads(p.read_text()) for p in sorted((output/'plans').glob('*.json'))}
    check_completed_ledger(state,plans)
    latest=plans[state['completed_plans'][-1]]
    for name,digest in latest['dependency_hashes'].items():
        if file_sha(ROOT/name)!=digest:raise ValueError('Sealed dependency changed: '+name)
    if json.loads((primary/'experiment_report.json').read_text())['status']!='complete':raise ValueError('Primary incomplete')
    if not json.loads((output/'comparison_with_paper.json').read_text())['final_comparison_complete']:raise ValueError('Primary comparison incomplete')
    auth=json.loads((output/'authorization.json').read_text())
    if not auth['ongoing_until_user_cancels'] or auth['primary']!=str(primary) or auth['api_total_cap_cny']!=300:raise ValueError('Authorization binding changed')
    require_space(output)
    return state,plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('primary','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--adopt-after-reboot',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
    output=a.output.resolve();primary=a.primary.resolve();state,plans=inspect_idle(output,primary)
    if a.preflight:
        print(json.dumps(dict(status='idle_ledger_verified',completed_plans=len(plans),completed_jobs=len(state['jobs']),old_pid_absent=True,source_sha256=file_sha(__file__),api_calls=0)));return
    if not a.adopt_after_reboot:raise ValueError('Explicit audited-adoption mode required')
    import fcntl
    lock=(output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Recheck under the same lock used by the original dispatcher, before any write.
    state,plans=inspect_idle(output,primary)
    backup=output/('reboot_adoption_'+now().replace(':','').replace('+','_'));backup.mkdir()
    for name in ('research_state.json','queue.pid','queue.log','authorization.json'):shutil.copy2(output/name,backup/name)
    atomic_write_json(backup/'audit.json',dict(reason='Server reboot after all015 jobs completed; no historical job is resumed or replayed.',
        original_started_utc=state['started_utc'],completed_plans=state['completed_plans'],old_state_sha256=file_sha(backup/'research_state.json'),
        original_queue_source_sha256=file_sha(ROOT/'scripts/queue_fiq_research.py'),adoption_source_sha256=file_sha(__file__),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),api_calls=0,old_sources_or_plans_modified=False))
    if a.detach:
        pid=os.fork()
        if pid:print(json.dumps(dict(detached_queue_pid=pid,audit=str(backup))));return
        os.setsid();signal.signal(signal.SIGHUP,signal.SIG_IGN)
        with open(os.devnull,'rb') as null,(output/'queue_after_reboot.log').open('ab',buffering=0) as log:
            os.dup2(null.fileno(),0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    child=None;state=copy.deepcopy(state)
    state.update(pid=os.getpid(),adopted_after_reboot_utc=now(),adoption_audit=str(backup),queue_source='scripts/queue_fiq_research_after_reboot.py')
    (output/'queue.pid').write_text(str(os.getpid()));atomic_write_json(output/'queue_identity.json',identity(os.getpid()))
    def save(**values):
        state.update(values,updated_utc=now());atomic_write_json(output/'research_state.json',state)
    def guard():
        if (output/'cancel.request').exists():raise KeyboardInterrupt('User cancellation requested')
        require_space(output)
    env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ}
    env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    try:
        while True:
            guard();pending=[p for p in sorted((output/'plans').glob('*.json')) if p.stem not in state['completed_plans']]
            if not pending:save(status='awaiting_next_research_plan');time.sleep(15);continue
            plan=json.loads(pending[0].read_text())
            if plan['id']!=pending[0].stem or plan['api_cap_cny']!=0:raise ValueError('Invalid or paid plan')
            for job in plan['jobs']:
                guard()
                if any(file_sha(ROOT/n)!=h for n,h in plan['dependency_hashes'].items()):raise ValueError('Sealed dependencies changed')
                cmd=job['command'];source=(ROOT/cmd[1]).resolve()
                if cmd[0]!=sys.executable or not source.is_relative_to(ROOT) or file_sha(source)!=job['source_sha256'] or not 0<job['max_seconds']<=3600 or job['kind']!='diagnostic':raise ValueError('Adopted queue only permits sealed zeroAPI small diagnostics')
                if any(j['id']==job['id'] for j in state['jobs']):raise ValueError('Never replay an existing job')
                destination=Path(cmd[cmd.index('--output')+1]).resolve()
                if not destination.is_relative_to(output) or destination.exists():raise ValueError('Diagnostic output must be new and inside research results')
                row=dict(id=job['id'],plan=plan['id'],status='running',started_utc=now(),command=cmd);state['jobs'].append(row);started=time.monotonic()
                with (output/(job['id']+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid;save(status=job['id'])
                    while child.poll() is None:
                        guard()
                        if time.monotonic()-started>job['max_seconds']:raise TimeoutError('Small job timed out')
                        time.sleep(1)
                row.update(status='complete' if child.returncode==0 else 'failed',returncode=child.returncode,finished_utc=now(),elapsed_seconds=time.monotonic()-started);save()
                if child.returncode:raise RuntimeError('Diagnostic failed; no automatic retry')
                result=json.loads((destination/'results.json').read_text())
                if result['status']!='complete' or result['api_calls']!=0:raise ValueError('Missing/invalid diagnostic completion')
                child=None
            state['completed_plans'].append(plan['id']);save(status='batch_complete')
    except BaseException as error:
        stop(child)
        if state['jobs'] and state['jobs'][-1]['status']=='running':state['jobs'][-1].update(status='stopped',returncode=child.returncode if child else None)
        save(status='cancelled' if isinstance(error,KeyboardInterrupt) else 'stopped',error_type=type(error).__name__,error=str(error))
        if not isinstance(error,KeyboardInterrupt):raise


if __name__=='__main__':main()
