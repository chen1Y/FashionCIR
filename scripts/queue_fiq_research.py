"""Own a sequential, cancellable follow-up queue after the immutable main comparison."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from report_fiq_comparison import build, metric_values


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(pid):
    try:
        base = Path('/proc') / str(pid)
        stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=stat[19], cwd=str((base/'cwd').resolve()),
                    command=(base/'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None


def ready(primary):
    # The main dispatcher marks a stage complete before durable publication.
    # Its top-level complete status is only written after all files are verified.
    if json.loads((primary/'experiment_report.json').read_text())['status'] != 'complete':
        return False
    result = build(primary)
    if not result['final_comparison_complete']:
        return False
    expected = {'shirt': {'train':5988,'val':2038}, 'toptee': {'train':6027,'val':1961}}
    for category, splits in expected.items():
        for split, count in splits.items():
            row = result['categories'][category][split+'_generation']
            if row['completed_queries'] != count or row['valid_queries']/count < .9:
                raise ValueError('Primary generation coverage/count mismatch')
    return True


def compatibility_resume_state(output):
    state = json.loads((output/'research_state.json').read_text())
    expected_error = 'Completed stage missing metric file: '
    if (state.get('status')!='stopped' or state.get('jobs') or state.get('completed_plans')
            or state.get('error_type')!='FileNotFoundError'
            or not state.get('error','').startswith(expected_error)):
        raise ValueError('Only the inspected pre-job publication race may use this recovery')
    if identity(int(state['pid'])) is not None:
        raise ValueError('Previous research dispatcher is still alive')
    return state


def parse_reload(log, category):
    text = log.read_text(errors='replace')
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            metric_values(value, category)
            found = value
        except (ValueError, KeyError, TypeError):
            continue
    if found is None:
        raise ValueError('Successful reload process did not emit complete metrics')
    return found


def first_plan(primary, output):
    protocol = json.loads((primary/'protocol.json').read_text())
    command = protocol['commands']['shirt'][0][1]
    clip = command[command.index('--clip-checkpoint')+1]
    data = command[command.index('--fashioniq-path')+1]
    jobs = []
    for category in ('shirt','toptee'):
        jobs.append(dict(id='reload_'+category, category=category, kind='reload', max_seconds=3600,
            command=[sys.executable,'src/dquTrain.py','--dataset',category,'--fashioniq-path',data,
                '--fashioniq-split','original-split','--clip-checkpoint',clip,'--seed','42',
                '--batch-size','16','--num-workers','4','--model-dir',str(output/'reload_eval'),
                '--resume',str(primary/category/'checkpoints'/f'{category}_dqu_original_seed42_best.pt'),
                '--eval-only']))
    jobs.append(dict(id='paper_comparison',kind='report',max_seconds=120,command=[sys.executable,
        'scripts/report_fiq_comparison.py','--results',str(primary),'--reloads',str(output/'baseline_reloads'),
        '--output',str(output/'comparison_with_paper')]))
    for category in ('shirt','toptee'):
        jobs.append(dict(id='frozen_pilot_'+category,kind='pilot',category=category,max_seconds=3600,
            command=[sys.executable,'src/run_fiq_small_pilot.py','--dataset',category,'--root',data,
                '--results',str(primary),'--output',str(output/f'pilot_{category}.json'),
                '--clip-checkpoint',clip]))
    return dict(id='001_frozen_hypotheses', created_utc=now(), api_cap_cny=0,
        hypothesis='Fixed dual baseline; bounded query confidence; add/remove top100 reranking; shuffled controls.',
        validation_policy='Predeclared rows, exploratory official val, report all; no val-selected deployment or significance claim.', jobs=jobs)


def seal_plan(plan):
    if plan['api_cap_cny'] != 0:
        raise ValueError('This dispatcher has no API credential or spending capability; paid work needs a separately metered plan')
    ids = [job['id'] for job in plan['jobs']]
    if len(ids) != len(set(ids)) or not all(re.fullmatch(r'[a-z0-9_]+', i) for i in ids):
        raise ValueError('Invalid or duplicate job IDs')
    for job in plan['jobs']:
        command = job['command']
        source = (ROOT/command[1]).resolve()
        if (command[0] != sys.executable or not source.is_relative_to(ROOT)
                or source.suffix != '.py' or not 0 < job['max_seconds'] <= 3600):
            raise ValueError('Unsupported command or oversized small job')
        job['source_sha256'] = file_sha(source)
    plan['dependency_hashes'] = {str(path.relative_to(ROOT)):file_sha(path)
        for folder in ('src','scripts') for path in sorted((ROOT/folder).glob('*.py'))}
    return plan


def stop(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--detach',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume-publication-wait',action='store_true',
        help='Recover only the inspected pre-job missing-metric publication race; preserve plans/budgets')
    args = p.parse_args()
    primary,output = args.primary.resolve(),args.output.resolve()
    recovered = None
    if args.resume_publication_wait:
        recovered = compatibility_resume_state(output)
    elif output.exists():
        raise FileExistsError('Inspect existing research queue; no blind duplicate/restart')
    main_pid = int((primary/'queue.pid').read_text())
    main_identity = identity(main_pid)
    main_report = json.loads((primary/'experiment_report.json').read_text())
    if main_identity is None and main_report['status'] != 'complete':
        raise ValueError('Main queue is not alive and not complete; inspect before arranging follow-up')
    if main_identity and ('resume_fiq_saved_pause.py' not in main_identity['command']
            or main_identity['cwd'] != str(Path('/home/lcy/fashionCIR/experiments/fiq-shirt-toptee-online-20261022'))):
        raise ValueError('Main PID identity mismatch')
    require_space(output)
    if recovered is not None:
        authorization = json.loads((output/'authorization.json').read_text())
        if authorization['primary'] != str(primary) or authorization['main_identity'] != main_identity:
            raise ValueError('Recovery must preserve the same live original primary dispatcher')
        paths = sorted((output/'plans').glob('*.json'))
        if len(paths)!=1 or paths[0].stem!='001_frozen_hypotheses':
            raise ValueError('Inspect changed plan set before recovery')
        plan = json.loads(paths[0].read_text())
        entry = 'scripts/queue_fiq_research.py'
        for name,digest in plan['dependency_hashes'].items():
            if name != entry and file_sha(ROOT/name)!=digest:
                raise ValueError('Recovery cannot change scientific dependencies')
        if plan['api_cap_cny']!=0 or any(file_sha(ROOT/j['command'][1])!=j['source_sha256'] for j in plan['jobs']):
            raise ValueError('Recovery cannot change job sources or spending')
    else:
        plan = seal_plan(first_plan(primary,output))
    if args.dry_run:
        print(json.dumps(dict(main_identity=main_identity,plan=plan),ensure_ascii=False))
        return
    if recovered is not None:
        backup = output/('publication_wait_recovery_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        backup.mkdir()
        for name in ('research_state.json','queue.log','authorization.json','queue.pid'):
            shutil.copy2(output/name,backup/name)
        shutil.copytree(output/'plans',backup/'plans')
        old_digest = plan['dependency_hashes']['scripts/queue_fiq_research.py']
        plan['dependency_hashes']['scripts/queue_fiq_research.py'] = file_sha(Path(__file__))
        atomic_write_json(output/'plans'/f"{plan['id']}.json",plan)
        atomic_write_json(backup/'repair.json',dict(reason='Wait for top-level completion before inspecting published files',
            previous_queue_source_sha256=old_digest,new_queue_source_sha256=file_sha(Path(__file__)),
            jobs_and_scientific_protocol_unchanged=True,api_calls=0,original_started_utc=recovered['started_utc']))
        with (output/'queue.log').open('a') as log:
            log.write('\n=== Inspected publication-wait compatibility recovery '+now()+' ===\n')
    else:
        output.mkdir(parents=True)
        (output/'plans').mkdir()
        atomic_write_json(output/'plans'/f"{plan['id']}.json",plan)
        atomic_write_json(output/'authorization.json',dict(created_utc=now(), main_identity=main_identity,
        primary=str(primary), ongoing_until_user_cancels=True, api_total_cap_cny=300,
        current_main_cap_cny=15, first_batch_cap_cny=0, credential_stored=False,
        budget_policy='One cumulative ceiling, never reset. Main keeps its existing category ledgers. Before any paid follow-up, inventory historical ledgers and atomically reserve a small batch (initially at most5 CNY) within300 including known/unknown/in-flight. This zero-API queue cannot spend.',
        resource_policy='disk>=8GiB; ephemeral features in tmpfs; no deletion of valid weights; one GPU child; max1h/job; no main protocol changes; cancel.request stops owned follow-up only.'))
    if args.detach:
        pid = os.fork()
        if pid:
            print(json.dumps(dict(detached_queue_pid=pid,output=str(output))))
            return
        os.setsid()
        signal.signal(signal.SIGHUP,signal.SIG_IGN)
        with open(os.devnull,'rb') as null,(output/'queue.log').open('ab',buffering=0) as log:
            os.dup2(null.fileno(),0)
            os.dup2(log.fileno(),1)
            os.dup2(log.fileno(),2)
    import fcntl
    lock = (output/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (output/'queue.pid').write_text(str(os.getpid()))
    state = recovered or dict(started_utc=now(),jobs=[],completed_plans=[])
    state.update(status='waiting_primary',pid=os.getpid())
    if recovered is not None:
        state.update(compatibility_resumed_utc=now(),recovery_audit=str(backup))
        state.pop('error',None)
        state.pop('error_type',None)
    child = None
    def save(**values):
        state.update(values,updated_utc=now())
        atomic_write_json(output/'research_state.json',state)
    def guard():
        if (output/'cancel.request').exists():
            raise KeyboardInterrupt('User cancellation requested')
        require_space(output)
    def gpu_free():
        return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    env = dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    try:
        save()
        while True:
            guard()
            report = json.loads((primary/'experiment_report.json').read_text())
            if report['status'] in ('failed','paused_by_user'):
                raise RuntimeError('Primary stopped; follow-up must not bypass incomplete comparison')
            if ready(primary) and identity(main_pid) != main_identity and gpu_free():
                # A reused PID is never killed or treated as the original parent.
                break
            if ready(primary) and main_identity is None and gpu_free():
                break
            time.sleep(30)
        if not (output/'comparison_with_paper.json').exists():
            # Verify all durable base files once, before any reload/pilot GPU job.
            for category,digest in report['checkpoint_hashes'].items():
                file = primary/category/'checkpoints'/f'{category}_dqu_original_seed42_best.pt'
                if file_sha(file) != digest:
                    raise ValueError('Protected main checkpoint digest mismatch')
        save(status='running_first_batch')
        while True:
            guard()
            plans = [f for f in sorted((output/'plans').glob('*.json')) if f.stem not in state['completed_plans']]
            if not plans:
                save(status='awaiting_next_research_plan')
                time.sleep(30)
                continue
            plan_path = plans[0]
            plan = json.loads(plan_path.read_text())
            if plan['id'] != plan_path.stem or plan['api_cap_cny'] != 0:
                raise ValueError('Unmetered or mismatched research plan')
            for job in plan['jobs']:
                guard()
                if any(file_sha(ROOT/name)!=digest for name,digest in plan['dependency_hashes'].items()):
                    raise ValueError('Sealed dependency source changed; inspect before next job')
                source = ROOT/job['command'][1]
                if file_sha(source) != job['source_sha256'] or job['max_seconds'] > 3600:
                    raise ValueError('Sealed job changed')
                if any(r['id']==job['id'] for r in state['jobs']):
                    raise ValueError('Duplicate job; inspect instead of rerunning')
                while not gpu_free():
                    save(status='waiting_external_gpu')
                    guard()
                    time.sleep(30)
                log_path = output/(job['id']+'.log')
                row = dict(id=job['id'],plan=plan['id'],status='running',started_utc=now(),command=job['command'])
                state['jobs'].append(row)
                started = time.monotonic()
                with log_path.open('x') as log:
                    child = subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    row['pid'] = child.pid
                    save(status=job['id'])
                    while child.poll() is None:
                        guard()
                        if time.monotonic()-started > job['max_seconds']:
                            raise TimeoutError('Small job exceeded declared wall budget')
                        with log_path.open('rb') as stream:
                            stream.seek(max(0,log_path.stat().st_size-8000))
                            tail = stream.read().decode(errors='replace')
                        if re.search(r'CUDA out of memory|OutOfMemoryError|(?:loss|train_loss)=(?:nan|inf)\b',tail,re.I):
                            raise RuntimeError('Small job OOM/nonfinite; preserve evidence')
                        time.sleep(10)
                row.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-started,finished_utc=now(),status='complete' if child.returncode==0 else 'failed')
                save()
                if child.returncode:
                    raise RuntimeError('Small job failed; no automatic retry')
                if job['kind']=='reload':
                    values = parse_reload(log_path,job['category'])
                    values.update(evaluated_utc=now(),server="legacy-research-host",checkpoint_sha256=report['checkpoint_hashes'][job['category']])
                    atomic_write_json(output/'baseline_reloads'/(job['category']+'.json'),values)
                elif job['kind']=='report':
                    if not json.loads((output/'comparison_with_paper.json').read_text())['final_comparison_complete']:
                        raise ValueError('Comparison not final; do not proceed to improvements')
                elif job['kind']=='pilot':
                    value = json.loads((output/f"pilot_{job['category']}.json").read_text())
                    if value['status']!='complete':
                        raise ValueError('Pilot incomplete')
                save()
            state['completed_plans'].append(plan['id'])
            save(status='batch_complete')
    except BaseException as error:
        stop(child)
        if state['jobs'] and state['jobs'][-1]['status']=='running':
            state['jobs'][-1].update(status='stopped',returncode=child.returncode if child else None)
        save(status='cancelled' if isinstance(error,KeyboardInterrupt) else 'stopped',error_type=type(error).__name__,error=str(error))
        if not isinstance(error,KeyboardInterrupt):
            raise


if __name__=='__main__':
    main()
