"""Run paired, fixed-split 029D continuation seeds in two-worker batches."""

import argparse
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_029d_gate_refresh import CODE, EXPECTED, FIQ_EXPECTED
from run_fiq_native_caption_031 import BASE, identity, read, sha, snapshot, write


GROUPS = (('dress', 'shoes'), ('shirt', 'toptee'))
SEEDS = (43, 44, 45, 46)
FIT_COUNTS = {'dress': 1024, 'shoes': 1024, 'shirt': 972, 'toptee': 991}
METHODS = ('native_dqu', 'original_native', 'refreshed_native', 'refreshed_native_beta0',
           'original_raw', 'refreshed_raw')
PRIOR = {
    **dict.fromkeys(GROUPS[0], BASE / 'runs/fiq-029d-gate-refresh-20261106'),
    **dict.fromkeys(GROUPS[1], BASE / 'runs/fiq-029d-gate-refresh-shirt-toptee-20261106'),
}


def check_result(root, category, seed):
    state = read(root / 'state.json')
    if state['status'] != 'complete' or state.get('seed', 42) != seed or state['api_calls'] != 0:
        raise ValueError(f'Incomplete or mismatched task: {category} seed {seed}')
    fit_count = FIT_COUNTS[category]
    selections = {}
    for arm in ('original', 'refreshed'):
        fit = read(root / arm / 'fit_results.json')
        if fit['updates'] != 3 * ((fit_count + 7) // 8) or fit['exposures'] != [3] * fit_count:
            raise ValueError('Incorrect training exposure')
        selected = max(fit['history'], key=lambda h: (h['metrics']['r10'] + h['metrics']['r50'],
                                                      -h['metrics']['retrieval_nce'], -h['epoch']))
        if selected != fit['selected'] or sha(root / arm / selected['weight']) != selected['sha256']:
            raise ValueError('Selected checkpoint or dev ranking changed')
        selections[arm] = {key: selected[key] for key in ('epoch', 'weight', 'sha256')}
    path = root / 'official_results.json'
    result = read(path)
    expected = (EXPECTED[category] if category in EXPECTED else FIQ_EXPECTED[category][:2])
    n, g = expected
    if (result['queries'], result['gallery']) != expected or not all(result['original_rank_reproduction'].values()):
        raise ValueError('Official denominator or previously verified reference ranks changed')
    for row in result['rows'].values():
        ranks = row['ranks']
        if len(ranks) != n or any(type(rank) is not int or not 1 <= rank <= g for rank in ranks):
            raise ValueError('Invalid official ranks')
        for k in (1, 10, 50):
            if not math.isclose(row[f'r{k}'], 100 * sum(rank <= k for rank in ranks) / n):
                raise ValueError('Official recall failed recomputation')
    metrics = {name: {k: result['rows'][name][k] for k in ('r1', 'r10', 'r50')} for name in METHODS}
    return dict(category=category, seed=seed, status='verified', root=str(root), queries=n, gallery=g,
                seconds=state['seconds'], selections=selections, result_sha256=sha(path), metrics=metrics,
                gate_stats=result['gate_stats'], native_gate_pair=result['comparisons']['refreshed_native_minus_original_native'])


def record_summary(root, runs):
    aggregates = {}
    for category in FIT_COUNTS:
        entries = sorted((r for r in runs if r['category'] == category), key=lambda r: r['seed'])
        averages = {}
        for name in METHODS:
            averages[name] = {}
            for k in ('r1', 'r10', 'r50'):
                values = [r['metrics'][name][k] for r in entries]
                averages[name][k] = dict(mean=statistics.mean(values),
                                         sample_std=statistics.stdev(values) if len(values) > 1 else None)
        comparisons = {}
        for name, a, b in (
            ('native_refreshed_minus_original', 'refreshed_native', 'original_native'),
            ('raw_refreshed_minus_original', 'refreshed_raw', 'original_raw'),
            ('native_refreshed_minus_dqu', 'refreshed_native', 'native_dqu'),
            ('native_attribute_minus_beta0', 'refreshed_native', 'refreshed_native_beta0'),
        ):
            comparisons[name] = {}
            for k in ('r1', 'r10', 'r50'):
                values = [r['metrics'][a][k] - r['metrics'][b][k] for r in entries]
                comparisons[name][k] = dict(mean_pp=statistics.mean(values),
                                            sample_std_pp=statistics.stdev(values) if len(values) > 1 else None,
                                            wins=sum(v > 0 for v in values), ties=sum(v == 0 for v in values))
        aggregates[category] = dict(seeds=[r['seed'] for r in entries], metrics=averages, comparisons=comparisons)
    write(root / 'results_summary.json', dict(runs=runs, aggregates=aggregates))


def select_gpu():
    first = snapshot()
    time.sleep(5)
    second = snapshot()
    cards = [next(d for d in s['devices'] if d['index'] == 3) for s in (first, second)]
    uid = cards[0]['uuid']
    if (any(d['uuid'] != uid or d['free_mib'] < 20480 or d['utilization'] > 10 for d in cards)
            or any(line.startswith(uid + ',') for s in (first, second) for line in s['compute_apps'])):
        raise RuntimeError('GPU 3 failed the two-worker availability check')
    return uid, [first, second]


def task_name(task):
    return f"{task['category']}_seed{task['seed']}"


def run_queue(root, *, placement=None, worker='run_029d_gate_refresh.py', extra_worker_args=None,
              verifier=check_result, summarize=record_summary):
    import fcntl

    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = root / 'queue_state.json'
    if path.exists():
        raise FileExistsError(path)
    plan = read(root / 'plan.json')
    for name, digest in plan['sources'].items():
        if sha(root / 'code' / name) != digest:
            raise ValueError('Staged queue source changed')
    state = dict(status='running', identity=identity(), total_new_tasks=sum(map(len,plan['batches'])), complete_new_tasks=0,
                 batches_done=0, active=[], failed_tasks=[])
    write(path, state)
    runs = list(plan.get('seed42',[]))
    started = time.monotonic()
    active = []
    try:
        summarize(root, runs)
        for i, pair in enumerate(plan['batches'], 1):
            if (root / 'cancel.request').exists():
                raise InterruptedError('Queue cancellation requested')
            if shutil.disk_usage(root).free < 4 * (1 << 30):
                raise RuntimeError('Insufficient free disk for next pair')
            if placement is None:
                uid,snapshots=select_gpu()
                devices=[uid]*len(pair)
            else:
                devices,snapshots=placement(len(pair))
            if len(devices)!=len(pair):
                raise RuntimeError('Missing GPU assignments for batch')
            write(root / f'gpu_batch_{i:02}.json', dict(devices=devices,snapshots=snapshots))
            for task,uid in zip(pair,devices):
                env = {name: os.environ[name] for name in ('HOME', 'PATH', 'LANG') if name in os.environ}
                env.update(CUDA_VISIBLE_DEVICES=uid, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(CODE))
                folder = root / task_name(task)
                folder.mkdir()
                cmd = [sys.executable, str(root/'code'/worker), str(folder),
                       '--category',task['category'],'--seed',str(task['seed'])]
                cmd += ['--feature-cache',str(root/'feature_cache')] if extra_worker_args is None else extra_worker_args
                with (folder / 'run.log').open('x') as log:
                    child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
                active.append((task, folder, child))
            state.update(batch=i, active=[dict(task, pid=child.pid, root=str(folder)) for task, folder, child in active])
            write(path, state)
            print(f'batch {i}: {state["active"]}', flush=True)
            pair_start = time.monotonic()
            while any(child.poll() is None for _, _, child in active):
                now = time.monotonic()
                if now - pair_start > 3660 or now - started > 14400:
                    raise TimeoutError('Queue wall-time limit')
                time.sleep(3)
            for task, folder, child in active:
                try:
                    if child.returncode != 0:
                        raise RuntimeError(f'Worker exited with {child.returncode}')
                    row = verifier(folder, task['category'], task['seed'])
                    write(folder / 'verified_summary.json', row)
                    runs.append(row)
                    state['complete_new_tasks'] += 1
                except Exception as exc:
                    state['failed_tasks'].append(dict(task, error=str(exc), root=str(folder)))
            active.clear()
            summarize(root, runs)
            state['active'] = []
            if state['failed_tasks']:
                raise RuntimeError(f'Batch {i} contains failed tasks; inspect their logs')
            state['batches_done'] = i
            write(path, state)
        state.update(status='complete', seconds=time.monotonic() - started)
    except BaseException as exc:
        state.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed',
                     error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for _, _, child in active:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        write(path, state)


def launch(root):
    if (root / 'plan.json').exists() or (root / 'queue_state.json').exists():
        raise FileExistsError('Queue is already planned or started')
    if shutil.disk_usage(root).free < 25 * (1 << 30):
        raise RuntimeError('Need 25 GiB free for checkpoints and feature banks')
    batches = [[dict(category=c, seed=s) for c in group] for s in SEEDS for group in GROUPS]
    seed42 = [check_result(PRIOR[c] / c, c, 42) for c in FIT_COUNTS]
    plan = dict(batches=batches, seed42=seed42, gpu_index=3, max_parallel=2, api_calls=0,
                fixed_split_and_initial_weights=True,
                sources={p.name: sha(p) for p in (root / 'code').glob('*.py')})
    write(root / 'plan.json', plan)
    env = {name: os.environ[name] for name in ('HOME', 'PATH', 'LANG') if name in os.environ}
    env['PYTHONPATH'] = str(CODE)
    with (root / 'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(root), '--run'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
    write(root / 'launch.json', dict(pid=process.pid, plan_sha256=sha(root / 'plan.json')))
    print(f'Queue PID {process.pid}; {len(batches)} two-task batches', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run_queue(args.root)
    else:
        launch(args.root)
