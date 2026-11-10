"""Eight new candidate pairs to check an explicit JSON template, not retrieval gains."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from cir_cumulative_budget import CumulativeBudget, micro
from experiment_storage import require_space
from fashioniq_noise import file_sha
from fiq_candidate_check import PairClient, judgment, now
from generate_shoes_online import Ledger
from generate_structured_edits import atomic_write_json
from shoes_online_pilot import BASE_URL, reference_data_url

BATCH = 'candidate_schema_probe_003b_20261026'
CAP = .05
MAX_CALLS = 8
PROMPT = '''Image 1 is a reference garment. Image 2 is a retrieved candidate.
Compare the candidate with the reference using the requested modification below.
Preserve visible unmentioned garment properties. Compare relative changes directly;
do not invent absolute properties from words such as longer or lighter. Ignore pose,
background and lighting. Do not guess hidden details, exact fabric or brand.
Neither image is identified as a ground-truth target.
Return ONE JSON object with exactly these four top-level keys. Use this structure:
{"overall_score": 2, "confidence": 0.5, "criteria": [{"requirement": "requested change", "verdict": "unclear"}], "reason": "Brief overall explanation."}
overall_score must be an integer from 0 to 4: 0 contradicts, 1 poor, 2 unclear,
3 mostly fits, 4 clearly fits. confidence must be a number from 0 to 1.
criteria must contain 1 to 4 objects with ONLY requirement and verdict keys.
Each verdict must be met, contradicted, or unclear. Do NOT put reasons inside criteria.
reason MUST be a top-level string of at most 120 characters; keep it to one short sentence.
If a detail is not visible, use unclear. Output JSON only. Requested modification: '''


def select_pairs(manifest, attempted):
    pairs = []
    for category in ('shirt', 'toptee'):
        selected = []
        for query in manifest['queries']:
            if query['category'] != category:
                continue
            for rank, candidate in enumerate(query['candidates']):
                old_index = query['query_index'] * 20 + rank
                if old_index not in attempted:
                    selected.append(dict(category=category, original_index=old_index,
                        reference_image=query['reference_image'], instruction=query['instruction'],
                        candidate_image=candidate['path']))
                if len(selected) == 4:
                    break
            if len(selected) == 4:
                break
        if len(selected) != 4:
            raise ValueError('Insufficient new pairs; never repeat known or unknown requests')
        pairs.extend(selected)
    return pairs


def validate(text):
    value = judgment(text)  # Keep the original numeric and reason acceptance checks.
    if set(value) != {'overall_score', 'confidence', 'criteria', 'reason'}:
        raise ValueError('Expected exactly four top-level fields')
    if len(value['criteria']) > 4 or any(set(x) != {'requirement', 'verdict'} for x in value['criteria']):
        raise ValueError('Criteria must have only requirement and verdict')
    return value


def source_hashes():
    src = Path(__file__).resolve().parent
    return {name: file_sha(src / name) for name in
        ('fiq_candidate_schema_probe.py', 'fiq_candidate_check.py', 'cir_cumulative_budget.py',
         'generate_shoes_online.py', 'shoes_online_pilot.py', 'generate_structured_edits.py')}


def prepare(output, previous, global_budget):
    if output.exists():
        raise FileExistsError('Never overwrite a prepared or attempted probe')
    require_space(output)
    old = json.loads((previous / 'protocol.json').read_text())
    if file_sha(previous / 'manifest.json') != old['manifest_sha256']:
        raise ValueError('Original manifest changed')
    db = sqlite3.connect('file:' + str(previous / 'requests.sqlite3') + '?mode=ro', uri=True)
    attempted = {r[0] for r in db.execute('SELECT idx FROM requests')}
    if db.execute("SELECT COUNT(*) FROM requests WHERE state IN ('reserved','metered')").fetchone()[0]:
        raise ValueError('Previous batch has unreconciled in-flight requests')
    db.close()
    budget = CumulativeBudget(global_budget)
    state = budget.db.execute('SELECT state FROM batches WHERE id=?', (old['batch_id'],)).fetchone()
    budget.db.close()
    if state != ('settled',):
        raise ValueError('Previous batch allocation must first be reconciled')
    manifest = json.loads((previous / 'manifest.json').read_text())
    pairs = select_pairs(manifest, attempted)
    paths = {p[k] for p in pairs for k in ('reference_image', 'candidate_image')}
    image_sha = {p: manifest['image_sha256'][p] for p in paths}
    for path, sha in image_sha.items():
        if file_sha(path) != sha:
            raise ValueError('Bound image changed')
    output.mkdir(parents=True)
    atomic_write_json(output / 'manifest.json', dict(pairs=pairs, image_sha256=image_sha,
        previous=str(previous), excluded_attempted_indices=sorted(attempted)))
    protocol = dict(batch_id=BATCH, max_cny=CAP, max_calls=MAX_CALLS, max_tokens=512,
        workers=1, rpm=60, retries=0, repairs=0, early_stop='Stop after two schema failures or any transport failure',
        sampling='First four unrequested candidate pairs per category in the existing fixed manifest; exclude known AND unknown prior request IDs',
        purpose='Output-format feasibility only; not retrieval metrics or independent semantic accuracy',
        prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(), source_sha256=source_hashes(),
        manifest_sha256=file_sha(output / 'manifest.json'), previous_protocol_sha256=file_sha(previous / 'protocol.json'),
        deadline_seconds=180, minimum_schema_coverage=1.0)
    atomic_write_json(output / 'protocol.json', protocol)
    atomic_write_json(output / 'state.json', dict(status='prepared', prepared_utc=now()))


def worker(output, key_fd, global_budget):
    protocol = json.loads((output / 'protocol.json').read_text())
    budget = CumulativeBudget(global_budget)
    allocation = budget.db.execute('SELECT cap,charged,state,protocol_sha256 FROM batches WHERE id=?', (BATCH,)).fetchone()
    budget.db.close()
    if allocation != (micro(CAP), None, 'reserved', file_sha(output / 'protocol.json')):
        raise ValueError('Require the exact cumulative allocation before reading a secret')
    if protocol['source_sha256'] != source_hashes() or protocol['manifest_sha256'] != file_sha(output / 'manifest.json'):
        raise ValueError('Sealed sources or manifest changed')
    with os.fdopen(key_fd) as stream:
        key = stream.read(4096).strip()
    if not key:
        raise ValueError('Empty credential')
    manifest = json.loads((output / 'manifest.json').read_text())
    ledger = Ledger(output / 'requests.sqlite3', protocol, CAP, MAX_CALLS)
    client = PairClient(key, BASE_URL, ledger)
    invalid = 0
    for index, pair in enumerate(manifest['pairs']):
        require_space(output)
        sample = dict(query_index=index, original_index=pair['original_index'], category=pair['category'], status='failed')
        try:
            for name in ('reference_image', 'candidate_image'):
                path = pair[name]
                if file_sha(path) != manifest['image_sha256'][path]:
                    raise ValueError('Bound image changed')
            ledger.begin('schema_probe', index, 'explicit_json_v2')
            answer = client.ask_pair(reference_data_url(pair['reference_image']),
                reference_data_url(pair['candidate_image']), PROMPT + pair['instruction'])
            ledger.finish(answer)
            sample['response'] = answer
            try:
                sample['judgment'] = validate(answer['text'])
                sample['status'] = 'ok'
            except ValueError as error:
                sample['error'] = str(error)
                invalid += 1
        except Exception as error:
            ledger.abandon()
            sample['error'] = client.safe_text(str(error))
            ledger.save_sample('schema_probe', sample)
            break  # Any transport/image/budget error stops paid work; never retry.
        ledger.save_sample('schema_probe', sample)
        if invalid >= 2:
            break
        time.sleep(1)
    samples = ledger.samples('schema_probe')
    valid = sum(s['status'] == 'ok' for s in samples)
    success = len(samples) == MAX_CALLS and valid == MAX_CALLS
    atomic_write_json(output / 'results.json', dict(status='format_probe_passed' if success else 'format_probe_failed',
        completed=len(samples), valid=valid, cost=ledger.summary(),
        note='Eight new pairs only. Schema success is not semantic accuracy or retrieval improvement. No automatic expansion.'))
    ledger.db.close()
    if not success:
        raise ValueError('The fixed output-format probe failed; no automatic retry or expansion')


def run(output, global_budget):
    import fcntl
    lock = (output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((output / 'state.json').read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    if state['status'] != 'prepared' or protocol['source_sha256'] != source_hashes():
        raise ValueError('No blind restart or changed sources')
    require_space(output)
    budget = CumulativeBudget(global_budget)
    budget.reserve(BATCH, CAP, file_sha(output / 'protocol.json'))
    atomic_write_json(output / 'global_allocation_before.json', budget.summary())
    budget.db.close()
    key = getpass.getpass('Qwen API key (memory only): ')
    if not key:
        raise ValueError('Empty credential; keep the allocation for inspection')
    readfd, writefd = os.pipe()
    child = None
    try:
        with (output / 'api.log').open('x') as log:
            child = subprocess.Popen([sys.executable, __file__, '--worker', '--output', str(output),
                '--global-budget', str(global_budget), '--api-key-fd', str(readfd)], pass_fds=(readfd,),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            os.close(readfd)
            os.write(writefd, key.encode())
            os.close(writefd)
            key = ''
            state.update(status='running', parent_pid=os.getpid(), worker_pid=child.pid, started_utc=now())
            atomic_write_json(output / 'state.json', state)
            started = time.monotonic()
            while child.poll() is None:
                require_space(output)
                if (output / 'cancel.request').exists() or time.monotonic() - started > 180:
                    raise TimeoutError('Probe cancelled or reached fixed deadline')
                time.sleep(1)
        db = sqlite3.connect('file:' + str(output / 'requests.sqlite3') + '?mode=ro', uri=True)
        inflight = db.execute("SELECT COUNT(*) FROM requests WHERE state IN ('reserved','metered')").fetchone()[0]
        upper = db.execute('SELECT COALESCE(SUM(charge),0) FROM requests').fetchone()[0]
        db.close()
        budget = CumulativeBudget(global_budget)
        budget.settle(BATCH, upper, no_inflight=inflight == 0)
        atomic_write_json(output / 'global_allocation_after.json', budget.summary())
        budget.db.close()
        state.update(status='complete' if child.returncode == 0 else 'failed', returncode=child.returncode, finished_utc=now())
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(status='stopped', error_type=type(error).__name__, full_allocation_retained=True, finished_utc=now())
        raise
    finally:
        atomic_write_json(output / 'state.json', state)


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--global-budget', type=Path, default=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/cir_api_budget_20261026.sqlite3'))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--worker', action='store_true')
    parser.add_argument('--api-key-fd', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output, args.previous, args.global_budget)
    elif args.run:
        run(args.output, args.global_budget)
    else:
        worker(args.output, args.api_key_fd, args.global_budget)


if __name__ == '__main__':
    main()
