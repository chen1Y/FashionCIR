"""One frozen rule derived from blind criterion review; offline, no API or fitting."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3

from fiq_candidate_full_math import rank_after
from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json


RULE = 'Copy each valid judgment; cap overall_score to0 if any criterion contradicted, else2 if any unclear, else4; never change raw responses'


def capped(judgment):
    if judgment is None:
        return None
    result = copy.deepcopy(judgment)
    verdicts = {c['verdict'] for c in judgment['criteria']}
    ceiling = 0 if 'contradicted' in verdicts else 2 if 'unclear' in verdicts else 4
    result['overall_score'] = min(result['overall_score'],ceiling)
    return result


def metrics(ranks):
    return {**{f'r{k}':100*sum(r<=k for r in ranks)/len(ranks) for k in (1,5,10,20,50)},
        'mrr':sum(1/r for r in ranks)/len(ranks),'queries':len(ranks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--preregister',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite or repeat a completed consistency check')
    require_space(args.output)
    registered = json.loads(args.preregister.read_text())
    if registered['rule']!=RULE or registered['api_calls']!=0 or registered['coefficient']!=.03:
        raise ValueError('Rule differs from the recorded pre-metric hypothesis')
    original = json.loads((args.source/'results.json').read_text())
    manifest = json.loads((args.source/'manifest.json').read_text())
    if original['status']!='complete' or json.loads((args.source/'state.json').read_text())['status']!='complete':
        raise ValueError('Source paid experiment must end before offline analysis')
    if file_sha(args.source/'manifest.json')!=original['protocol']['manifest_sha256']:
        raise ValueError('Source candidate manifest changed')
    db = sqlite3.connect('file:'+str(args.source/'requests.sqlite3')+'?mode=ro',uri=True)
    if db.execute("SELECT COUNT(*) FROM requests WHERE state IN ('reserved','metered')").fetchone()[0]:
        raise ValueError('Source has in-flight requests')
    samples = {s['query_index']:s for (value,) in db.execute('SELECT value FROM samples') for s in [json.loads(value)]}
    db.close()
    if len(samples)!=320 or len(manifest['queries'])!=16:
        raise ValueError('Keep every original query and candidate record')
    evidence = dict(source_results_sha256=file_sha(args.source/'results.json'),
        source_manifest_sha256=file_sha(args.source/'manifest.json'),preregister_sha256=file_sha(args.preregister),
        source_sha256=file_sha(__file__),math_source_sha256=file_sha(Path(__file__).with_name('fiq_candidate_full_math.py')))
    categories = {}
    for category in ('shirt','toptee'):
        queries = [q for q in manifest['queries'] if q['category']==category]
        rows = {name:[] for name in ('baseline','strict_rerank','strict_shuffled','neutral_rerank','neutral_shuffled',
            'capped_strict','capped_strict_shuffled','capped_neutral','capped_neutral_shuffled')}
        changed,valid = 0,0
        for q in queries:
            raw = [samples[q['query_index']*20+i]['judgment'] for i in range(20)]
            limited = [capped(j) for j in raw]
            valid += sum(j is not None for j in raw)
            changed += sum(j is not None and j['overall_score']!=k['overall_score'] for j,k in zip(raw,limited))
            rows['baseline'].append(q['baseline_target_rank'])
            for prefix,judges,names in [('raw',raw,('strict_rerank','strict_shuffled','neutral_rerank','neutral_shuffled')),
                                       ('capped',limited,('capped_strict','capped_strict_shuffled','capped_neutral','capped_neutral_shuffled'))]:
                for name,shuffled,neutral in zip(names,(False,True,False,True),(False,False,True,True)):
                    rows[name].append(rank_after(q,judges,shuffled,neutral))
        for name in ('baseline','strict_rerank','strict_shuffled','neutral_rerank','neutral_shuffled'):
            if rows[name]!=original['categories'][category]['ranks'][name]:
                raise ValueError('Cannot reproduce original006 per-query ranks')
        values = {name:metrics(ranks) for name,ranks in rows.items()}
        if any(value['r20']!=values['baseline']['r20'] or value['r50']!=values['baseline']['r50'] for value in values.values()):
            raise ValueError('Head-only permutation illegally changed tail recall')
        categories[category] = dict(queries=len(queries),valid_judgments=valid,score_caps_applied=changed,
            metrics=values,ranks=rows,source_ranks_reproduced_exactly=True)
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json',dict(rule=registered,evidence=evidence,api_calls=0,training=False,
        scope='Exploratory same16 train-internal-dev queries. Blind visual score review motivated this rule before final006 retrieval metrics were inspected, not before data collection; no independent holdout or expert GT.'))
    atomic_write_json(args.output/'results.json',dict(status='complete',categories=categories,evidence=evidence,
        raw_responses_unchanged=True,new_api_calls=0,training=False))
    atomic_write_json(args.output/'state.json',dict(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(status='complete',categories={c:dict(score_caps_applied=v['score_caps_applied'],metrics=v['metrics']) for c,v in categories.items()})),flush=True)


if __name__=='__main__':main()
