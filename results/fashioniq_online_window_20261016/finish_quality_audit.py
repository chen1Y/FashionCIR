import json,pathlib,sys,copy,datetime,subprocess,shutil
sys.path.insert(0,str(pathlib.Path.cwd()/'src'))
from fashioniq_augmentation import build_overlay,write_json
from verify_fashioniq_augmentation import audit
from cir_gpu_window import process_info,same_process
r=pathlib.Path('/root/autodl-tmp/results/fashioniq_augmentation_20261016');root=pathlib.Path('/root/autodl-tmp/data/FashionIQ')
state=json.loads((r/'followup_state.json').read_text());proof=json.loads((r/'gpu_hold.json').read_text())
assert state['status']=='complete' and proof['status']=='released' and proof['release_result']=='same_dispatcher_resumed'
queue=process_info(proof['queue']['pid']);assert same_process(queue,proof['queue']) and queue['state'] not in ('T','t','Z')
assert process_info(state['labeling_pid']) is None
write_json(r/'loader_check.json',audit(root,r/'overlay'))
raw=json.loads((r/'qwen_reviewed.json').read_text());screened=copy.deepcopy(raw);flags=json.loads((r/'quality_holdouts.json').read_text())
for flag in flags['holdouts']:
 i=flag['index'];x=screened['records'][i];assert i>=24 and x['decision']=='accept' and x['candidate']==flag['candidate'] and x['target']==flag['target']
 x['decision']='reject';x['assistant_quality_holdout']=flag['reason']
screened['quality_screening']=flags
summary=build_overlay(root,screened,r/'overlay_quality_screened');write_json(r/'screened_loader_check.json',audit(root,r/'overlay_quality_screened'))
snapshot=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),followup=state,gpu_hold=proof,shoes_queue=queue,shoes_report_status=json.loads((r.parent/'cirr_shoes_20261015/experiment_report.json').read_text())['status'],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip(),disk_free_gib=shutil.disk_usage(r).free/(2**30),screened_summary=summary)
write_json(r/'completion_resource_check.json',snapshot)
print(json.dumps(dict(status='passed',new_accepted=state['new_accepted'],new_rejected=state['new_rejected'],screened_summary=summary)),flush=True)
