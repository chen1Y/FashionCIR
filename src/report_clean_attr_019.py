"""019 no-model result/flow report, preserving all fixed examples and failures."""
import argparse,json,hashlib,re,sqlite3,collections,gzip
from pathlib import Path
import torch
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

def comparison(new,old,k):
 a=sum(x<=k<y for x,y in zip(new,old));b=sum(y<=k<x for x,y in zip(new,old))
 return dict(gains=a,losses=b,both=sum(x<=k and y<=k for x,y in zip(new,old)),neither=sum(x>k and y>k for x,y in zip(new,old)),net_pp=100*(a-b)/len(new))

def main():
 p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);a=p.parse_args();r=a.research;folder=r/'attr_clean_alignment_019';control=r/'attr_clean_019_control';gen=r/'clean_attr_generation_019'
 state=json.loads((control/'state.json').read_text());plan=json.loads((control/'generation_plan.json').read_text());root=Path(__file__).resolve().parents[1]
 if state['status']!='complete' or any(j['status']!='complete' or j['returncode']!=0 for j in state['jobs']):raise ValueError('019 incomplete')
 for n,h in plan['source_hashes'].items():
  if file_sha(root/n)!=h:raise ValueError('Sealed source changed')
 completion=json.loads((gen/'completion.json').read_text());g=json.loads((gen/'generated_slots.json').read_text());requests=sqlite3.connect('file:'+str(gen/'requests.sqlite3')+'?mode=ro',uri=True)
 states=dict(requests.execute('select state,count(*) from requests group by state'));requests.close()
 if states.get('reserved',0) or states.get('metered',0):raise ValueError('Paid requests still in flight')
 heldpath=r/'train_only_candidate_head/official_heldout_byte_index.json.gz';index=json.loads(gzip.decompress(heldpath.read_bytes()));expected=json.loads((heldpath.parent/'byte_partition_audit_v2.json').read_text())['heldout_byte_index_sha256']
 if file_sha(heldpath)!=expected:raise ValueError('Heldout byte index changed')
 for n,h in index['metadata_sha256'].items():
  if file_sha(Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/image_splits')/n)!=h:raise ValueError('Heldout metadata changed')
 held=set(index['image_sha256'].values());summary=dict(status='verified_complete',generation_cost=completion['cost'],request_states=states,categories={},api_generation_targets_sent=False,full_gallery_evaluated=False)
 lines=['# Qwen四槽文本：生成→Token训练→检索','', '固定每类前2fit+前2dev，未按检索成败挑选。每类仅32dev、继承小候选池，所有行和失败保留；不是完整FashionIQ或独立测试。','', '## 全部结果','', '|类别|方案|R@1|R@5|R@10|NCE|','|---|---|---:|---:|---:|---:|']
 for cat in ('shirt','toptee'):
  f=folder/cat;d=json.loads((f/'results.json').read_text());old=json.loads((r/'attr_pair_alignment_018'/cat/'results.json').read_text());m=json.loads((f/'manifest.json').read_text());raw=json.loads((r/'attr_pair_alignment_018'/cat/'manifest.json').read_text());generated=g['categories'][cat]
  if d['status']!='complete' or d['updates']!=50 or d['presentations']!=200 or d['exposure']!=old['arms']['tokens_last2']['exposure'] or d['queries_retained']!=160:raise ValueError('Updates/exposure/rows mismatch')
  if d['baseline']!=old['baseline'] or d['original018']!=old['arms']['tokens_last2']['selected']:raise ValueError('Old baselines changed')
  if d['selected']!=min(d['history'],key=lambda h:h['metrics']['retrieval'])['metrics']:raise ValueError('Wrong checkpoint selection')
  if file_sha(d['weight_path'])!=d['weight_sha256'] or d['frozen_sha256_before']!=d['frozen_sha256_after'] or not d['old_vocabulary_exact'] or not d['vit_frozen']:raise ValueError('Weight/freeze mismatch')
  payload=torch.load(d['weight_path'],map_location='cpu',weights_only=True);delta=payload['state']
  if list(delta['new_token_rows'].shape)!=[4,1024] or any(not torch.isfinite(t).all() for t in [delta['new_token_rows']]+list(delta['blocks'].values())):raise ValueError('Nonfinite or wrong shape saved weights')
  basepath=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_shirt_toptee_20261022')/cat/'checkpoints'/(cat+'_dqu_original_seed42_best.pt')
  base=torch.load(basepath,map_location='cpu',weights_only=True,mmap=True)['model_state'];block_delta={}
  for block in (22,23):
   names=[n for n in delta['blocks'] if n.startswith('transformer.resblocks.'+str(block)+'.')]
   if len(names)!=12:raise ValueError('Wrong trained block state')
   diffs=[delta['blocks'][n]-base['clip.'+n] for n in names]
   block_delta[str(block)]=dict(delta_l2=sum(float(t.square().sum()) for t in diffs)**.5,max_abs_change=max(float(t.abs().max()) for t in diffs))
  del base,payload,delta
  gallery=r/'train_gallery_diagnostic'/cat/'manifest.json';source005=json.loads((gallery.parent.parent/'completion_summary.json').read_text())['categories'][cat]
  if file_sha(gallery)!=source005['manifest_sha256']:raise ValueError('Gallery byte manifest changed')
  hashes=json.loads(gallery.read_text())['image_sha256'];used={s:set() for s in ('fit','dev')}
  for row,orig in zip(m['records'],raw['records']):
   for k in ('split','original_row','source_index','reference','target','pool_ids','original_order'):
    if row[k]!=orig[k]:raise ValueError('Original pool/query changed')
   used[row['split']].update(hashes[i] for i in [row['reference']]+row['pool_ids'])
  intersections=dict(fit_dev=len(used['fit']&used['dev']),fit_heldout=len(used['fit']&held),dev_heldout=len(used['dev']&held))
  if any(intersections.values()):raise ValueError('Byte isolation failed')
  for metric in [d['baseline'],d['original018'],d['initial'],d['selected']]+[h['metrics'] for h in d['history']]:
   if len(metric['ranks'])!=32:raise ValueError('Dev row count')
   for k in (1,5,10):
    if abs(metric['candidate_pool_r'+str(k)]-100*sum(x<=k for x in metric['ranks'])/32)>1e-8:raise ValueError('Rank/metric mismatch')
  pairs={'frozen_baseline':d['baseline'],'old_Qwen_last2':d['original018'],'new_Qwen_initial':d['initial'],'new_Qwen_last2':d['selected']}
  for name,metric in pairs.items():lines.append('|'+cat+'|'+name+'|'+ '|'.join(f"{metric['candidate_pool_r'+str(k)]:.3f}" for k in (1,5,10))+'|'+f"{metric['retrieval']:.6f}"+'|')
  reasons=collections.Counter(e for x in generated for e in x['errors']);relative=[]
  for row in generated:
   if row['status']=='ok' and re.search(r'\b(longer|shorter|darker|lighter|less|more)\b',row['modification_text'],re.I):
    text=' '.join(row['target_attrs'].values());relative.append(dict(source_index=row['source_index'],relative_word_retained=bool(re.search(r'\b(longer|shorter|darker|lighter|less|more|than|reduced|increased)\b',text,re.I))))
  summary['categories'][cat]=dict(rows=pairs,selected_step=d['selected_step'],valid=d['generated_valid'],total=d['queries_retained'],fallback_reasons=dict(reasons),byte_intersections=intersections,
   versus_old={str(k):comparison(d['selected']['ranks'],d['original018']['ranks'],k) for k in (1,5,10)},versus_baseline={str(k):comparison(d['selected']['ranks'],d['baseline']['ranks'],k) for k in (1,5,10)},
   relative_language_proxy=dict(queries=len(relative),retained=sum(x['relative_word_retained'] for x in relative),note='Lexical proxy only, not expert semantic correctness; audit does not change training or select queries.'),tokens=d['tokens'],block_changes=block_delta,weight_path=d['weight_path'],weight_sha256=d['weight_sha256'])
 for cat in ('shirt','toptee'):
  d=json.loads((folder/cat/'process_examples.json').read_text());result=json.loads((folder/cat/'results.json').read_text());rng=torch.Generator().manual_seed(42);selected_exposure=torch.zeros(128,dtype=torch.long)
  for _ in range(result['selected_step']):selected_exposure[torch.randperm(128,generator=rng)[:4]]+=1
  lines+=['','## '+cat+' 固定流程样例','']
  for e in d['examples']:
   if e['split']=='dev':
    local=e['index']-128
    if e['before']['gt_rank']!=result['initial']['ranks'][local] or e['after']['gt_rank']!=result['selected']['ranks'][local]:raise ValueError('Batch1 illustrative rank differs from batch4 aggregate; inspect before reporting')
   lines+=['### '+e['split']+' source_index='+str(e['source_index']),'', '参考图ID：'+e['reference']+'；真实GT ID（仅监督/报告，不发Qwen）：'+e['target'],'','原修改句：'+e['modification_text'],'','原属性：`'+json.dumps(e['old_attrs'],ensure_ascii=False)+'`','','新实际输入：','```text',e['actual_input'],'```','',
    '回退：'+str(e['fallback'])+'；完整50步总呈现次数：'+str(e['times_in_training'])+'；截至选中checkpoint呈现次数：'+str(int(selected_exposure[e['index']]) if e['split']=='fit' else 0)+'（dev为0，但参与NCE选模；全体查询共享4Token）。','',
    f"同新文本训练前GT rank={e['before']['gt_rank']}、cos={e['before']['gt_cosine']:.6f}；选中权重GT rank={e['after']['gt_rank']}、cos={e['after']['gt_cosine']:.6f}。",'',
    '训练前top3：`'+json.dumps(e['before']['top3'],ensure_ascii=False)+'`','','训练后top3：`'+json.dumps(e['after']['top3'],ensure_ascii=False)+'`','',
    '各上下文化属性向量与初始向量余弦（包含Token及末2层共同影响）：`'+json.dumps(e['attribute_cosine_to_initial'])+'`','']
  lines+=['### 所有查询共享的4个Token参数','', '|Token|初始范数|选中范数|移动距离|相对移动%|与初始余弦|','|---|---:|---:|---:|---:|---:|']
  for slot,s in d['shared_token_stats'].items():lines.append('|'+slot+'|'+ '|'.join(f'{s[k]:.6f}' for k in ('initial_norm','current_norm','delta_norm','relative_change_percent','cosine_to_initial'))+'|')
 lines+=['','## 限制','', '程序门禁和同模型自检不是人工GT，不保证所有矛盾/臆测均已排除；相对词可能仍被改写为绝对词，相关词汇代理全量附报且不用于改训练。软GT patch教师不是人工区域属性标签，loss或向量移动不等于语义子空间已解耦。原DQU看过train，dev用于选模，字节SHA不排除重新编码近重复。','']
 atomic_write_json(folder/'completion_summary.json',summary);(folder/'process_report.md').write_text('\n'.join(lines));print(json.dumps({'status':summary['status'],'cost':summary['generation_cost'],'categories':{c:dict(valid=v['valid'],selected_r1=v['rows']['new_Qwen_last2']['candidate_pool_r1'],relative=v['relative_language_proxy']) for c,v in summary['categories'].items()}}))

if __name__=='__main__':main()
