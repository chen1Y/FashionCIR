"""Export all018 text inputs without loading a pretrained model or embeddings."""
import argparse,json,hashlib
from pathlib import Path
import torch
from torch import nn
import open_clip
from attr_slot_fusion import MultiAttrTextEncoder
from attr_token_pair_alignment import packed_audit

def main():
 p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 output=a.research/'attr_token_inputs_018';output.mkdir(exist_ok=False);reports={}
 for cat in ('shirt','toptee'):
  source=a.research/'attr_pair_alignment_018'/cat/'manifest.json';records=json.loads(source.read_text())['records']
  # Packing only depends on vocab IDs and context length, not model weights/width.
  dummy=nn.Module();dummy.token_embedding=nn.Embedding(49408,1);dummy.positional_embedding=nn.Parameter(torch.zeros(77,1))
  m=MultiAttrTextEncoder(dummy,open_clip.get_tokenizer('ViT-H-14'))
  audits=packed_audit(m,[r['view']['target_attrs'] for r in records])
  native=json.loads((a.research/'attr_pair_018_preflight'/(cat+'_packed_examples.json')).read_text())
  for new,saved in zip(audits,native):
   if any(new[k]!=saved[k] for k in new):raise ValueError('Packing export does not reproduce native model input')
  for e,r in zip(audits,records):
   e.update(split=r['split'],source_index=r['source_index'],reference=r['reference'],target=r['target'],modification_text=r['view']['mod_text'],view_trace=r['view']['trace'])
  result=dict(category=cat,records=audits,manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),native_first8_exact=True,
   queries=len(audits),truncated_queries=sum(any(e['truncated']) for e in audits),empty_slot_queries=sum(not all(e['valid']) for e in audits),
   mean_real_sequence_length=sum(len(e['token_ids']) for e in audits)/len(audits),api_calls=0,pretrained_model_loaded=False)
  (output/(cat+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2));reports[cat]={k:v for k,v in result.items() if k!='records'}
 lines=['# 实际属性文本与可学习 Token 输入','', '每类固定前8条fit样例，未按检索结果挑选；JSON另保存全部fit128/dev32行。这里展示截断后实际文本，标点空格来自CLIP BPE解码。','']
 for cat in ('shirt','toptee'):
  result=json.loads((output/(cat+'.json')).read_text());lines+=['## '+cat,'']
  for e in result['records'][:8]:
   lines+=['### source_index '+str(e['source_index']),'','原修改句：'+e['modification_text'],'','```text',e['actual_packed_text'],'```','',
    'align位置：'+str(e['align_positions'])+'；EOS：'+str(e['eos_position'])+'；valid：'+str(e['valid'])+'；truncated：'+str(e['truncated']),'']
 (output/'examples.md').write_text('\n'.join(lines));(output/'summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2));print(json.dumps(reports))

if __name__=='__main__':main()
