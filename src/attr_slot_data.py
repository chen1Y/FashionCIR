"""Deterministic, auditable views of existing Qwen records; never changes labels."""
from __future__ import annotations
from typing import Any

SLOTS=('color','shape','detail','texture')
CATEGORY_SLOT={'color':'color','shape':'shape','fit':'shape','length':'shape','style':'shape',
    'part':'detail','detail':'detail','material':'texture','texture':'texture','pattern':'texture'}


def slot_view(sample:dict[str,Any]|None,mod_text:str)->dict[str,Any]:
    """Accept native target_attrs or conservatively adapt retain/remove/add.

    Explicit additions supersede retained values of the same category and region.
    Omitted retained values are traced, not silently added to deprecated attributes.
    Invalid/missing generation falls back to original modification alone.
    """
    result=dict(target_attrs={s:'' for s in SLOTS},drop_attrs='',mod_text=mod_text,trace=[],fallback=False)
    if not sample:
        result.update(fallback=True,trace=['missing_structured_record']);return result
    if 'target_attrs' in sample:
        result['target_attrs']={s:str(sample.get('target_attrs',{}).get(s,'') or '').strip() for s in SLOTS}
        result['drop_attrs']=str(sample.get('drop_attrs','') or '').strip();return result
    if sample.get('status')!='ok' or not sample.get('validation',{}).get('valid') or not sample.get('semantic_validation',{}).get('valid'):
        result.update(fallback=True,trace=['invalid_structured_record']);return result
    edit=sample.get('structured_edit') or {};add=edit.get('add') or [];remove=edit.get('remove') or [];retain=edit.get('retain') or []
    norm=lambda s:' '.join(str(s).lower().replace('_',' ').split())
    dropped={norm(x.get('attribute','')) for x in remove};changed={(norm(x.get('category','')),norm(x.get('region',''))) for x in add}
    values={s:[] for s in SLOTS}
    for kind,items in [('add',add),('retain',retain)]:
        for item in items:
            attr=str(item.get('attribute','')).strip();category=norm(item.get('category',''));region=norm(item.get('region',''))
            if not attr:continue
            if kind=='retain' and (norm(attr) in dropped or (category,region) in changed):
                result['trace'].append('omit_superseded_retain:'+attr);continue
            slot=CATEGORY_SLOT.get(category,'detail')
            if attr not in values[slot]:values[slot].append(attr)
    result['target_attrs']={s:', '.join(v) for s,v in values.items()}
    result['drop_attrs']=', '.join(dict.fromkeys(str(x.get('attribute','')).strip() for x in remove if str(x.get('attribute','')).strip()))
    return result


def collate_slot_views(rows:list[dict[str,Any]])->dict[str,Any]:
    """Rectangular text fields and explicit validity masks, with no fake attributes."""
    return dict(target_attrs=[r['target_attrs'] for r in rows],drop_attrs=[r['drop_attrs'] for r in rows],
        mod_text=[r['mod_text'] for r in rows],attr_valid=[[bool(r['target_attrs'][s]) for s in SLOTS] for r in rows],
        has_drop=[bool(r['drop_attrs']) for r in rows],trace=[r['trace'] for r in rows])


class SlotViewDataset:
    """Dataset protocol adapter usable by DataLoader with collate_slot_views."""
    def __init__(self,records:list[dict[str,Any]])->None:self.records=records
    def __len__(self)->int:return len(self.records)
    def __getitem__(self,index:int)->dict[str,Any]:
        row=self.records[index];return slot_view(row.get('structured_sample',row if 'target_attrs' in row else None),str(row.get('mod_text',row.get('modification_text',''))))
