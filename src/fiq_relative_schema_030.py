"""030 query-only hypotheses are alternatives, never target labels or repairs."""
import hashlib,json,re
SLOTS=('color','shape','detail','texture')
AXES={'lightness':('darker body color','lighter body color'),'brightness':('muted body color','bright body color'),'saturation':('muted low saturation colors','vivid high saturation colors'),'sleeve_length':('very short sleeves','very long sleeves'),'garment_length':('short garment length','long garment length'),'formality':('casual everyday style','formal dressy style')}
CUES=('more formal','less formal','more casual','less casual','less colorful','more colorful','lighter','darker','brighter','longer','shorter')
PROMPT='''You see ONLY a reference garment image, an original modification instruction,
and immutable BASE_TARGET_ATTRS previously generated for this exact query. There is
NO target image, label, ranking or target ID. This is an uncertainty experiment.
First describe visible REFERENCE attributes separately in source_attrs. Do not put
requested changes into source_attrs. Empty strings mean unknown; never invent fabric,
hidden structure, exact measurements or product identity.
For every relative cue, identify its attribute slot, region and axis if grounded.
Allowed axes: lightness, brightness, saturation, sleeve_length, garment_length,
formality, unknown. Sign is1 for increasing the axis, -1 for decreasing,0 for unknown.
lighter is not automatically white or beige; brighter is not necessarily lighter.
Longer sleeves can still be short sleeves. A request for longer may refer to hem
length, not sleeves. If not disambiguated, use unknown and do not apply a direction.
explicit_target=true if the instruction already gives a specific corresponding
color/length/style. Do not enumerate replacements of an explicit requested value.
Do not call a reference attribute a false label or a semantic negative.

Only when source evidence is clear AND the target value is unspecified, propose
2 or3 plausible WHOLE target attribute hypotheses, ordered by increasing requested
change. They are possibilities, NOT ground truth. Each is a separate encoding,
NOT a comma-separated list of incompatible possibilities in one slot.
Keep all requested relative cue words verbatim in EACH relevant hypothesis field.
Copy BASE_TARGET_ATTRS fields EXACTLY for slots without an actionable relation.
Preserve the source hue unless a hue change is explicitly requested. For a black
body requested lighter, possible separate hypotheses could be 'charcoal, lighter
than reference', 'medium grey, lighter than reference', 'light grey, lighter than
reference'; do not claim beige is implied. For sleeveless reference requested longer
sleeves, separate short/elbow/three-quarter options may be plausible, not mandatory
long sleeves. Never replace longer with long or shorter with short alone.
Do not generate SOS/EOS/align tokens. Each target field must fit at most16 CLIP BPE
tokens, texture15; keep copied fields unchanged. If no grounded enumeration exists,
return hypotheses=[] and record uncertainty. This still retains the original query.

Return JSON ONLY, exactly:
{"source_attrs":{"color":"","shape":"","detail":"","texture":""},
"relations":[{"cue":"lighter","slot":"color","axis":"lightness","sign":1,
"region":"body","source_phrase":"black body","explicit_target":false}],
"hypotheses":[{"color":"charcoal, lighter than reference","shape":"COPY BASE",
"detail":"COPY BASE","texture":"COPY BASE"},{"color":"medium grey, lighter than
reference","shape":"COPY BASE","detail":"COPY BASE","texture":"COPY BASE"}],
"uncertainties":[]}
The example COPY BASE strings must be replaced by the exact base field values.
'''

def cues(text):return [x for x in CUES if re.search(r'\b'+re.escape(x)+r'\b',text.lower())]
def eligible(text):return bool(cues(text))
def request_key(sha,text,base):return hashlib.sha256(json.dumps(dict(version='relative030-v1',model='qwen3.7-flash',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),image_sha256=sha,instruction=text,base_attrs=base),sort_keys=True).encode()).hexdigest()
def prompt(text,base):return PROMPT+'\nINSTRUCTION: '+text+'\nBASE_TARGET_ATTRS: '+json.dumps(base,ensure_ascii=False)
def validate(d,text,base,encode):
    errors=[]
    if not isinstance(d,dict) or set(d)!={'source_attrs','relations','hypotheses','uncertainties'}:return ['schema']
    def attrs(x):return isinstance(x,dict) and set(x)==set(SLOTS) and all(isinstance(x[k],str) and len(x[k])<=180 for k in SLOTS)
    if not attrs(d['source_attrs']):return ['source_attrs']
    if not isinstance(d['relations'],list) or len(d['relations'])>4 or not isinstance(d['hypotheses'],list) or len(d['hypotheses']) not in (0,2,3):return ['relation_or_hypothesis_count']
    if not isinstance(d['uncertainties'],list) or any(not isinstance(x,str) or len(x)>250 for x in d['uncertainties']):errors.append('uncertainties')
    seen=set();action=set();active=[]
    for r in d['relations']:
        if not isinstance(r,dict) or set(r)!={'cue','slot','axis','sign','region','source_phrase','explicit_target'}:errors.append('relation_schema');continue
        if r['cue'] not in cues(text) or r['slot'] not in SLOTS or r['axis'] not in (*AXES,'unknown') or type(r['sign']) is not int or r['sign'] not in (-1,0,1) or type(r['explicit_target']) is not bool or any(not isinstance(r[k],str) or len(r[k])>180 for k in ('region','source_phrase')):errors.append('invalid_relation');continue
        if r['slot'] in seen:errors.append('multiple_relations_same_slot');continue
        seen.add(r['slot'])
        if r['axis']=='unknown' and r['sign']!=0:errors.append('unknown_axis_sign')
        if r['axis']!='unknown' and r['sign']==0:errors.append('known_axis_zero_sign')
        expected_sign={'lighter':1,'darker':-1,'brighter':1,'longer':1,'shorter':-1,'more formal':1,'less formal':-1,'more casual':-1,'less casual':1,'more colorful':1,'less colorful':-1}
        allowed_axis={'lighter':('lightness',),'darker':('lightness',),'brighter':('brightness','saturation'),'longer':('sleeve_length','garment_length'),'shorter':('sleeve_length','garment_length'),'more formal':('formality',),'less formal':('formality',),'more casual':('formality',),'less casual':('formality',),'more colorful':('saturation',),'less colorful':('saturation',)}
        if r['axis']!='unknown' and (r['axis'] not in allowed_axis[r['cue']] or r['sign']!=expected_sign[r['cue']]):errors.append('cue_axis_or_polarity_mismatch')
        if r['axis'] in ('sleeve_length','garment_length','formality') and r['slot']!='shape':errors.append('axis_slot')
        if r['axis'] in ('lightness','brightness','saturation') and r['slot']!='color':errors.append('axis_slot')
        if r['axis']!='unknown' and not r['explicit_target'] and r['source_phrase'].strip() and d['source_attrs'][r['slot']].strip():action.add(r['slot']);active.append(r)
    if d['hypotheses'] and not action:errors.append('ungrounded_hypotheses')
    for h in d['hypotheses']:
        if not attrs(h):errors.append('hypothesis_attrs');continue
        for k,limit in zip(SLOTS,(16,16,16,15)):
            if '<|' in h[k] or '[EOS]' in h[k] or '[SOS]' in h[k] or len(encode(h[k]))>limit:errors.append('token_budget_or_control_token')
            if k not in action and h[k]!=base[k]:errors.append('changed_non_action_slot')
        for r in active:
            if not re.search(r'\b'+re.escape(r['cue'])+r'\b',h[r['slot']].lower()):errors.append('lost_comparative')
    if len({json.dumps(h,sort_keys=True) for h in d['hypotheses']})!=len(d['hypotheses']):errors.append('duplicate_hypotheses')
    return sorted(set(errors))
