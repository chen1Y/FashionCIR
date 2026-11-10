"""019 conservative final-state slots; rejects ambiguity, never invents repairs."""
import json,re,hashlib
SLOTS=('color','shape','detail','texture')
LIMITS=(16,16,16,15)
PROMPT='''You see ONLY a reference garment photo and its requested modification.
Describe the INTENDED FINAL garment, not the unchanged reference. There is NO target
image. The instruction overrides conflicting reference attributes. Never mix obsolete
and requested states: long sleeves cannot retain sleeveless/tank-top construction;
recoloring the body must not retain its old body color. Distinct trim/print colors may
coexist ONLY when clearly scoped to different visible regions. Keep unmentioned visible
properties only when compatible with the requested change. Preserve comparisons such
as longer/darker/less colorful relative to the reference; do not invent an absolute
color, sleeve length, fabric, hidden detail or product identity. Ignore background/pose.
Use positive, mutually compatible FINAL-state phrases. Put neckline/sleeves/fit in
shape, construction/decorations in detail, pattern/texture in texture. Omit uncertain
attributes with an empty string. Do not fill empty slots with guesses. Recheck all
four slots jointly for contradictions BEFORE returning. Your self-check is not human GT.
Return JSON ONLY, exactly this structure:
{"target_attrs":{"color":"white base","shape":"sleeveless, relaxed fit",
"detail":"","texture":"black dot pattern"},"uncertainties":[]}
Each value must be a short English phrase, ideally <=8 words. CLIP BPE limits for
color/shape/detail/texture are16/16/16/15 tokens respectively. Do not output SOS/EOS,
align tokens, retain/remove/add fields, ranks, targets or explanations outside JSON.
Instruction: '''

def prompt(instruction):return PROMPT+instruction

def conflict_errors(attrs):
 text=' '.join(attrs.values()).lower();errors=[]
 bare=bool(re.search(r'\b(?:sleeveless|tank[ -]+top)\b',text))
 sleeves=bool(re.search(r'\b(?:long|short|three[ -]quarter|full[ -]length)\s+sleeves?\b',text))
 if bare and sleeves:errors.append('sleeveless_or_tank_top_with_explicit_sleeves')
 if re.search(r'\blong\s+sleeves?\b',text) and re.search(r'\bshort\s+sleeves?\b',text):errors.append('long_and_short_sleeves')
 if re.search(r'\bsolid(?:[- ]colou?red)?\b',attrs['texture'].lower()) and re.search(r'\b(?:floral|polka|striped|stripes|dots|checkered|graphic)\b',attrs['texture'].lower()):errors.append('solid_and_pattern_same_texture_slot')
 return errors

def validate(payload,encode):
 errors=[]
 if not isinstance(payload,dict) or set(payload)!={'target_attrs','uncertainties'}:return ['unexpected_schema']
 a=payload['target_attrs']
 if not isinstance(a,dict) or set(a)!=set(SLOTS) or any(not isinstance(a[k],str) for k in SLOTS):return ['invalid_slots']
 if not isinstance(payload['uncertainties'],list) or len(payload['uncertainties'])>4 or any(not isinstance(x,str) or len(x)>200 for x in payload['uncertainties']):errors.append('invalid_uncertainties')
 if not any(a.values()):errors.append('all_slots_empty')
 for k,limit in zip(SLOTS,LIMITS):
  if len(a[k])>160 or len(encode(a[k]))>limit:errors.append(k+'_exceeds_actual_packer_budget')
  if '<|' in a[k] or '[SOS]' in a[k] or '[EOS]' in a[k]:errors.append('model_supplied_control_token')
 errors+=conflict_errors(a)
 return errors

def request_key(image_sha,instruction):
 return hashlib.sha256(json.dumps(dict(model='qwen3.7-flash',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),image_sha256=image_sha,instruction=instruction),sort_keys=True).encode()).hexdigest()
