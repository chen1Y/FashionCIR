"""020 native slots with explicit lexical retention of comparative constraints."""
import hashlib,json,re
from attr_clean_schema import validate as validate_slots,PROMPT as OLD_PROMPT

VERSION='expanded-slots-relative020-v1'
PROMPT=OLD_PROMPT.rsplit('Instruction: ',1)[0]+'''
MANDATORY: Copy every comparative keyword in the instruction (longer, shorter,
darker, lighter, fewer, less, more) into the corresponding final attribute phrase.
Examples: "longer sleeves" -> "sleeves longer than reference", NEVER "long sleeves";
"less colorful" -> "less colorful than reference", do NOT retain obsolete red accents;
"more blue and bright blue" -> "more blue, bright blue".
Do not invent a boat-neck overlay or layered garment to reconcile unclear neckline
wording: omit the uncertain construction and record uncertainty. Keep each slot brief.
The original instruction will ALSO be encoded separately by the retrieval system.
Return exactly target_attrs and uncertainties, no additional keys.
Instruction: '''
COMPARATIVES=('longer','shorter','darker','lighter','fewer','less','more')

def validate(payload,instruction,encode):
 errors=validate_slots(payload,encode)
 if errors:return errors
 text=' '.join(payload['target_attrs'].values())
 for word in COMPARATIVES:
  if re.search(r'\b'+word+r'\b',instruction,re.I) and not re.search(r'\b'+word+r'\b',text,re.I):errors.append('missing_comparative_'+word)
 return errors

def request_key(sha,instruction):
 return hashlib.sha256(json.dumps(dict(model='qwen3.7-flash',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),image_sha256=sha,instruction=instruction),sort_keys=True).encode()).hexdigest()
