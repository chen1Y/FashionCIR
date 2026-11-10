"""021 explicit relative-state instructions, independently versioned after020 stop."""
import hashlib,json
from expanded_attr_schema import validate,COMPARATIVES

VERSION='expanded-slots-relative021-v1'
PROMPT='''You see only a REFERENCE garment image and the user's modification text.
Produce concise attributes of the requested FINAL garment. No target image is given.
Return only JSON with exactly these keys:
{"target_attrs":{"color":"","shape":"","detail":"","texture":""},"uncertainties":[]}
Use color for colors; shape for sleeves/neckline/fit; detail for construction/decoration;
texture for pattern/material. Keep compatible visible reference properties, omit guesses.
Never preserve the old body color when recoloring; never invent fabric or a hidden layer.
Keep each attribute within 8 words where possible and at most16 CLIP tokens (texture15).
Do not emit SOS, EOS, special tokens or retain/remove/add lists.

CRITICAL: Relative comparisons and absolute states are different facts, not contradictions.
Longer does NOT mean long; shorter does NOT mean short. A short sleeve can be longer
than a reference sleeve. A brown shirt can also be darker than the reference.
Every occurrence of longer/shorter/darker/lighter/fewer/less/more in the instruction
must survive verbatim in the relevant target_attrs value, NOT only uncertainties.
These values are encoded; uncertainties are NOT encoded. Do not silently drop relations.
Examples:
Instruction: darker, longer sleeves, brown with parrots
color: darker brown; shape: sleeves longer than reference; texture: parrot print
Instruction: short sleeves and shorter sleeves
shape: short sleeves, shorter than reference
Instruction: longer sleeves, short sleeve, marijuana graphic
shape: short sleeves, longer than reference; detail: marijuana graphic
Instruction: less colorful, men's shirt
color: less colorful than reference; shape: men's shirt
Do not fill the less-colorful color value with obsolete reference red/color accents.
For actual unresolved conflict, omit the uncertain construction and note uncertainty;
do not invent overlays or mistake comparative and absolute sleeve lengths for conflict.
Recheck all four attributes for contradictions and ALL requested comparisons before JSON.
Instruction: '''


def request_key(sha,instruction):
    return hashlib.sha256(json.dumps(dict(model='qwen3.7-flash',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),image_sha256=sha,instruction=instruction),sort_keys=True).encode()).hexdigest()
