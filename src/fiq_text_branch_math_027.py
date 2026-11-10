"""027 fixed query-side text ablations; gallery remains original-ID images."""
import math,string
ROWS=('dqu_caption_mod_native','dqu_corrected_mod_native','dqu_raw_mod_native','current_frozen_text','current_original_text','current_slots_text','current_dual_text')


def corrected_modification(captions,correction):
    if len(captions)!=2:raise ValueError('FashionIQ requires exactly two captions')
    translation=str.maketrans({c:' ' for c in string.punctuation})
    def clean(s):return ' '.join(correction.get(t,t) for t in str(s).lower().translate(translation).strip().split())
    return clean(captions[0])+' and '+clean(captions[1])


def textual_query(reference_caption,modification):return reference_caption+', but '+modification


def metrics(ranks):
    if not ranks or any(type(x) is not int or x<1 for x in ranks):raise ValueError('Invalid ranks')
    return {'r'+str(k):100*sum(x<=k for x in ranks)/len(ranks) for k in (1,10,50)}


def transition(after,before,k):
    if not after or len(after)!=len(before):raise ValueError('Rank denominator mismatch')
    pairs=[(a<=k,b<=k) for a,b in zip(after,before)]
    out=dict(gains=sum(a and not b for a,b in pairs),losses=sum(b and not a for a,b in pairs),both=sum(a and b for a,b in pairs),neither=sum(not a and not b for a,b in pairs))
    return dict(out,net_pp=100*(out['gains']-out['losses'])/len(after))


def first_valid_index(records,samples,indices):
    valid=[i for i in indices if samples[i]['status']=='ok' and any(samples[i]['target_attrs'].values())]
    if not valid:raise ValueError('No valid fixed example; do not choose using ranks')
    return min(valid,key=lambda i:(records[i]['source_index'],i))
