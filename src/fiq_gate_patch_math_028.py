"""Independent scalar references for masked gates and exact sum-of-patch-maxima."""
import math


def masked_weights(logits,valid):
    if len(logits)!=len(valid):raise ValueError('Gate mask mismatch')
    active=[x for x,m in zip(logits,valid) if m]
    if not active:return [0.]*len(logits)
    if any(not math.isfinite(x) for x in active):raise ValueError('Nonfinite gate logits')
    shift=max(active);weights=[math.exp(x-shift) if m else 0. for x,m in zip(logits,valid)];total=sum(weights)
    return [x/total for x in weights]


def summax_reference(slots,valid,patches):
    if len(slots)!=len(valid) or not patches:raise ValueError('Invalid local match')
    if not any(valid):return None
    def cosine(a,b):
        if len(a)!=len(b):raise ValueError('Feature width mismatch')
        aa=sum(x*x for x in a);bb=sum(x*x for x in b)
        if aa<=0 or bb<=0:raise ValueError('Zero local vector')
        return sum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb)
    return sum(max(cosine(s,p) for p in patches) for s,m in zip(slots,valid) if m)
