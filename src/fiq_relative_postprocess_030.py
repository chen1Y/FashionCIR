"""030-P deterministic, weak-caption hypotheses; never consumes failed030 responses."""
import copy,re
SLOTS=('color','shape','detail','texture')
AXES={'lightness':('darker body color','lighter body color'),'brightness':('muted body color','bright body color'),'saturation':('muted low saturation colors','vivid high saturation colors'),'sleeve_length':('very short sleeves','very long sleeves'),'garment_length':('short garment length','long garment length'),'formality':('casual everyday style','formal dressy style')}
COLORS=('black','white','grey','gray','navy','blue','red','pink','green','yellow','orange','purple','brown','beige','cream','silver','gold')


def analyse(instruction):
    text=instruction.lower();found=[]
    def add(cue,slot,axis,sign):
        if re.search(r'\b'+re.escape(cue)+r'\b',text):found.append(dict(cue=cue,slot=slot,axis=axis,sign=sign))
    if not re.search(r'\b(weight|lighterweight|lightweight|material)\b',text):add('lighter','color','lightness',1)
    add('darker','color','lightness',-1);add('brighter','color','brightness',1);add('less colorful','color','saturation',-1);add('more colorful','color','saturation',1)
    for cue,sign in (('longer',1),('shorter',-1)):
        if re.search(r'\b'+cue+r'\b(?:\W+\w+){0,2}\W+sleeves?\b',text) or re.search(r'\bsleeves?\b(?:\W+\w+){0,2}\W+'+cue+r'\b',text):add(cue,'shape','sleeve_length',sign)
        elif re.search(r'\b'+cue+r'\b(?:\W+\w+){0,2}\W+(?:hem|shirt|top|length)\b',text):add(cue,'shape','garment_length',sign)
    for cue,sign in (('more formal',1),('less formal',-1),('more casual',-1),('less casual',1)):add(cue,'shape','formality',sign)
    result=[]
    for r in found:
        if any(x['axis']==r['axis'] and x['sign']!=r['sign'] for x in found):continue
        explicit=(r['slot']=='color' and any(re.search(r'\b'+c+r'\b',text) for c in COLORS)) or (r['axis']=='sleeve_length' and bool(re.search(r'\b(short|long|sleeveless|tank|elbow|three.quarter)\b',text)))
        if not explicit and (r['slot'],r['axis'],r['sign']) not in {(x['slot'],x['axis'],x['sign']) for x in result}:result.append(r)
    return result


def source_phrase(caption,slot,axis):
    text=caption.lower()
    if slot=='color':
        colors={c for c in COLORS if re.search(r'\b'+c+r'\b',text)}
        if 'gray' in colors:colors.remove('gray');colors.add('grey')
        if 'navy' in colors:colors.discard('blue')
        return next(iter(colors))+' body color' if len(colors)==1 else ''
    if axis=='sleeve_length':
        states=[]
        for value,pattern in [('sleeveless',r'\bsleeveless\b|\btank top\b'),('short sleeves',r'\bshort[ -]sleeve'),('long sleeves',r'\blong[ -]sleeve')]:
            if re.search(pattern,text):states.append(value)
        return states[0] if len(states)==1 else ''
    if axis=='garment_length':
        if re.search(r'\blong (?:shirt|top|hem)\b',text):return 'long garment length'
        if re.search(r'\bshort (?:shirt|top|hem)\b',text):return 'short garment length'
    if axis=='formality':
        if 'casual' in text and 'formal' not in text:return 'casual everyday style'
        if 'formal' in text and 'casual' not in text:return 'formal dressy style'
    return ''


def options(r,source):
    cue=r['cue'];direction=r['sign'];axis=r['axis']
    if axis=='lightness':
        hue=source.split()[0]
        if hue=='black':values=['charcoal','medium grey','light grey'] if direction>0 else []
        elif hue=='white':values=[] if direction>0 else ['light grey','medium grey','charcoal']
        elif hue in ('grey','silver'):values=['light grey','pale grey','off white'] if direction>0 else ['medium grey','charcoal','black']
        else:values=[prefix+' '+hue for prefix in (('light','pale','very pale') if direction>0 else ('dark','very dark','deep'))]
        return [v+', '+cue+' than reference' for v in values]
    if axis=='sleeve_length':
        if source=='sleeveless':values=['short sleeves','elbow length sleeves','three quarter sleeves'] if direction>0 else []
        elif source=='short sleeves':values=['elbow length sleeves','three quarter sleeves','long sleeves'] if direction>0 else ['cap sleeves','very short sleeves','sleeveless']
        elif source=='long sleeves':values=[] if direction>0 else ['three quarter sleeves','elbow length sleeves','short sleeves']
        else:values=[]
        return [v+', '+cue+' than reference' for v in values]
    return []


def derive(instruction,base,caption,old_status,encode):
    if old_status!='ok':return dict(status='fallback',reason='old_failed_unknown_blocked',relations=[],source_phrases={k:'' for k in SLOTS},hypotheses=[])
    relations=analyse(instruction);source={k:'' for k in SLOTS};choices={}
    for r in relations:
        same=[x for x in relations if x['slot']==r['slot']]
        # Multiple axes remain available for pole sum, but no invented single source.
        if len(same)!=1:continue
        phrase=source_phrase(caption,r['slot'],r['axis']);source[r['slot']]=phrase
        if phrase:
            values=options(r,phrase)
            if len(values) in (2,3):choices[r['slot']]=values
    hypotheses=[]
    if choices:
        for i in range(3):
            h=dict(base)
            for slot,values in choices.items():h[slot]=values[min(i,len(values)-1)]
            if all(len(encode(h[k]))<=limit for k,limit in zip(SLOTS,(16,16,16,15))) and h not in hypotheses:hypotheses.append(h)
    if len(hypotheses)<2:hypotheses=[]
    return dict(status='ok' if relations else 'fallback',reason=None if relations else 'no_unambiguous_unspecified_relative_axis',relations=relations,source_phrases=source,hypotheses=hypotheses,source_evidence='old reference caption weak lexical proxy; not target/human GT',source_caption=caption)
