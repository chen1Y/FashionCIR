import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_relative_schema_030 import validate,eligible,request_key
class Relative030Tests(unittest.TestCase):
    def setUp(self):
        self.base=dict(color='lighter than reference',shape='short sleeves',detail='',texture='solid');self.value=dict(source_attrs=dict(color='black body',shape='short sleeves',detail='',texture='solid'),relations=[dict(cue='lighter',slot='color',axis='lightness',sign=1,region='body',source_phrase='black body',explicit_target=False)],hypotheses=[dict(self.base,color='charcoal, lighter than reference'),dict(self.base,color='grey, lighter than reference')],uncertainties=[])
    def test_separate_hypotheses_preserve_relative(self):self.assertEqual(validate(self.value,'lighter color',self.base,str.split),[])
    def test_lost_relation_rejected(self):
        self.value['hypotheses'][0]['color']='charcoal';self.assertIn('lost_comparative',validate(self.value,'lighter color',self.base,str.split))
    def test_non_action_changes_rejected(self):
        self.value['hypotheses'][0]['shape']='long sleeves';self.assertIn('changed_non_action_slot',validate(self.value,'lighter color',self.base,str.split))
    def test_unknown_keeps_query_without_guess(self):
        self.value['relations'][0].update(axis='unknown',sign=0,source_phrase='');self.value['hypotheses']=[];self.assertEqual(validate(self.value,'lighter color',self.base,str.split),[])
    def test_explicit_target_no_enumeration(self):
        self.value['relations'][0]['explicit_target']=True;self.assertIn('ungrounded_hypotheses',validate(self.value,'lighter white color',self.base,str.split))
    def test_key_binds_original_and_base(self):
        self.assertNotEqual(request_key('a','longer',self.base),request_key('a','long',self.base));self.assertTrue(eligible('longer sleeves'));self.assertFalse(eligible('long sleeves'))
    def test_wrong_polarity_rejected_not_repaired(self):
        self.value['relations'][0]['sign']=-1;self.assertIn('cue_axis_or_polarity_mismatch',validate(self.value,'lighter color',self.base,str.split))
if __name__=='__main__':unittest.main()
