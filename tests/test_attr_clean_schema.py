import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from attr_clean_schema import validate,request_key

class CleanSchemaTests(unittest.TestCase):
 def row(self,**kwargs):
  a=dict(color='white base',shape='sleeveless relaxed fit',detail='',texture='black dot pattern');a.update(kwargs)
  return dict(target_attrs=a,uncertainties=[])
 def test_valid_empty_detail_and_scoped_colors(self):self.assertEqual(validate(self.row(),str.split),[])
 def test_cross_slot_conflict(self):self.assertIn('sleeveless_or_tank_top_with_explicit_sleeves',validate(self.row(detail='long sleeves'),str.split))
 def test_length_guard_no_silent_truncation(self):self.assertIn('texture_exceeds_actual_packer_budget',validate(self.row(texture='word '*16),str.split))
 def test_control_token_rejected(self):self.assertIn('model_supplied_control_token',validate(self.row(detail='<|align_color|>'),str.split))
 def test_relative_phrase_and_empty_slots(self):self.assertEqual(validate(self.row(shape='longer sleeves than reference',detail=''),str.split),[])
 def test_same_request_dedup_different_prompt_input(self):
  self.assertEqual(request_key('abc','pink'),request_key('abc','pink'));self.assertNotEqual(request_key('abc','pink'),request_key('abc','blue'))

if __name__=='__main__':unittest.main()
