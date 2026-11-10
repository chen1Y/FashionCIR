import unittest
from expanded_attr_schema_021 import PROMPT,validate,request_key

class Relative021Tests(unittest.TestCase):
    def payload(self,shape='',color=''):
        return dict(target_attrs=dict(color=color,shape=shape,detail='',texture=''),uncertainties=[])
    def test_absolute_and_comparative_compatible(self):
        self.assertEqual(validate(self.payload('short sleeves, longer than reference'),'longer sleeves and short sleeves',lambda s:s.split()),[])
    def test_multiple_comparisons_required(self):
        self.assertIn('missing_comparative_darker',validate(self.payload('sleeves longer than reference','brown'),'darker and longer sleeves',lambda s:s.split()))
    def test_uncertainty_cannot_replace_relation(self):
        p=self.payload('short sleeves');p['uncertainties']=['shorter than reference']
        self.assertIn('missing_comparative_shorter',validate(p,'shorter sleeves',lambda s:s.split()))
    def test_independent_prompt_identity(self):
        from expanded_attr_schema import request_key as old
        self.assertNotEqual(request_key('sha','shorter'),old('sha','shorter'))
        self.assertIn('NOT encoded',PROMPT)

if __name__=='__main__':unittest.main()
