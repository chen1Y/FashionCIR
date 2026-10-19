import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_fiq_augmentation_ablation import merge_structured


class MergeTests(unittest.TestCase):
    def sample(self, ref, target):
        return dict(candidate=ref,target=target,category='dress',split='train',status='ok',
                    validation={'valid':True},semantic_validation={'valid':True},
                    generation_input={'target_was_provided_to_model':False})

    def test_original_preserved_and_rejects_leakage_and_collisions(self):
        base={'samples':[self.sample('a','b')],'model':'old'}
        extra={'samples':[self.sample('c','d')]}
        result=merge_structured(base,extra,[('c','d')])
        self.assertEqual(result['samples'][:1],base['samples'])
        self.assertEqual(len(base['samples']),1)
        result['samples'][0]['status']='changed'
        self.assertEqual(base['samples'][0]['status'],'ok')
        bad=copy.deepcopy(extra);bad['samples'][0]['generation_input']['target_was_provided_to_model']=True
        with self.assertRaises(ValueError):merge_structured(base,bad,[('c','d')])
        with self.assertRaises(ValueError):merge_structured(base,extra,[('x','d')])
        with self.assertRaises(ValueError):merge_structured(base,{'samples':[self.sample('a','b')]},[('a','b')])

    def test_insufficient_effective_coverage_stops_training(self):
        extra={'samples':[self.sample('a','b')]}
        extra['samples'][0]['status']='failed'
        with self.assertRaises(ValueError):merge_structured({'samples':[]},extra,[('a','b')])


if __name__=='__main__':unittest.main()
