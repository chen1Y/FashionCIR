import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_text_branch_math_027 import corrected_modification,textual_query,metrics,transition,first_valid_index,ROWS
try:
    import torch
except ImportError:torch=None


class TextBranchTests(unittest.TestCase):
    def test_official_correction_and_reference_caption_composition(self):
        m=corrected_modification(['DARKER, blue!','Longer-sleeves.'],{'darker':'dark'})
        self.assertEqual(m,'dark blue and longer sleeves');self.assertEqual(textual_query('a white top',m),'a white top, but dark blue and longer sleeves')
        with self.assertRaises(ValueError):corrected_modification(['one'],{})
    def test_metrics_and_four_cells_preserve_denominator(self):
        self.assertEqual(metrics([1,2,11,51]),dict(r1=25.,r10=50.,r50=75.));self.assertEqual(transition([1,11,1,11],[11,1,1,11],10),dict(gains=1,losses=1,both=1,neither=1,net_pp=0.))
        with self.assertRaises(ValueError):transition([1],[1,2],10)
    def test_example_is_fixed_by_source_index_not_retrieval_outcome(self):
        records=[dict(source_index=17),dict(source_index=3),dict(source_index=0)];samples=[dict(status='ok',target_attrs={'color':'blue'}),dict(status='ok',target_attrs={'shape':'longer'}),dict(status='fallback',target_attrs={})]
        self.assertEqual(first_valid_index(records,samples,[0,1,2]),1);self.assertEqual(len(ROWS),7)
    @unittest.skipIf(torch is None,'Server Torch required')
    def test_query_text_ablation_has_no_reference_image_argument(self):
        from run_fiq_text_branch_027 import text_only_query
        out=dict(t_mod=torch.tensor([[3.,0.]]),text=torch.tensor([[0.,2.]]),reference=torch.tensor([[100.,100.]]))
        self.assertTrue(torch.equal(text_only_query(out,'original'),torch.tensor([[1.,0.]])))
        for arm in ('slots','dual'):self.assertTrue(torch.equal(text_only_query(out,arm),torch.tensor([[0.,1.]])))


if __name__=='__main__':unittest.main()
