import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gate_fiq_full_024 import verify_paid_sets,verify_sample


class GateReuseTests(unittest.TestCase):
    def test_paid_or_unknown_key_cannot_be_resent(self):
        with self.assertRaises(ValueError):verify_paid_sets({'old':dict(response=None)},{'old':dict(response={'text':'ok'})})
    def test_any_new_unknown_blocks_even_if_last_batch_would_pass_rate(self):
        with self.assertRaises(ValueError):verify_paid_sets({},dict(last=dict(response=None)))
        verify_paid_sets({'old':dict(response=None)},{'new':dict(response={'text':'ok'})})
    def test_saved_known_and_invalid_texts_match_durable_source(self):
        for status in ('ok','fallback'):
            answer=dict(status=status,target_attrs={'shape':'longer sleeves'})
            verify_sample(dict(answer,request_key='known',origin='reused:x'),{'known':answer})
            with self.assertRaises(ValueError):verify_sample(dict(answer,request_key='known',target_attrs={},origin='reused:x'),{'known':answer})
    def test_unproven_text_rejected_but_historical_block_retained(self):
        verify_sample(dict(request_key='old',origin='historical_block'),{})
        with self.assertRaises(ValueError):verify_sample(dict(request_key='old',origin='new023'),{})


if __name__=='__main__':unittest.main()
