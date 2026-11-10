import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_candidate_full_math import historical_action,pair_key,rank_after,schema_guard
from fiq_candidate_full_gallery import answer_judgment,worker


class FullCandidateTests(unittest.TestCase):
    def test_pair_binding_uses_both_image_bytes_and_exact_instruction(self):
        key = pair_key('a','b','longer')
        self.assertNotEqual(key,pair_key('b','a','longer'))
        self.assertNotEqual(key,pair_key('a','b','shorter'))

    def test_unknown_and_changed_prompt_never_become_new_calls(self):
        old = dict(state='unknown',prompt_sha256='p',model='m',origin='old')
        self.assertEqual(historical_action([old],'p','m')['action'],'blocked')
        old.update(state='known',response=dict(text='paid invalid JSON'))
        self.assertEqual(historical_action([old],'other','m')['action'],'blocked')
        self.assertEqual(historical_action([old],'p','other')['action'],'blocked')

    def test_known_invalid_response_reused_without_paid_repair(self):
        old = dict(state='known',prompt_sha256='p',model='m',origin='old',response=dict(text='invalid'))
        self.assertEqual(historical_action([old],'p','m')['response']['text'],'invalid')
        self.assertEqual(historical_action([],'p','m')['action'],'new')

    def query(self):
        return dict(query_index=0,gold_target_id='1',baseline_target_rank=2,
            candidates=[dict(candidate_id=str(i),similarity=1-i*.001) for i in range(20)])

    def test_neutral_failure_does_not_drop_query_or_reshape_candidate_set(self):
        q = self.query();j = [None]*20;j[1] = dict(confidence=1,overall_score=4)
        self.assertEqual(rank_after(q,j),2)
        self.assertEqual(rank_after(q,j,neutral=True),1)
        self.assertEqual(rank_after(q,j,neutral=True,shuffled=True),1)
        q.update(gold_target_id='outside',baseline_target_rank=51)
        self.assertEqual(rank_after(q,j,neutral=True),51)

    def test_no_failure_makes_strict_and_neutral_rows_identical(self):
        j = [dict(confidence=1,overall_score=i%5) for i in range(20)]
        for shuffled in (False,True):
            self.assertEqual(rank_after(self.query(),j,shuffled),rank_after(self.query(),j,shuffled,True))

    def test_early_schema_guard_counts_returned_responses_not_blocked_history(self):
        good = dict(status='ok',response={});bad = dict(status='failed',response={})
        blocked = dict(status='failed',history_action='blocked')
        self.assertFalse(schema_guard([good]*15+[blocked]*20))
        self.assertTrue(schema_guard([good]*14+[bad]*2+[blocked]))
        self.assertFalse(schema_guard([good]*15+[bad]))

    def test_truncated_response_is_not_accepted_as_complete(self):
        with self.assertRaises(ValueError):answer_judgment(dict(text='{}',finish_reason='length'))

    @unittest.skipUnless(sys.platform=='linux','Anonymous-FD paid runtime is Linux-only')
    def test_unallocated_worker_cannot_read_credential_or_make_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'protocol.json').write_text('{}')
            with patch('fiq_candidate_full_gallery.os.fdopen') as secret:
                with self.assertRaises(ValueError):worker(root,999,root/'budget.sqlite3')
                secret.assert_not_called()


if __name__=='__main__':unittest.main()
