from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cir_cumulative_budget import CumulativeBudget,micro
from fiq_candidate_check import judgment,rank_after,worker
from fiq_candidate_schema_probe import select_pairs, validate
from fiq_candidate_check_v2 import schema_guard


class CumulativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'global.sqlite3'
        self.budget=CumulativeBudget(self.path,10)

    def tearDown(self):
        self.budget.db.close();self.tmp.cleanup()

    def test_history_is_deduplicated_and_not_decreased(self):
        self.budget.import_history('a',2,{})
        self.budget.import_history('a',1,{})
        self.assertEqual(self.budget.summary()['accounted_upper_cny'],2)

    def test_cumulative_reservations_cannot_reset(self):
        self.budget.import_history('old',8,{})
        self.budget.reserve('one',1.5,'hash')
        with self.assertRaises(ValueError):self.budget.reserve('two',1,'hash')
        with self.assertRaises(ValueError):self.budget.reserve('one',1.5,'hash')
        with self.assertRaises(ValueError):CumulativeBudget(self.path,300)

    def test_other_connection_observes_the_same_allocation(self):
        other=CumulativeBudget(self.path,10)
        try:
            self.budget.reserve('a',5,'h')
            other.reserve('b',5,'h')
            with self.assertRaises(ValueError):self.budget.reserve('c',.1,'h')
        finally:other.db.close()

    def test_inflight_keeps_full_allocation(self):
        self.budget.reserve('a',1.5,'h')
        with self.assertRaises(ValueError):self.budget.settle('a',.1,False)
        self.assertEqual(self.budget.summary()['accounted_upper_cny'],1.5)
        self.budget.settle('a',.1,True)
        self.assertEqual(self.budget.summary()['accounted_upper_cny'],.1)

    def test_money_rounds_conservatively_and_invalid_values_fail(self):
        self.assertEqual(micro(.0000001),1)
        with self.assertRaises(ValueError):micro(float('nan'))


class CandidateScoringTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='linux','Anonymous-FD worker runtime is Linux-only')
    def test_worker_cannot_read_secret_or_send_before_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'protocol.json').write_text('{}')
            with patch('fiq_candidate_check.os.fdopen') as secret:
                with self.assertRaises(ValueError):worker(root,999,root/'budget.sqlite3')
                secret.assert_not_called()

    def test_schema_rejects_boolean_and_nonfinite_confidence(self):
        with self.assertRaises(ValueError):judgment('{"overall_score":true,"confidence":1}')
        with self.assertRaises(ValueError):judgment('{"overall_score":4,"confidence":NaN}')

    def test_head_rerank_cannot_rescue_tail_or_drop_failed_queries(self):
        q=dict(query_index=0,baseline_target_rank=25,gold_target_id='tail',
            candidates=[dict(candidate_id=str(i),similarity=1-i*.001) for i in range(20)])
        js=[dict(overall_score=4,confidence=1) for _ in range(20)]
        self.assertEqual(rank_after(q,js),25)
        q.update(gold_target_id='0',baseline_target_rank=1)
        js[0]=None
        self.assertEqual(rank_after(q,js),1)

    def test_equal_judgments_preserve_original_order(self):
        q=dict(query_index=0,baseline_target_rank=3,gold_target_id='2',
            candidates=[dict(candidate_id=str(i),similarity=1.) for i in range(20)])
        js=[dict(overall_score=3,confidence=.8) for _ in range(20)]
        self.assertEqual(rank_after(q,js),3)

    def test_probe_excludes_all_prior_attempts_including_unknown(self):
        queries = [dict(query_index=i, category=cat, reference_image='ref', instruction='longer',
            candidates=[dict(path=str(j)) for j in range(20)])
            for i, cat in enumerate(('shirt', 'toptee'))]
        pairs = select_pairs(dict(queries=queries), {0, 1, 20, 21, 22})
        self.assertEqual([p['original_index'] for p in pairs], [2, 3, 4, 5, 23, 24, 25, 26])

    def test_nested_explanation_is_rejected_without_loosening_original_schema(self):
        import json
        value = dict(overall_score=3, confidence=.8,
            criteria=[dict(requirement='longer', verdict='met', reason='visible')])
        with self.assertRaises(ValueError):validate(json.dumps(value))
        value['reason'] = 'The candidate is longer.'
        with self.assertRaises(ValueError):validate(json.dumps(value))
        del value['criteria'][0]['reason']
        self.assertEqual(validate(json.dumps(value))['overall_score'], 3)

    def test_schema_guard_stops_before_scheduling_another_paid_wave(self):
        good = dict(status='ok')
        bad = dict(status='failed')
        self.assertFalse(schema_guard([good] * 7))
        self.assertTrue(schema_guard([good] * 7 + [bad]))
        self.assertFalse(schema_guard([good] * 9 + [bad]))


if __name__=='__main__':unittest.main()
