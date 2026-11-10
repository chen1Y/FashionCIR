from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_dev_support_audit import selected_fields,support_groups,audit_ranks,transitions


class DevSupportAuditTests(unittest.TestCase):
    def test_global_local_mapping_partition(self):
        g=support_groups([1024,1026],[True,False,True,True],{'1024':[],'1026':[],'1027':['outside']})
        self.assertEqual(g,dict(all=[0,1,2,3],supported=[0,2],byte_eligible_unsupported=[3],byte_ineligible=[1]))

    def test_reject_bad_selected_or_reason_binding(self):
        for selected in ([1023],[1025],[1024,1024]):
            with self.assertRaises(ValueError):support_groups(selected,[True,False],{'1024':[]})
        with self.assertRaises(ValueError):support_groups([],[True],{'1024':[]})

    def test_four_cells_and_original_indices(self):
        d=transitions([2,1,1,3],[1,2,1,4],[0,1,2,3],1)
        self.assertEqual([d[k] for k in ('gains','losses','both_hit','neither_hit')],[1,1,1,1])
        self.assertEqual(d['gained_query_indices'],[0]);self.assertEqual(d['lost_query_indices'],[1]);self.assertEqual(d['delta_pp'],0)

    def test_empty_group_and_buckets_preserved(self):
        ranks={k:[1,10,25] for k in ('fixed05','head011','armA','armB','shuffleA','shuffleB')}
        g=dict(all=[0,1,2],supported=[0,1],byte_eligible_unsupported=[2],byte_ineligible=[])
        d=audit_ranks(ranks,g,100);empty=d['byte_ineligible']
        self.assertEqual(empty['queries'],0);self.assertIsNone(empty['metrics']['armB']['r1'])
        self.assertIsNone(empty['transitions']['armB_minus_armA']['r1']['delta_pp'])
        self.assertEqual(sum(v['queries'] for v in d['all']['fixed05_rank_buckets'].values()),3)

    def test_overlap_or_missing_strata_rejected(self):
        ranks={k:[1] for k in ('fixed05','head011','armA','armB','shuffleA','shuffleB')}
        with self.assertRaises(ValueError):audit_ranks(ranks,dict(all=[0],supported=[0],byte_eligible_unsupported=[0],byte_ineligible=[]),100)

    def test_tail_rank_and_recall_invariant(self):
        ranks={k:[21] for k in ('fixed05','head011','armA','armB','shuffleA','shuffleB')};ranks['armB']=[22]
        with self.assertRaises(ValueError):audit_ranks(ranks,dict(all=[0],supported=[0],byte_eligible_unsupported=[],byte_ineligible=[]),100)

    def test_noninteger_nonfinite_rank_rejected(self):
        for value in (float('nan'),float('inf'),True,0,100):
            ranks={k:[value] for k in ('fixed05','head011','armA','armB','shuffleA','shuffleB')}
            with self.assertRaises(ValueError):audit_ranks(ranks,dict(all=[0],supported=[0],byte_eligible_unsupported=[],byte_ineligible=[]),100)

    def test_skip_unrequested_official_value_without_decoding_it(self):
        text='{"status":"complete","official_validation":{"nested":[{"s":"escaped \\\" } ]"}],"bad_token":NOT_JSON},"internal_dev":{"rank":[1,20]}}'
        result=selected_fields(text,{'status','internal_dev'})
        self.assertEqual(result,dict(status='complete',internal_dev={'rank':[1,20]}))
        with self.assertRaises(ValueError):selected_fields('{"x":1,"x":2}',{'x'})


if __name__=='__main__':unittest.main()
