import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluate_fiq_candidate_consistency import capped


class ConsistencyTests(unittest.TestCase):
    def judge(self,score,*verdicts):
        return dict(overall_score=score,confidence=.9,criteria=[dict(requirement='x',verdict=v) for v in verdicts])

    def test_contradiction_dominates_without_mutating_original(self):
        raw = self.judge(4,'met','unclear','contradicted')
        self.assertEqual(capped(raw)['overall_score'],0)
        self.assertEqual(raw['overall_score'],4)

    def test_unclear_cannot_receive_the_highest_score(self):
        self.assertEqual(capped(self.judge(4,'met','unclear'))['overall_score'],2)

    def test_no_score_increases_and_failures_stay_explicit(self):
        self.assertIsNone(capped(None))
        self.assertEqual(capped(self.judge(1,'met'))['overall_score'],1)
        self.assertEqual(capped(self.judge(4,'met'))['overall_score'],4)


if __name__=='__main__':unittest.main()
