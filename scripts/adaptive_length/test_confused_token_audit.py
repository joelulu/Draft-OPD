import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('confused_token_audit.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AuditTests(unittest.TestCase):
    def test_thresholds_earliest_and_any_later_follower(self):
        p=[[.6,.45],[.95,.02],[.96,.02],[.6,.3],[.95,.02],[.96,.02]]
        self.assertIsNone(a.choose(p,.1))
        self.assertEqual(a.choose(p,.2),1)
        self.assertEqual(a.choose(p,.4),1)
        self.assertEqual(a.choose([[.5,.49],[.95,.02]],.4),1)
        # A distant confident token qualifies; exactly 0.8 does not.
        self.assertEqual(a.choose([[.5,.49],[.6,.3],[.81,.1]],.1),1)
        self.assertIsNone(a.choose([[.5,.49],[.8,.1]],.1))
        self.assertIsNone(a.choose([[.5,.49]],.4))
    def test_categories_and_one_based_first_error(self):
        self.assertEqual(a.category(4,3),'hit_first_error')
        self.assertEqual(a.category(3,3),'false_positive_correct')
        self.assertEqual(a.category(5,3),'after_first_error')
        self.assertEqual(a.category(None,3),'miss')
        self.assertEqual(a.category(4,16),'false_positive_correct')
    def test_paired_summary_excludes_eos_and_counts_uncoverable(self):
        rows=[]
        for accepted,pos,repaired,eos in [(3,4,8,False),(15,None,15,False),(2,3,9,True)]:
            rows.append(dict(accepted=accepted,perfect_accepted=repaired,any_eos=eos,
                rules={str(t):dict(position=pos,category=a.category(pos,accepted),accepted_repaired=repaired) for t in a.THRESHOLDS}))
        for s in a.summarize(rows):
            self.assertEqual(s['blocks_non_eos'],2)
            self.assertEqual(s['hit_first_error'],1)
            self.assertEqual(s['uncoverable_errors'],1)
            self.assertEqual(s['recall_all_errors'],.5)
            self.assertEqual(s['recall_coverable_errors'],1)
            self.assertEqual(s['mean_gain'],2.5)
    def test_position15_is_now_coverable(self):
        probs=[[.99,.001]]*14+[[.5,.49],[.81,.1]]
        self.assertEqual(a.choose(probs,.1),15)
        row=dict(accepted=14,perfect_accepted=16,any_eos=False,
            rules={str(t):dict(position=15,category='hit_first_error',accepted_repaired=16) for t in a.THRESHOLDS})
        for summary in a.summarize([row]):
            self.assertEqual(summary['uncoverable_errors'],0)
            self.assertEqual(summary['recall_coverable_errors'],1)

    def test_empty_summary(self):
        self.assertIsNone(a.summarize([])[0]['mean_gain'])

if __name__=='__main__': unittest.main()
