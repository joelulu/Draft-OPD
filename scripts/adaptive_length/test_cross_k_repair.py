import unittest
from cross_k_repair import select_tokens,summarize_cross,METHODS

class CrossTests(unittest.TestCase):
    def test_vote_and_probability_without_target(self):
        probs={8:{1:.2,2:.7},16:{1:.55,2:.4},24:{1:.1,2:.8}}
        top={8:[2,1],16:[1,2],24:[2,1]}
        r=select_tokens(probs,top,1)
        self.assertEqual(r['vote'],2);self.assertEqual(r['mean_probability'],2)
        self.assertEqual(r['contrast16_8'],1);self.assertEqual(r['contrast24_16'],2)
        self.assertEqual(r['contrast8_16'],2);self.assertEqual(r['contrast16_24'],1)
    def test_missing_k8_preserves_k16_and_ties(self):
        r=select_tokens({16:{1:.5,2:.5},24:{1:.5,2:.5}},{16:[1,2],24:[2,1]},1)
        self.assertEqual(r['contrast16_8'],1);self.assertEqual(r['vote'],1)
        self.assertEqual(r['contrast8_16'],1)
    def test_reverse_uses_same_candidate_pool(self):
        # K8 has an extra favorite token, but both directions stay in K16 top2.
        probs={8:{1:.1,2:.2,3:.6},16:{1:.6,2:.3,3:.01},24:{1:.7,2:.2,3:.01}}
        top={8:[3,2],16:[1,2],24:[1,2]}
        r=select_tokens(probs,top,1)
        self.assertEqual(r['contrast8_16'],2)
        self.assertNotEqual(r['contrast8_16'],3)

    def test_summary_keeps_harm_and_common_eos_filter(self):
        rows=[]
        for eos,accepted,new in [(False,5,2),(False,3,7),(True,1,16)]:
            rows.append(dict(cross_any_eos=eos,accepted=accepted,rules={'0.1':dict(category='hit_first_error',accepted_repaired=9)},
                cross={'0.1':{m:dict(position=3,changed=True,matches_target_at_position=new>accepted,accepted_repaired=new) for m in METHODS}}))
        for s in summarize_cross(rows,[.1]):
            self.assertEqual(s['harmed_blocks'],1);self.assertEqual(s['improved_blocks'],1)
            self.assertEqual(s['mean_gain'],.5);self.assertEqual(s['blocks'],2)

if __name__=='__main__':unittest.main()
