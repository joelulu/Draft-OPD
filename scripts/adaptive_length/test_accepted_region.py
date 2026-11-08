import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('accepted_region',Path(__file__).with_name('analyze_accepted_region.py'))
exp=importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def fixture():
    row=dict(arm='mixed',prompt_key='abc',prompt_index=13,split='test',generated_position=192,
             trials={},pairs={})
    for k,a in zip(exp.KS,[3,2,2,2,8,8,8,8]):
        positions=[]
        for p in range(1,k+1):
            token=1 if p==1 and k==8 else 0
            positions.append(dict(position=p,top5_ids=[token,2,3,4,5],top5_probs=[.6,.1,.1,.1,.1],
                top1_top2_margin=1.,reference_prob=.3 if token else .6,
                reference_hit=not token,reference_margin=-1. if token else 1.))
        row['trials'][str(k)]=dict(accepted=a,eos_adjusted=False,positions=positions)
    for small,large in zip(exp.KS,exp.KS[1:]):
        positions=[]
        for p in range(1,small+1):
            changed=(row['trials'][str(small)]['positions'][p-1]['top5_ids'][0] !=
                     row['trials'][str(large)]['positions'][p-1]['top5_ids'][0])
            positions.append(dict(position=p,js=.1,top1_equal=not changed,top5_overlap=.8 if changed else 1.,
                top5_order_equal=not changed,peak_delta=.02,margin_delta=.03))
        row['pairs'][f'{small}:{large}']=dict(positions=positions)
    return row


class RegionTests(unittest.TestCase):
    def test_unknown_positions_are_not_zero_or_stable(self):
        row=fixture()
        self.assertIsNone(exp.position_history(row,8,4))
        p=exp.position_history(row,4,4)
        self.assertEqual(p['observed_k_count'],1)
        self.assertIsNone(p['path_js_mean'])
        self.assertIsNone(p['ever_top1_changed'])

    def test_oscillation_is_not_hidden_by_equal_endpoints(self):
        p=exp.position_history(fixture(),1,16)
        self.assertEqual(p['top1_flip_count'],2)
        self.assertTrue(p['ever_top1_changed'])
        self.assertFalse(p['net_top1_changed'])
        self.assertTrue(p['changed_then_returned'])
        self.assertAlmostEqual(p['current_vs_history_top1_change'],1/3)
        self.assertAlmostEqual(p['current_vs_history_top5_overlap'],(1+.8+1)/3)
        self.assertAlmostEqual(p['path_js_sum'],.3)
        self.assertAlmostEqual(p['path_js_mean'],.1)

    def test_k16_previous12_uses_24_available_position_transitions(self):
        positions,regions,_=exp.analyze_state(fixture())
        r=next(r for r in regions if r['current_k']==16 and r['region']=='previous_prefix')
        self.assertEqual(r['requested_region_end'],12)
        self.assertEqual(r['visible_positions'],12)
        self.assertEqual(r['position_transition_observations'],24)
        self.assertEqual(r['positions_with_history'],12)
        p=next(p for p in positions if p['current_k']==16 and p['position']==12)
        self.assertEqual(p['observed_k_count'],2)
        self.assertEqual(p['transition_count'],1)

    def test_true_accept8_and_capped_k4(self):
        _,regions,_=exp.analyze_state(fixture())
        r=next(r for r in regions if r['current_k']==4 and r['region']=='best_accepted')
        self.assertEqual(r['observed_max_accept'],8)
        self.assertEqual(r['smallest_best_k'],20)
        self.assertEqual(r['visible_positions'],4)
        self.assertEqual(r['positions_with_history'],0)
        r=next(r for r in regions if r['current_k']==16 and r['region']=='best_accepted')
        self.assertEqual(r['visible_positions'],8)
        self.assertEqual(r['position_transition_observations'],20)

    def test_zero_accept_region_stays_empty(self):
        row=fixture()
        for trial in row['trials'].values(): trial['accepted']=0
        _,regions,_=exp.analyze_state(row)
        for r in regions:
            if r['region']=='best_accepted':
                self.assertEqual(r['visible_positions'],0)
                self.assertIsNone(r['path_js_edge_weighted'])

    def test_cli_reads_saved_data_and_writes_examples_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            folder=root/'gsm8k'
            folder.mkdir()
            row=fixture()
            payload=json.dumps(row)+'\n'
            (folder/'states.jsonl').write_text(payload)
            (folder/'token_legend.json').write_text(json.dumps({str(i):str(i) for i in range(6)}))
            result=subprocess.run([sys.executable,str(Path(exp.__file__)),'--root',str(root),
                '--output',str(root/'report')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((folder/'states.jsonl').read_text(),payload)
            self.assertTrue((root/'report/gsm8k_mixed_prompt13_state192.md').exists())
            self.assertTrue((root/'report/cumulative_regions.csv').exists())
            self.assertIn('A*=8',result.stdout)


if __name__=='__main__':
    unittest.main()
