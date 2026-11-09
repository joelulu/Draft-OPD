import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('top5_matrix',Path(__file__).with_name('export_top5_matrix.py'))
exp=importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def fixture():
    row=dict(arm='mixed',prompt_key='p',prompt_index=13,generated_position=0,
        prefix_sha256=exp.signature([1,2]),anchor_id=9,context_length=2,trials={})
    verification={}
    for k in exp.KS:
        row['trials'][str(k)]=dict(accepted=min(8,k),block_ids=[9]+[1]*k,positions=[dict(position=i+1,
            top5_ids=[1,2,3,4,5],top5_probs=[.5,.2,.1,.05,.01],reference_id=2,
            reference_prob=.2,verify_target_id=3) for i in range(k)])
        verification[str(k)]=[dict(position=i+1,token_id=3,probability=.7) for i in range(k)]
    ref={k:row[k] for k in ('prompt_key','prefix_sha256','anchor_id','generated_position')}
    ref.update(target_fingerprint='finger',positions=[dict(position=i+1,token_id=2,probability=.9) for i in range(32)],
               verification_by_arm={'mixed':verification})
    return row,ref


class MatrixTests(unittest.TestCase):
    def test_export_keeps_draft_and_two_target_probabilities_separate(self):
        row,ref=fixture()
        exp.validate_reference(ref,{'mixed':row},'finger')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'matrix'
            exp.export({'mixed':row},ref,out,12)
            with (out/'matrix_mixed.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(rows[4]['K4'],'')
            self.assertEqual(float(rows[0]['target_probability']),.9)
            self.assertIn('Tverify: 3 @ 0.7',rows[0]['K4'])
            with (out/'top5_long.csv').open() as f:
                long=list(csv.DictReader(f))
            first=long[0]
            self.assertEqual(float(first['draft_probability']),.5)
            self.assertEqual(float(first['target_ar_probability']),.9)
            self.assertEqual(float(first['target_verify_probability']),.7)
            self.assertEqual(len(long),sum(min(k,12)*5 for k in exp.KS))
            page=(out/'matrix.html').read_text()
            self.assertIn('Target AR',page)
            self.assertIn('after an earlier draft error',page)
            self.assertIn('0.900000',page)

    def test_reference_mismatch_is_rejected(self):
        row,ref=fixture()
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            exp.validate_reference(ref,{'mixed':row},'wrong')
        bad=copy.deepcopy(ref)
        bad['positions'][0]['token_id']=30
        with self.assertRaisesRegex(ValueError,'stored reference'):
            exp.validate_reference(bad,{'mixed':row},'finger')
        bad=copy.deepcopy(ref)
        bad['verification_by_arm']['mixed']['4'][0]['token_id']=30
        with self.assertRaisesRegex(ValueError,'verification reference'):
            exp.validate_reference(bad,{'mixed':row},'finger')

    def test_cpu_cli_reuses_existing_target_reference(self):
        row,ref=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            folder=root/'gsm8k'
            folder.mkdir()
            (folder/'states.jsonl').write_text(json.dumps(row)+'\n')
            (folder/'metadata.json').write_text(json.dumps(dict(fingerprints=dict(target='finger'))))
            (root/'ref.json').write_text(json.dumps(ref))
            result=subprocess.run([sys.executable,str(Path(exp.__file__)),'--root',str(root),
                '--arm','mixed','--target-reference',str(root/'ref.json'),'--output',str(root/'output')],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((root/'output/matrix.html').exists())


class RealTargetTests(unittest.TestCase):
    def test_real_qwen_target_AR_and_perK_verification_probabilities(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM,DynamicCache
        torch.set_num_threads(1)
        torch.manual_seed(9)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
            max_position_embeddings=128,attn_implementation='sdpa')).eval().requires_grad_(False)
        ids=torch.tensor([[1,2,3]])
        cache=DynamicCache()
        prefix=ids.clone()
        with torch.inference_mode():
            output=model(ids,past_key_values=cache,use_cache=True,output_hidden_states=True,logits_to_keep=1)
            anchor=output.logits[:,-1:].argmax(-1)
            for _ in range(3):
                output=model(anchor,past_key_values=cache,use_cache=True,output_hidden_states=True,logits_to_keep=1)
                prefix=torch.cat((prefix,anchor),dim=1)
                anchor=output.logits[:,-1:].argmax(-1)
            expected=dict(prefix_sha256=exp.signature(prefix[0].tolist()),anchor_id=int(anchor.item()))
            refs,reconstructed=exp.reconstruct_reference(model,ids,3,None,torch,expected)
            torch.testing.assert_close(prefix,reconstructed)
            self.assertEqual(len(refs),32)
            row=dict(anchor_id=expected['anchor_id'],trials={})
            c=DynamicCache()
            model(prefix,past_key_values=c,use_cache=True,output_hidden_states=True,logits_to_keep=1)
            for k in exp.KS:
                block=prefix.new_tensor([[expected['anchor_id']]+[i%31 for i in range(k)]])
                c.crop(prefix.shape[1])
                out=model(block,position_ids=torch.arange(prefix.shape[1],prefix.shape[1]+k+1)[None],
                          past_key_values=c,use_cache=True,output_hidden_states=True)
                target_ids=out.logits[0,:k].argmax(-1).tolist()
                row['trials'][str(k)]=dict(block_ids=block[0].tolist(),
                    positions=[dict(verify_target_id=t) for t in target_ids])
            verify=exp.verify_saved_blocks(model,prefix,{'mixed':row},torch)
            for k in exp.KS:
                self.assertEqual(len(verify['mixed'][str(k)]),k)
                self.assertTrue(all(0<p['probability']<=1 for p in verify['mixed'][str(k)]))
                # The first candidate conditions only on common context+anchor.
                self.assertEqual(verify['mixed'][str(k)][0]['token_id'],refs[0]['token_id'])
                self.assertAlmostEqual(verify['mixed'][str(k)][0]['probability'],refs[0]['probability'],places=5)


if __name__=='__main__':
    unittest.main()
