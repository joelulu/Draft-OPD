import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('analyzer', Path(__file__).with_name('analyze_native_eval.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, repeat='.', scale=1):
        for arm in module.ARMS:
            folder = self.root / repeat / arm / 'eval'
            folder.mkdir(parents=True)
            meta = dict(schema=2, native_block_size=16, fixed_lengths=[4, 8],
                        tail_policy='clip', seed=42, max_new_tokens=100, profile_rounds=False)
            (folder / 'metadata_rank000.json').write_text(json.dumps(meta))
            rows = []
            for prompt, durations in [('a', [1, 2]), ('b', [4, 1])]:
                for k, duration in zip([4, 8], durations):
                    rows.append(dict(prompt_key=prompt, mode=f'fixed_{k}', tokens=100,
                                     decode_s=duration*scale, spec_rounds=10,
                                     accepted_tokens=20, spec_committed_tokens=30, profile_rounds=False))
            (folder / 'results_rank000.jsonl').write_text('\n'.join(map(json.dumps, rows)))

    def test_oracle_uses_total_tokens_over_total_time(self):
        self.fixture()
        _, choices, report = module.analyze(self.root)
        self.assertAlmostEqual(report['best_fixed']['fixed']['decode_tok_s'], 200/3)
        self.assertAlmostEqual(report['hindsight_tok_s']['mixed'], 100)
        self.assertAlmostEqual(report['mixed_hindsight_gain_over_fixed_best_pct'], 50)
        self.assertEqual([r['mode'] for r in choices[:2]], ['fixed_4', 'fixed_8'])

    def test_repeats_pooled_and_reported_separately(self):
        self.fixture('repeat_1')
        self.fixture('repeat_2', scale=2)
        _, _, report = module.analyze(self.root)
        self.assertEqual(report['repeats'], 2)
        self.assertAlmostEqual(report['hindsight_tok_s']['mixed'], 400/6)
        self.assertAlmostEqual(report['best_fixed']['fixed']['repeat_median_tok_s'], 50)

    def mutate(self, fn):
        p = self.root / 'mixed/eval/results_rank000.jsonl'
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        fn(rows)
        p.write_text('\n'.join(map(json.dumps, rows)))

    def test_missing_grid_rejected(self):
        self.fixture()
        self.mutate(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            module.analyze(self.root)

    def test_duplicate_rejected(self):
        self.fixture()
        self.mutate(lambda rows: rows.append(rows[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            module.analyze(self.root)

    def test_different_prompt_set_rejected(self):
        self.fixture()
        self.mutate(lambda rows: [r.update(prompt_key='other') for r in rows if r['prompt_key'] == 'a'])
        with self.assertRaisesRegex(ValueError, 'Prompt sets'):
            module.analyze(self.root)

    def test_invalid_timing_rejected(self):
        self.fixture()
        self.mutate(lambda rows: rows[0].update(decode_s=float('nan')))
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            module.analyze(self.root)

    def test_profile_and_metadata_mismatch_rejected(self):
        self.fixture()
        p = self.root / 'mixed/eval/metadata_rank000.json'
        meta = json.loads(p.read_text())
        meta['profile_rounds'] = True
        p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'Profiling'):
            module.analyze(self.root)
        meta.update(profile_rounds=False, max_new_tokens=256)
        p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'Different evaluation'):
            module.analyze(self.root)


if __name__ == '__main__':
    unittest.main()
