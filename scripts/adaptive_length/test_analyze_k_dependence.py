import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('k_dependence', Path(__file__).with_name('analyze_k_dependence.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(accepted=None):
    accepted = accepted or [min(k, 6) for k in audit.KS]
    return dict(round=0, context_length=100, trials=[dict(requested_k=k, actual_k=k,
        accepted_raw=a, emitted=a+1, block_ids=[99]+list(range(k)))
        for k, a in zip(audit.KS, accepted)])


class AuditTests(unittest.TestCase):
    def test_truncation_invariant_and_capacity_limit_not_counted(self):
        row, pairs, skip = audit.analyze_state(fixture())
        self.assertIsNone(skip)
        self.assertFalse(row['needs_extra_width'])
        self.assertFalse(row['any_truncation_accept_mismatch'])
        self.assertFalse(any(p['extra_width_helps'] for p in pairs))
        self.assertEqual(row['smallest_k_for_max'], 8)
        self.assertEqual(row['smallest_grid_k_with_capacity'], 8)

    def test_six_accept_at16_but_shorter_capacity8_loses(self):
        record = fixture([2, 3, 4, 6, 5, 5, 5, 5])
        record['trials'][1]['block_ids'][3] = 123
        row, pairs, _ = audit.analyze_state(record)
        self.assertTrue(row['needs_extra_width'])
        self.assertEqual(row['smallest_grid_k_with_capacity'], 8)
        self.assertEqual(row['smallest_k_for_max'], 16)
        pair = next(p for p in pairs if p['small_k']==8 and p['large_k']==16)
        self.assertTrue(pair['extra_width_helps'])
        self.assertTrue(pair['changed_within_large_accepted_prefix'])
        self.assertEqual(pair['extra_width_accept_gain'], 3)

    def test_identical_candidates_acceptance_disagreement_flagged(self):
        row, pairs, _ = audit.analyze_state(fixture([2, 3, 4, 6, 5, 5, 5, 5]))
        self.assertTrue(row['any_identical_prefix_accept_mismatch'])

    def test_zero_accept_and_exclusions(self):
        row, _, _ = audit.analyze_state(fixture([0]*8))
        self.assertFalse(row['needs_extra_width'])
        self.assertIsNone(row['smallest_grid_k_with_capacity'])
        record = fixture()
        record['trials'][-1].update(actual_k=31, block_ids=[99]+list(range(31)))
        self.assertEqual(audit.analyze_state(record)[2], 'tail_clipped')
        record = fixture()
        record['trials'][0]['emitted'] = 1
        self.assertEqual(audit.analyze_state(record)[2], 'eos_adjusted')

    def test_anchor_mismatch_rejected(self):
        record = fixture()
        record['trials'][0]['block_ids'][0] = -1
        with self.assertRaisesRegex(ValueError, 'anchors'):
            audit.analyze_state(record)

    def test_cli_both_arms_exports_and_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = dict(objective='accept', trials=[fixture(), fixture([0]*8)])
            for arm in ('fixed','mixed'):
                folder = root/'schedules'/arm
                folder.mkdir(parents=True)
                (folder/'prompt_accept.json').write_text(json.dumps(payload))
            before = (root/'schedules/fixed/prompt_accept.json').read_bytes()
            result = subprocess.run([sys.executable, str(Path(audit.__file__)), '--root', str(root),
                '--output', str(root/'audit')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root/'audit/summary.json').read_text())
            self.assertEqual(summary['fixed']['states'], 2)
            self.assertEqual(summary['mixed']['needs_extra_width_count'], 0)
            self.assertEqual(before, (root/'schedules/fixed/prompt_accept.json').read_bytes())
            self.assertTrue((root/'audit/pairs.csv').exists())


if __name__ == '__main__':
    unittest.main()
