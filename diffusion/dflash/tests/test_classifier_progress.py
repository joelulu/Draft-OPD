"""Dependency-free progress/ETA regression checks."""
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'adaptive_length/two_stage/progress.py'
spec = importlib.util.spec_from_file_location('classifier_progress_under_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.rows = []
        self.progress = module.Progress('train', 1, 200, every=100, seconds=30,
            clock=lambda: self.now, emit=lambda line, **kw: self.rows.append((json.loads(line),kw)))

    def test_initial_report_has_no_false_eta(self):
        self.progress.report(0)
        row, options = self.rows[-1]
        self.assertIsNone(row['eta_s'])
        self.assertEqual(row['percent'],0)
        self.assertTrue(options['flush'])

    def test_count_time_and_completion_each_trigger_progress(self):
        self.assertTrue(self.progress.due(1))
        self.assertFalse(self.progress.due(2))
        self.assertTrue(self.progress.due(100))
        self.now=30
        self.assertTrue(self.progress.due(2))
        self.progress.report(2)
        self.assertFalse(self.progress.due(3))
        self.assertTrue(self.progress.due(200))

    def test_throughput_eta_and_losses_are_serialized(self):
        self.now=20
        self.progress.report(50, optimizer_updates=3, train_loss={'full_kv_accept':1.5})
        row,_=self.rows[-1]
        self.assertEqual(row['states_per_s'],2.5)
        self.assertEqual(row['eta_s'],60.)
        self.assertEqual(row['train_loss']['full_kv_accept'],1.5)
        self.now=80
        self.progress.report(200)
        self.assertEqual(self.rows[-1][0]['eta_s'],0.)

    def test_invalid_limits_rejected(self):
        for kwargs in ({'total':0}, {'total':10,'every':0}, {'total':10,'seconds':0}):
            with self.assertRaises(ValueError):
                module.Progress('train',1,**kwargs)


if __name__=='__main__':
    unittest.main()
