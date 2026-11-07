"""Execute real worker methods on CPU without importing CUDA-only SGLang deps."""
import ast
import random
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]


def worker_class():
    path = ROOT/'sglang-dflash/python/sglang/srt/speculative/dflash_worker.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DFlashWorker')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_select_round_width','_ensure_draft_block_buffers')]
    mini = ast.Module(body=[ast.ClassDef(name='WorkerMethods',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(mini)
    namespace = dict(torch=torch)
    exec(compile(mini,str(path),'exec'),namespace)
    return namespace['WorkerMethods']


def make_worker(seed):
    worker = worker_class()()
    worker._candidate_lengths = (4,8,12,16,20,24,28,32)
    worker._length_rng = random.Random(seed)
    worker.block_size = 33
    worker.device = 'cpu'
    worker._draft_block_spec_info = SimpleNamespace(draft_token_num=33)
    worker.model_runner = SimpleNamespace(attn_backend=SimpleNamespace(speculative_num_draft_tokens=33))
    worker.draft_model_runner = SimpleNamespace(attn_backend=SimpleNamespace(speculative_num_draft_tokens=33))
    worker._draft_block_ids_buf = None
    worker._block_pos_offsets = torch.arange(33)
    return worker


def test_actual_width_buffers_and_backend_metadata_follow_every_round():
    a,b = make_worker(42),make_worker(42)
    seen=set()
    for _ in range(80):
        for worker in (a,b):
            worker._select_round_width()
            worker._ensure_draft_block_buffers(3)
            assert worker._draft_block_ids_buf.shape == (3,worker.block_size)
            assert worker._draft_block_spec_info.draft_token_num == worker.block_size
            assert worker.model_runner.attn_backend.speculative_num_draft_tokens == worker.block_size
            assert worker.draft_model_runner.attn_backend.speculative_num_draft_tokens == worker.block_size
            assert worker._block_pos_offsets.numel() == worker.block_size
        assert a.block_size == b.block_size  # TP ranks with the same decode schedule agree.
        seen.add(a.block_size-1)
    assert seen == set(a._candidate_lengths)
