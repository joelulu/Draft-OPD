"""Shared tail policy and optional event timing for the B=1 evaluation."""
from contextlib import contextmanager
import time
import torch


def actual_length(requested, remaining):
    """One shared budget rule, even for untrained tail widths. Zero means AR."""
    if requested < 1 or remaining < 1:
        raise ValueError("Requested width and remaining output budget must be positive.")
    return min(requested, remaining - 1)


def fixed_lengths(lengths, native_block_size, extra=()):
    # Runtime supports at most 32 candidates. Native block_size includes anchor.
    native = int(native_block_size) - 1
    values = sorted(set(lengths) | set(extra) | ({native} if 1 <= native <= 32 else set()))
    if not values or any(k < 1 or k > 32 for k in values):
        raise ValueError("Fixed candidate lengths must be within [1, 32].")
    return values


class StageTimer:
    """CUDA events resolve only after decode's final sync; no per-stage sync."""
    names = ("head_ms", "draft_ms", "verify_ms", "ar_ms")

    def __init__(self, enabled, device):
        self.enabled, self.cuda = enabled, device.type == "cuda"
        self.events = []
        self.totals = {name: 0.0 if enabled else None for name in self.names}

    @contextmanager
    def measure(self, name):
        if self.cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            try:
                yield
            finally:
                end.record()
                self.events.append((name, begin, end))
        else:
            begin = time.perf_counter()
            try:
                yield
            finally:
                self.totals[name] += (time.perf_counter() - begin) * 1000

    def resolve(self):
        for name, begin, end in self.events:
            self.totals[name] += begin.elapsed_time(end)
        return self.totals
