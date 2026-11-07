"""Small, dependency-free JSON progress reporter for long classifier epochs."""
import json
import time


class Progress:
    def __init__(self, phase, epoch, total, every=100, seconds=30., clock=time.monotonic, emit=print):
        if every < 1 or seconds <= 0 or total < 1:
            raise ValueError('Progress limits must be positive')
        self.phase, self.epoch, self.total = phase, epoch, total
        self.every, self.seconds, self.clock, self.emit = every, seconds, clock, emit
        self.start = self.last = clock()

    def due(self, count):
        return count == 1 or count == self.total or count % self.every == 0 or self.clock()-self.last >= self.seconds

    def report(self, count, **metrics):
        now = self.clock()
        elapsed = max(0., now-self.start)
        rate = count/elapsed if elapsed else 0.
        row = dict(event='classifier_progress', phase=self.phase, epoch=self.epoch,
                   states=count, total_states=self.total, percent=100*count/self.total,
                   elapsed_s=round(elapsed, 2), states_per_s=round(rate, 3),
                   eta_s=round((self.total-count)/rate, 2) if rate else None, **metrics)
        self.emit(json.dumps(row), flush=True)
        self.last = now
