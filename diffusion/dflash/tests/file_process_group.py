"""Test-only synchronous FileStore collectives, for socket-disabled sandboxes.

Exercises real DDP reducers across real processes without a network transport.
Production and normal server tests use NCCL/Gloo, never this backend.
"""
from datetime import timedelta
import io
import torch
import torch.distributed as dist


class Completed(dist.Work):
    def __init__(self,tensors):
        super().__init__()
        self.future=torch.futures.Future()
        self.future.set_result(tensors)
    def wait(self,timeout=None):
        return True
    def is_completed(self):
        return True
    def get_future(self):
        return self.future


class FileProcessGroup(dist.ProcessGroup):
    def __init__(self,store,rank,size,timeout):
        super().__init__(rank,size)
        self.store,self.rank_id,self.world,self.counter=store,rank,size,0
        self.store.set_timeout(timedelta(seconds=60))
    def getBackendName(self):
        return 'file_test'
    def exchange(self,tensors):
        prefix=f'{self.counter}'
        self.counter+=1
        stream=io.BytesIO()
        torch.save([t.detach().clone() for t in tensors],stream)
        self.store.set(f'{prefix}/{self.rank_id}',stream.getvalue())
        return [torch.load(io.BytesIO(self.store.get(f'{prefix}/{rank}')),weights_only=True) for rank in range(self.world)]
    def allreduce(self,tensors,opts=None):
        rows=self.exchange(tensors)
        with torch.no_grad():
            for i,tensor in enumerate(tensors):
                tensor.copy_(sum(row[i] for row in rows))
        return Completed(tensors)
    def broadcast(self,tensors,opts):
        rows=self.exchange(tensors)
        with torch.no_grad():
            for i,tensor in enumerate(tensors):
                tensor.copy_(rows[opts.rootRank][i])
        return Completed(tensors)
    def allgather(self,outputs,inputs,opts=None):
        rows=self.exchange(inputs)
        with torch.no_grad():
            for i,group in enumerate(outputs):
                for rank,tensor in enumerate(group):
                    tensor.copy_(rows[rank][i])
        return Completed([t for group in outputs for t in group])
    def barrier(self,opts=None):
        self.exchange([])
        return Completed([])


def register():
    if 'file_test' not in dist.Backend.backend_list:
        dist.Backend.register_backend('file_test',lambda store,rank,size,timeout:FileProcessGroup(store,rank,size,timeout),devices=['cpu'])
