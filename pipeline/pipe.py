from functools import partial
from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    num_cycles = num_batches + num_partitions - 1

    for t in range(num_cycles):
        schedule = []

        for i in range(num_partitions):
            micro_batch_idx = t - i

            if 0 <= micro_batch_idx < num_batches:
                schedule.append((micro_batch_idx, i))
        
        yield schedule

    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN SOLUTION
        micro_batches = list(x.split(self.split_size, dim=0))

        schedule = _clock_cycles(len(micro_batches), len(self.partitions))

        for sched in schedule:
            self.compute(micro_batches, sched)
        
        return torch.cat([b.to(self.devices[-1]) for b in micro_batches], dim=0)
        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN SOLUTION
        for micro_batch_id, partition_id in schedule:
            device = devices[partition_id]

            micro_batch = batches[micro_batch_id].to(device)
            partition = partitions[partition_id]

            compute = partial(partition, micro_batch)
            task = Task(compute)

            self.in_queues[partition_id].put(task) 
        
        for micro_batch_id, partition_id in schedule:
            success, result = self.out_queues[partition_id].get() # each partition_id only has 1 task

            if not success:
                _, e, traceback = result
                raise e.with_traceback(traceback) 
            
            _, output = result
            batches[micro_batch_id] = output
        # END SOLUTION

