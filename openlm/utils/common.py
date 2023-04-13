
import time
import os

import torch
import torch.nn as nn
import torch.utils.data as data

from .distributed import is_dist_avail_and_init


def get_n_trainable_parameters(model: nn.Module):
    """
    Return the number of trainable parameters in the model.
    """
    n_trainable_params = 0
    n_all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()

        n_all_param += num_params
        if param.requires_grad:
            n_trainable_params += num_params
    return n_all_param, n_trainable_params


class time_enumerate:
    def __init__(self, seq, start=0, infinite=False):
        self.seq = seq
        self.start = start
        self.counter = self.start-1
        self.infinite = infinite

    def __iter__(self):
        self.seq_iter = iter(self.seq)
        return self

    def __next__(self):
        while True:
            try:
                start_time = time.perf_counter()
                item = next(self.seq_iter)
                end_time = time.perf_counter()
                self.counter += 1
                return end_time-start_time, self.counter, item
            except StopIteration:
                if self.infinite:
                    self.__iter__()
                else:
                    raise StopIteration


class ThroughputTester():
    def __init__(self):
        self.reset()

    def reset(self):
        self.batch_size = 0
        self.start = time.perf_counter()

    def update(self, tensor):
        batch_size, *_ = tensor.shape
        self.batch_size += batch_size
        self.end = time.perf_counter()

    def compute(self):
        if self.batch_size == 0:
            return 0
        else:
            return self.batch_size/(self.end-self.start)


def set_sampler_epoch(
    epoch: int,
    dataloader: data.DataLoader,
):
    if is_dist_avail_and_init():
        if hasattr(dataloader, "sampler"):
            dataloader.sampler.set_epoch(epoch)


CURRENT_DEVICE = None


def set_proper_device(local_rank: int = None):
    global CURRENT_DEVICE
    if torch.cuda.is_available():
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", None))
        if local_rank is None:
            raise ValueError(f"Can not set device to {local_rank}")
        torch.cuda.set_device(local_rank)
        CURRENT_DEVICE = torch.cuda.current_device()
    else:
        CURRENT_DEVICE = "cpu"


def get_current_device():
    global CURRENT_DEVICE
    return CURRENT_DEVICE
