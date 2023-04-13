import os
import logging
import pathlib
import shutil
import typing
import inspect
from functools import wraps

import torch
import torch.distributed as dist

_logger = logging.getLogger(__name__)


def distributed_init():
    world_size = int(os.environ.get("WORLD_SIZE", 0))
    if world_size <= 0:
        raise ValueError(f"'world_size' should be greater than zero, but got {world_size}")
    if world_size > 1:
        dist.init_process_group(backend="nccl")


def is_dist_avail_and_init() -> bool:
    """

    Returns:
        bool: True if distributed mode is initialized correctly, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    """

    Returns:
        int: The rank of the current node in distributed system, return 0 if distributed 
        mode is not initialized.
    """
    return dist.get_rank() if is_dist_avail_and_init() else 0

def local_rank() -> int:
    """

    Returns:
        int: The lcoal rank of the current node in distributed system.
    """
    return os.environ.get("LOCAL_RANK", 0)

def world_size() -> int:
    """

    Returns:
        int: The world size of the  distributed system, return 1 if distributed mode is not 
        initialized.
    """
    return dist.get_world_size() if is_dist_avail_and_init() else 1


def is_master() -> bool:
    """

    Returns:
        int: True if the rank current node is euqal to 0. Thus it will always return True if 
        distributed mode is not initialized.
    """
    return rank() == 0


def torchsave(obj: typing.Any, f: str) -> None:
    """A simple warp of torch.save. This function is only performed when the current node is the
    master. It will do nothing otherwise. 

    Args:
        obj (typing.Any): The object to save.
        f (str): The output file path.
    """
    if is_master():
        f: pathlib.Path = pathlib.Path(f)
        tmp_name = f.with_name("tmp.pt")
        torch.save(obj, tmp_name)
        shutil.move(tmp_name, f)


def dummy_func(*args, **kargs):
    pass


class DummyClass:
    def __getattribute__(self, obj):
        return dummy_func


class FakeObj:
    def __getattr__(self, name):
        return do_nothing


def do_nothing(*args, **kwargs) -> FakeObj:
    return FakeObj()


def only_master_fn(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if is_master() or kwargs.get('run_anyway', False):
            kwargs.pop('run_anyway', None)
            return fn(*args, **kwargs)
        else:
            return FakeObj()

    return wrapper


def only_master_cls(cls):
    for key, value in cls.__dict__.items():
        if callable(value):
            setattr(cls, key, only_master_fn(value))

    return cls


def only_master_obj(obj):
    cls = obj.__class__
    for key, value in cls.__dict__.items():
        if callable(value):
            obj.__dict__[key] = only_master_fn(value).__get__(obj, cls)

    return obj


def only_master(something):
    if inspect.isfunction(something):
        return only_master_fn(something)
    elif inspect.isclass(something):
        return only_master_cls(something)
    else:
        return only_master_obj(something)
