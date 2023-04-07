import os
import logging
import pathlib
import shutil
import typing

import torch
import torch.distributed as dist

_logger = logging.getLogger(__name__)


def distributed_init(dist_backend, init_method, world_size, rank):
    if world_size <= 0:
        raise ValueError(f"'world_size' should be greater than zero, but got {world_size}")
    if world_size > 1:
        _logger.info(f"Init distributed mode(backend={dist_backend}, "
                     f"init_mothod={init_method}, "
                     f"rank={rank}, pid={os.getpid()}, world_size={world_size}, "
                     f"is_master={is_master()}).")
        dist.init_process_group(backend=dist_backend, init_method=init_method,
                                world_size=world_size, rank=rank)


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