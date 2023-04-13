import pathlib
import logging
import pprint
import dataclasses
import typing

from pyhocon import ConfigTree
from torch.utils.collect_env import get_pretty_env_info
import torch.distributed as dist

from openlm.config import Args
from openlm.utils.logging import init_logger, create_code_snapshot
from openlm.utils.distributed import rank, distributed_init
from openlm.utils.metrics import AverageMetric, EstimatedTimeArrival
from openlm.utils.common import set_proper_device


_logger = logging.getLogger(__name__)


def _init(args: Args):
    distributed_init()

    set_proper_device()

    init_logger(rank=rank(), filenmae=args.output_dir/"default.log")

    create_code_snapshot(name="code", include_suffix=[".py", ".conf"],
                         source_directory=".", store_directory=args.output_dir)

    _logger.info("Collect envs from system:\n" + get_pretty_env_info())
    _logger.info("Args:\n" + pprint.pformat(dataclasses.asdict(args)))


def train(
    max_epochs: int,
    train_one_epoch_func: typing.Callable,
    **kwargs
):
    eta = EstimatedTimeArrival(max_epochs)
    for epoch in range(1, max_epochs+1):
        train_one_epoch_func(epoch=epoch, **kwargs)
        eta.step()
        _logger.info(f"Epoch={epoch:04d} complete, {eta}")
