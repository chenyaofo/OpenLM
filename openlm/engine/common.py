
import logging
import pprint
import dataclasses
import typing

import torch4x
import deepspeed

from torch.utils.collect_env import get_pretty_env_info

from openlm.config import Args


_logger = logging.getLogger(__name__)


def init_for_training(args: Args):
    torch4x.init_logger(filenmae=args.output_dir/"default.log")
    torch4x.set_active_device()

    deepspeed.init_distributed()
    torch4x.create_code_snapshot(name="code", include_suffix=[".py", ".conf"],
                                 source_directory=".", store_directory=args.output_dir)

    args.conf.put("ds_config.train_batch_size", args.conf.get_int("ds_config.train_micro_batch_size_per_gpu") * \
                  torch4x.world_size() * args.conf.get_int("gradient_accumulation_steps"))
    
    _logger.info("Collect envs from system:\n" + get_pretty_env_info())
    _logger.info("Args:\n" + pprint.pformat(dataclasses.asdict(args)))


def train(
    max_epochs: int,
    train_one_epoch_func: typing.Callable,
    **kwargs
):
    eta = torch4x.EstimatedTimeArrival(max_epochs)
    for epoch in range(1, max_epochs+1):
        train_one_epoch_func(epoch=epoch, max_epochs=max_epochs, **kwargs)
        eta.step()
        _logger.info(f"Epoch={epoch:04d} complete, {eta}")
