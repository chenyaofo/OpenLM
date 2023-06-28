
import logging
import pprint
import dataclasses
import typing

import torch
import torch4x
import deepspeed

from torch.utils.collect_env import get_pretty_env_info
from deepspeed.utils import logger as _logger
from deepspeed.utils import log_dist

from openlm.config import Args


def batch2device(batch: dict[str, torch.Tensor], device: str):
    return {k:v.to(device=device, non_blocking=True) for k, v in batch.items()}
    # batch_on_device = {}
    # for k, v in batch.items():
    #     try:
    #         batch_on_device[k] = v.to(device=device, non_blocking=True)
    #     except:
    #         batch_on_device[k] = v
    # return batch_on_device


def init_for_training(args: Args):
    # torch4x.init_logger(filenmae=args.outdir/"default.log")
    # torch4x.init_logger()
    torch4x.set_active_device()

    deepspeed.init_distributed()
    torch4x.create_code_snapshot(name="code", include_suffix=[".py", ".conf"],
                                 source_directory=".", store_directory=args.outdir)

    args.conf.put("ds_config.train_batch_size", args.conf.get_int("ds_config.train_micro_batch_size_per_gpu") *
                  torch4x.world_size() * args.conf.get_int("ds_config.gradient_accumulation_steps"))

    log_dist("Collect envs from system:\n" + get_pretty_env_info(), ranks=[0])
    log_dist("Args:\n" + pprint.pformat(dataclasses.asdict(args)), ranks=[0])



