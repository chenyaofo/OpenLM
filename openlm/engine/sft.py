
import functools
from openlm.utils.common import ThroughputTester, time_enumerate
from openlm.utils.metrics import AverageMetric, EstimatedTimeArrival
from openlm.utils.distributed import world_size, is_dist_avail_and_init, local_rank
import torch.utils.data as data

import torch.nn as nn
import torch.optim as optim
import logging

from torch.utils.tensorboard import SummaryWriter

from pyhocon import ConfigTree

import torch
from openlm import REGISTRY
from openlm.config import Args
from openlm.utils.distributed import only_master

from openlm.utils.common import set_sampler_epoch, get_current_device

from .common import init_for_training, train

import pathlib
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, CPUOffload

import deepspeed

import torch4x

_logger = logging.getLogger(__name__)


def prepare_for_training(conf: ConfigTree, output_dir: str):

    model, tokenizer = REGISTRY.build_from(conf.get("model"), dict(ds_config=conf.get("ds_config")))
    # model.half()
    train_dataloader = REGISTRY.build_from(conf.get("data"), dict(tokenizer=tokenizer))

    optimizer = REGISTRY.build_from(conf.get("optimizer"), dict(params=[p for p in model.parameters() if p.requires_grad]))

    writer = only_master(SummaryWriter(output_dir))

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=conf.get("ds_config"),
        # lr_scheduler=lr_scheduler,
        dist_init_required=True)
    
    return model, optimizer, train_dataloader, writer


_logger = logging.getLogger(__name__)


def train_one_epoch(
    epoch: int,
    max_epochs:int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: data.DataLoader,
    writer: SummaryWriter,
    device: str,
    log_interval: int
):
    model.train()

    set_sampler_epoch(epoch, loader)

    time_cost_metric = AverageMetric("time_cost")
    loss_metric = AverageMetric("loss")
    eta = EstimatedTimeArrival(len(loader))
    speed_tester = ThroughputTester()

    for time_cost, iter_, (input_ids, labels, attention_mask) in time_enumerate(loader, start=1):
        input_ids = input_ids.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        attention_mask = attention_mask.to(device=device, non_blocking=True)

        # optimizer.zero_grad()
        # output = model(input_ids, attention_mask=attention_mask, labels=labels)
        # loss: torch.Tensor = output.loss
        # loss.backward()
        # optimizer.step()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        loss = outputs.loss
        model.backward(loss)
        model.step()

        writer.add_scalar("loss", loss.item())

        time_cost_metric.update(time_cost)
        loss_metric.update(loss)
        eta.step()
        speed_tester.update(input_ids)

        if iter_ % log_interval == 0 or iter_ == len(loader):
            _logger.info(", ".join([
                f"epoch={epoch:04d}/{max_epochs:04d}",
                f"iter={iter_:05d}/{len(loader):05d}",
                f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} samples/s",
                f"{loss_metric}",
                f"{torch4x.formated_cuda_info()}",
                f"{eta}",
            ]))
            time_cost_metric.reset()
            speed_tester.reset()


def main(args: Args):
    init_for_training(args)
    conf: ConfigTree = args.conf
    output_dir: pathlib.Path = args.output_dir

    model, optimizer, train_dataloader, writer = prepare_for_training(conf, output_dir)

    train(
        max_epochs=conf.get_int("max_epochs"),
        train_one_epoch_func=train_one_epoch,
        model=model,
        optimizer=optimizer,
        loader=train_dataloader,
        writer=writer,
        device=torch4x.get_active_device(),
        log_interval=conf.get_int("log_interval")
    )
