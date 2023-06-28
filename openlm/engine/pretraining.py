import math
import pathlib
import typing
import itertools

import torch
import deepspeed
import torch4x

import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils import logger as _logger
from deepspeed.utils import log_dist
from pyhocon import ConfigTree
from torch4x.register import REGISTRY
from torch4x import ThroughputTester, time_enumerate
from torch4x import AverageMetric, EstimatedTimeArrival
from torch4x import world_size, only_rank_0, set_sampler_epoch

from openlm.config import Args, get_args
from .common import init_for_training, batch2device

CHECKPOINT_TAG = "lastest"

def prepare_for_training(conf: ConfigTree, outdir: str):
    model, tokenizer = REGISTRY.build_from(conf.get("model"), dict(ds_config=conf.get("ds_config")))
    model: nn.Module

    train_dataloader = REGISTRY.build_from(conf.get("data"), dict(tokenizer=tokenizer))

    optimizer = REGISTRY.build_from(conf.get("optimizer"), dict(params=[p for p in model.parameters() if p.requires_grad]))

    lr_scheduler = REGISTRY.build_from(conf.get("lr_scheduler"), dict(optimizer=optimizer))

    writer = only_rank_0(SummaryWriter(outdir))

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=conf.get("ds_config"),
        lr_scheduler=lr_scheduler,
        dist_init_required=True
    )

    return model, optimizer, lr_scheduler, train_dataloader, writer


def train(
    start_steps: int,
    max_steps: int,
    model_engine: DeepSpeedEngine,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    loader: DataLoader,
    writer: SummaryWriter,
    device: str,
    log_interval: int,
    ckpt_interval: int,
    outdir: str
):
    model_engine.train()

    # set_sampler_epoch(epoch, loader)

    time_cost_metric = AverageMetric("time_cost")
    loss_metric = AverageMetric("loss")
    eta = EstimatedTimeArrival(max_steps - start_steps)
    thput = ThroughputTester()

    for time_cost, step, batch in time_enumerate(itertools.cycle(loader), start=start_steps):
        batch_on_device = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
        # batch2device(batch, device)
        # import ipdb; ipdb.set_trace()
        outputs = model_engine(**batch_on_device, use_cache=False)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

        writer.add_scalar("loss", loss.item(), global_step=step)

        time_cost_metric.update(time_cost)
        loss_metric.update(loss)
        eta.step()
        thput.update(batch_on_device["labels"])

        if step % log_interval == 0:
            perplexity = math.exp(loss_metric.compute())
            log_dist(", ".join([
                f"step={step:05d}/{max_steps:05d}",
                f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
                f"fps={thput.compute()*torch4x.world_size():.0f} samples/s",
                f"{loss_metric}",
                f"perplexity={perplexity:.2f}",
                f"{torch4x.formated_cuda_info()}",
                f"{eta}",
            ]), ranks=[0])
            time_cost_metric.reset()
            loss_metric.reset()

        if step % ckpt_interval == 0:
            model_engine.save_checkpoint(outdir, tag=CHECKPOINT_TAG, client_state=dict(step=step))


def main(args: Args = get_args()):
    init_for_training(args)
    conf: ConfigTree = args.conf
    outdir: pathlib.Path = args.outdir

    model, optimizer, lr_scheduler, train_dataloader, writer = prepare_for_training(conf, outdir)
    model: DeepSpeedEngine

    if model._get_all_ckpt_names(outdir, tag=CHECKPOINT_TAG):
        _, client_state = model.load_checkpoint(outdir, tag=CHECKPOINT_TAG)
        client_state:dict
        start_steps = client_state.get("step", 1)
        log_dist(f"Load checkpoint from {outdir} at step={start_steps}")
    else:
        start_steps = 1

    train(
        start_steps=start_steps,
        max_steps=conf.get_int("max_steps"),
        model_engine=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loader=train_dataloader,
        writer=writer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_interval=conf.get_int("log_interval"),
        ckpt_interval=conf.get_int("ckpt_interval"),
        outdir=outdir
    )
