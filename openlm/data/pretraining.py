import os

import torch

from torch.utils.data import DataLoader
from torchdata.datapipes.iter import FileLister, FileOpener, TFRecordLoader, Mapper, Shuffler, Batcher, Collator, ShardingFilter
from torch4x.register import REGISTRY


@REGISTRY.register
def pretraining_dataset(
    root: str,
    batch_size: int,
    num_workers: int,
):

    dp = FileLister(root, masks="*.tfrecord", non_deterministic=False)
    dp = ShardingFilter(dp)
    dp = FileOpener(dp, mode="rb")
    dp = TFRecordLoader(dp, spec={
        "metadata": (tuple(), None),
        "tokens": (tuple(), torch.int32),
    })
    dp = Shuffler(dp, buffer_size=int(os.environ.get("TFREC_BUFFER_SIZE", 5000)))
    dp = Mapper(dp, fn=lambda content: content["tokens"])
    # dp = Batcher(dp, batch_size=10)
    # dp = Collator(dp)

    loader = DataLoader(
        dp,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return loader
