from transformers import get_scheduler, SchedulerType

from torch4x.register import REGISTRY


def lr_scheduler_from_hf_transformer(lr_scheduler_type, num_warmup_steps, num_training_steps, optimizer):
    return get_scheduler(
        name=SchedulerType(lr_scheduler_type),
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

REGISTRY.register(lr_scheduler_from_hf_transformer)