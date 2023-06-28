import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam as RawDeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam

from torch4x.register import REGISTRY

class DeepSpeedCPUAdam(RawDeepSpeedCPUAdam):
    def __init__(self, params, **kwargs):
        super(DeepSpeedCPUAdam, self).__init__(model_params=params, **kwargs)

REGISTRY.register(optim.Adam)
REGISTRY.register(optim.AdamW)
REGISTRY.register(optim.Adamax)
REGISTRY.register(DeepSpeedCPUAdam)
REGISTRY.register(FusedAdam)