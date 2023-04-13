import torch.optim as optim

from openlm.utils.register import REGISTRY

REGISTRY.register(optim.Adam)
REGISTRY.register(optim.AdamW)
REGISTRY.register(optim.Adamax)