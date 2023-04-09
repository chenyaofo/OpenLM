
import torch
import torch.nn as nn


def get_n_trainable_parameters(model: nn.Module):
    """
    Return the number of trainable parameters in the model.
    """
    n_trainable_params = 0
    n_all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()

        n_all_param += num_params
        if param.requires_grad:
            n_trainable_params += num_params
    return n_all_param, n_trainable_params
