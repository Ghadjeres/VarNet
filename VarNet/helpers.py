import torch
from torch import nn


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return tensor.to('cuda:0')
    else:
        return tensor


def to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
