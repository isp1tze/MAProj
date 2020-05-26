import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_torch(np_array):
    return torch.from_numpy(np_array)


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(
            tgt_param.data * (1.0 - tau) + src_param.data * tau
        )

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.weight.data.fill_(0.)
        m.bias.data.fill_(0.)
