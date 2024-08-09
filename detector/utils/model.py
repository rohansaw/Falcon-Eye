import numpy as np
import torch


def as_logit(p):
    return np.log(p / (1 - p))


def set_bias_prior(model, datamodule):
    layer = model.head.logits
    weights = layer.weight.data
    biases = layer.bias.data

    logits = as_logit(datamodule.prior)
    layer.bias.data = torch.tensor(logits, dtype=torch.float32)
