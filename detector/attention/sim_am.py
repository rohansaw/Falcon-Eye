import torch
import torch.nn as nn
import lightning as L


class SimAM(L.LightningModule):
    """
    Simple Paramter-Free Attention Module, https://proceedings.mlr.press/v139/yang21o.html
    Adapted from https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py
    """

    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )

        return x * self.activaton(y)
