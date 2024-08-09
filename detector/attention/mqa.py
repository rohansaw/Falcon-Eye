import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MultiQuerySelfAttention(nn.Module):
    """Implemented based on paper: https://arxiv.org/pdf/1911.02150"""

    def __init__(self, dim_in, dim_k, dim_v, n_query=8):
        super(MultiQuerySelfAttention, self).__init__()
        self.dim_k = dim_k
        self.dim_in = dim_in
        self.proj_q = nn.Parameter(torch.empty(n_query, dim_in, dim_k))
        self.proj_k = nn.Parameter(torch.empty(dim_in, dim_k))
        self.proj_v = nn.Parameter(torch.empty(dim_in, dim_v))
        self.proj_o = nn.Parameter(torch.empty(n_query, dim_in, dim_v))

        init.kaiming_uniform_(self.proj_q)
        init.kaiming_uniform_(self.proj_k)
        init.kaiming_uniform_(self.proj_v)
        init.kaiming_uniform_(self.proj_o)

    def forward(self, x):
        Q = torch.einsum("bnd , hdk->bhnk", x, self.proj_q)
        K = torch.einsum("bmd, dk->bmk", x, self.proj_k)
        V = torch.einsum("bmd, dv->bmv", x, self.proj_v)
        logits = torch.einsum("bhnk , bmk->bhnm", Q, K)
        weights = F.softmax(logits, dim=-1)
        O = torch.einsum("bhnm, bmv->bhnv ", weights, V)
        Y = torch.einsum("bhnv , hdv->bnd ", O, self.proj_o)
        return Y

        # query = self.proj_q(x)
        # key = self.proj_k(x)
        # value = self.proj_v(x)
        # query_dim = query.size(-1)
        # assert query_dim == self.dim_k, "Query dim must be equal to dim_k"
        # similarities = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        #     query_dim
        # )
        # attn = F.softmax(similarities, dim=-1)
        # out = torch.matmul(attn, value)
        # return self.proj_o(out)
