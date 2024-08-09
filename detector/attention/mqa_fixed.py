import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, num_heads, key_dim, value_dim, dropout=0):
        super(MultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout

        self.query_proj = nn.Parameter(torch.randn(num_heads, key_dim, key_dim))
        self.key_proj = nn.Parameter(torch.randn(key_dim, key_dim))
        self.value_proj = nn.Parameter(torch.randn(key_dim, value_dim))
        self.output_proj = nn.Parameter(torch.randn(key_dim, num_heads, value_dim))
        self.dropout_layer = nn.Dropout(dropout)

    def reshape_input(self, t):
        batch_size, dims, channels = t.size()
        return t.view(batch_size, dims, channels)

    def forward(self, x):
        # For self-attention, query, key, and value are derived from the same input x
        reshaped_x = self.reshape_input(x)
        reshaped_m = self.reshape_input(x)  # m is the same as x for self-attention

        q = torch.einsum('bnd,hkd->bnhk', reshaped_x, self.query_proj)
        k = torch.einsum('bmd,dk->bmk', reshaped_m, self.key_proj)
        logits = torch.einsum('bnhk,bmk->bnhm', q, k)

        logits = logits / torch.sqrt(torch.tensor(self.key_dim, dtype=x.dtype))
        attention_scores = self.dropout_layer(F.softmax(logits, dim=-1))

        v = torch.einsum('bmd,dv->bmv', reshaped_m, self.value_proj)
        o = torch.einsum('bnhm,bmv->bnhv', attention_scores, v)
        result = torch.einsum('bnhv,dhv->bnd', o, self.output_proj)

        return result.view_as(x)