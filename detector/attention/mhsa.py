import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    """Implemented based on paper: https://arxiv.org/pdf/1911.02150"""

    def __init__(self, dim_in, dim_k):
        super(SelfAttentionHead, self).__init__()
        self.query_mat = nn.Linear(dim_in, dim_k)
        self.key_mat = nn.Linear(dim_in, dim_k)
        self.value_mat = nn.Linear(dim_in, dim_k)

    def forward(self, x):
        query = self.query_mat(x)
        key = self.key_mat(x)
        value = self.value_mat(x)
        query_dim = query.size(-1)
        similarities = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(query_dim)
        )
        attn = F.softmax(similarities, dim=-1)
        return torch.matmul(attn, value)


class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, dim_in, dim_k, num_heads=8, dropout=0.1):
        super(MultiHeadedSelfAttention, self).__init__()
        self.dim_in = dim_in
        self.heads = nn.ModuleList(
            [SelfAttentionHead(dim_in, dim_k) for _ in range(num_heads)]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.output_mat = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x):
        head_res = [head(x) for head in self.heads]
        heads_concat = torch.cat(head_res, dim=-1)
        output =  self.output_mat(heads_concat)
        output = self.dropout(output)
        return output
