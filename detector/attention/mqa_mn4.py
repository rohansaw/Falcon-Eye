import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_2d(inp, oup, kernel_size, stride, norm=True, act=True):
    layers = [nn.Conv2d(inp, oup, kernel_size, stride, bias=False)]
    if norm:
        layers.append(nn.BatchNorm2d(oup))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class MultiQueryAttention(nn.Module):
    def __init__(self, inp, num_heads, key_dim=None, value_dim=None, dropout=0.0):
        """Multi Query Attention"""
        super().__init__()
        key_dim = key_dim if key_dim is not None else inp
        value_dim = value_dim if value_dim is not None else inp
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout

        self.head_dim = key_dim // num_heads

        self._query_proj = conv_2d(inp, num_heads * key_dim, 1, 1, norm=False, act=False)
        self._key_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)
        self._value_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)

        self._output_proj = conv_2d(num_heads * key_dim, inp, 1, 1, norm=False, act=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x should have the shape [batch_size, h * w, channels]
        batch_size, seq_length, channels = x.size()
        
        # Reshape x to [batch_size, channels, h, w] to apply 2D convolutions
        h_w = int(seq_length ** 0.5)
        x = x.permute(0, 2, 1).view(batch_size, channels, h_w, h_w)

        q = self._query_proj(x)
        q = q.view(batch_size, self.num_heads, self.key_dim, -1).permute(0, 1, 3, 2) # [batch_size, num_heads, seq_length, key_dim]

        k = self._key_proj(x)
        v = self._value_proj(x)
        k = k.view(batch_size, 1, self.key_dim, -1) # [batch_size, 1, key_dim, seq_length]
        v = v.view(batch_size, 1, -1, self.key_dim) # [batch_size, 1, seq_length, key_dim]

        # calculate attn score
        attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, dim=-1)

        context = torch.matmul(attn_score, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_heads * self.key_dim, h_w, h_w)
        output = self._output_proj(context)
        output = output.view(batch_size, channels, h_w * h_w).permute(0, 2, 1) # [batch_size, seq_length, channels]
        return output
