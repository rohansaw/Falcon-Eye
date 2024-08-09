import math
import torch
from torch import nn, Tensor
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, size, channels, device="cpu"):
        super().__init__()
        self.channels = channels // 2
        scale = 10000
        
        # Create position arrays for height and width
        y_pos = np.arange(size)[:, np.newaxis]
        x_pos = np.arange(size)[:, np.newaxis]
        
        # Create angle calculations for height and width
        i = np.arange(self.channels)
        angle_freqs = np.power(scale, 2 * i / self.channels)
        angle_freqs_h = angle_freqs.reshape(1, -1)
        angle_freqs_w = angle_freqs.reshape(1, -1)
        
        # Generate positional encodings for height and width
        h_encodings = np.concatenate([np.sin(y_pos / angle_freqs_h), np.cos(y_pos / angle_freqs_h)], axis=1)
        w_encodings = np.concatenate([np.sin(x_pos / angle_freqs_w), np.cos(x_pos / angle_freqs_w)], axis=1)
        
        # Convert to tensors
        h_encodings = torch.tensor(h_encodings, dtype=torch.float32).unsqueeze(1)
        w_encodings = torch.tensor(w_encodings, dtype=torch.float32).unsqueeze(0)
        
        # Tile across the respective dimensions
        h_encodings = h_encodings.repeat(1, size, 1)
        w_encodings = w_encodings.repeat(size, 1, 1)
        
        # Combine encodings for the final positional encodings tensor
        encodings = h_encodings + w_encodings
        self.encodings = encodings.permute(2, 0, 1).unsqueeze(0).to(device)

    def forward(self, x):
        return x + self.encodings


class PositionalEncodingOld(nn.Module):
    """https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device="cpu"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
