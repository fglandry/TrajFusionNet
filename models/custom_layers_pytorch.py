import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """ Use this class when the attention is not location-aware"""

    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.qkv_dim = input_dim
        self.input_dim = input_dim

        self.query = nn.Linear(self.input_dim, self.qkv_dim)
        self.key = nn.Linear(self.input_dim, self.qkv_dim)
        self.value = nn.Linear(self.input_dim, self.qkv_dim)

        self.attention = ScaledDotProductAttention(self.input_dim)
        
    def forward(self, x, y):
        queries = self.query(x)
        keys = self.key(y)
        values = self.value(y)

        context, attn = self.attention(
            queries,
            keys,
            values,
            #mask=None
        )

        #weighted = context.flatten(1, 2).unsqueeze(2) # ToDo: verify
        weighted = context

        return weighted, attn


class SelfAttention(nn.Module):
    """ Use this class when the attention is not location-aware"""

    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.qkv_dim = input_dim
        self.input_dim = input_dim

        self.query = nn.Linear(self.input_dim, self.qkv_dim)
        self.key = nn.Linear(self.input_dim, self.qkv_dim)
        self.value = nn.Linear(self.input_dim, self.qkv_dim)

        self.attention = ScaledDotProductAttention(self.input_dim)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        context, attn = self.attention(
            queries,
            keys,
            values,
            #mask=None
        )

        weighted = context.flatten(1, 2).unsqueeze(2)
        return weighted, attn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn
