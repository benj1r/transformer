import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(
            self,
            embed_size,
            heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "embed_size needs to be divisible by heads."
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
    
        self.fc = nn.Linear(heads * head_dim, embed_size)

    def scaled_dot_attention(q, k, mask): 
        e = torch.einsum('nqhd,nkhd->nhqk', [q, k])
        if mask is not None:
            e = e.masked_fill(mask==0, -1e20)
        e = (e / self.embed_size ** (1/2))
        return e

    def forward(self, values, keys, queries):
        N = queries.shape[0]

        values_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = keys.reshape(N, queries_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(values)
        queries = self.queries(values)

        attention = torch.softmax(scaled_dot_attention(values, keys, queries, mask),dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
                N, queries_len, self.heads * self.head_dim)
        out = self.fc(out)
        return out
