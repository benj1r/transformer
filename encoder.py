import torch
import torch.nn as nn

from attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            fwd_expansion):
        super(EncoderBlock, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.fwd_expansion = fwd_expansion
        
        self.MHA = MultiHeadAttention(self.embed_size, self.heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)
        self.feed_fwd = nn.Sequential(
                nn.Linear(embed_size, fwd_expansion * embed_size),
                nn.RelU(),
                nn.Linear(embed_size * fwd_expansion, embed_size)
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attention = self.MHA(values, keys, queries, mask)
        x = self.dropout(self.norm1(attention + query))
        fwd = self.feed_fwd(x)
        out = self.dropout(self.norm2(fwd + x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            fwd_expansion,
            dropout,
            max_len
            ):
        pass

    def forward(self, x, mask):
        pass

