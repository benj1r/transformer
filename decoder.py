import torch
import torch.nn as nn

from attention import MultiHeadAttention
from encoder import EncoderBlock

class DecoderBlock(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            fwd_expansion,
            dropout,
            device
            ):
        super(DecoderBlock, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.fwd_expansion = fwd_expansion
        self.dropout = dropout
        self.device = device

        self.MHA = MultiHeadAttention(self.embed_size)
        self.norm = nn.LayerNorm(self.embed_size)
        self.block = EncoderBlock(self.embed_size, self.heads, self.dropout, self.fwd_expansion)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, values, keys, src_mask, trg_mask):
        masked_attention = self.MHA(x, x, x, trg_mask)
        queries = self.dropout(self.norm(attention+x))
        out = self.block(values, keys, queries, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            num_layers,
            heads,
            fwd_expansion,
            dropout,
            device,
            max_len
            ):
        pass

    def forward(self, x, enc_out, src_mask, trg_mask):
        pass

