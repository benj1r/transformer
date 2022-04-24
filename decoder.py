import torch
import torch.nn as nn

from attention import MultiHeadAttention
from encoder import EncoderBlock

class DecoderBlock(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            fwd_expansion,
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
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.device = device
        self.fwd_expansion = fwd_expansion
        self.dropout = dropout
        self.max_len = max_len
        
        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
                [ 
                    DecoderBlock(self.embed_size,
                        self.heads,
                        self.dropout,
                        self.fwd_expansion,
                        self.device)
                    for _ in range(self.num_layers)
                    ]
                )
        
        self.fc = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape

        positions = torch.arange(0,seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.vocab_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc(x)
        return out
