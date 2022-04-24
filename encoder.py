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
        super(Encoder, self).__init__()
        
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
                    EncoderBlock(self.embed_size,
                        self.heads,
                        self.dropout,
                        self.fwd_expansion)
                    for _ in range(self.num_layers)
                    ]
                )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape

        positions = torch.arange(0,seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.vocab_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
