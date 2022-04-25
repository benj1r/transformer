import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab,
            trg_vocab,
            src_pad,
            trg_pad,
            embed_size,
            num_layers,
            fwd_expansion,
            heads,
            dropout,
            device,
            max_len):
        self.encoder = Encoder(
                src_vocab,
                embed_size,
                num_layers,
                heads,
                device,
                fwd_expansion,
                dropout,
                max_len
                )

        self.decoder = Decoder(
                trg_vocab,
                embed_size,
                num_layers,
                heads,
                fwd_expansion,
                dropout,
                device,
                max_len
                )
        
        self.src_pad = src_pad
        self.trg_pad = trg_pad

    def gen_src_mask(self, src):
        src_mask = (src != self.src_pad).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def gen_trg_mask(self, trg):
        N, trg_len = trg.shape
        mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return mask.to(self.device)

    def forward(self, src, trg):
        src_mask = gen_src_mask(src)
        trg_mask = gen_trg_mask(src)

        enc = self.encoder(src, src_mask)
        dec = self.decoder(trg, enc, src_mask, trg_mask)
        return dec

