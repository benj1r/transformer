import torch
import torch.nn as nn


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
        pass

    def gen_src_mask(self, src):
        pass

    def gen_trg_mask(self, trg):
        pass

    def forward(self, src, trg):
        pass

