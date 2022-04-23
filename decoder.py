import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            fwd_expansion,
            dropout,
            device
            ):
        pass

    def forward(self, x, values, keys, src_mask, trg_mask):
        pass


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

