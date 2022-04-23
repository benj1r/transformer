import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            dropout,
            fwd_expansion):
        pass

    def forward(self, values, keys, queries, mask):
        pass


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

