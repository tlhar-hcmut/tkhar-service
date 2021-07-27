import math

import torch
from torch import nn
from torch.nn.modules.container import ModuleList

from .encoder_block import EncoderBlock
from .position import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            input_size_transformer,
            len_feature_new=[64, 128, 256],
            len_seq=300,
            num_head=3,
            dropout=0,
            num_block=3,
            **kwargs):

        super(TransformerEncoder, self).__init__(**kwargs)
        self.len_feature_input = input_size_transformer[-1]
        self.len_feature_new = len_feature_new
        self.pos_encoding = PositionalEncoding(self.len_feature_input, dropout)

        self.blks = nn.Sequential()

        for i in range(num_block):
            module = EncoderBlock(input_size_transformer,
                                  len_feature_new[i], num_head, dropout)
            self.blks.add_module(str(i), module)
            input_size_transformer = (
                *input_size_transformer[:-1], len_feature_new[i])

    def forward(self, X, mask, is_output_att=False):
        # X = self.pos_encoding(X * math.sqrt(self.len_feature_input))
        X = self.pos_encoding(X, mask)

        # add container for attentiosn
        X = [X, []]
        X, mask, is_output_att = self.blks((X, mask, is_output_att))
        if is_output_att:
            return X[1]
        return X[0]
