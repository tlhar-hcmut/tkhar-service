import math

import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(
        self, len_feature_input_selfA, len_feature_hidden_selfA, dropout=0, **kwargs
    ):

        super(SelfAttention, self).__init__(**kwargs)
        self.len_feature_hidden_selfA = len_feature_hidden_selfA
        self.W_k = nn.Linear(
            len_feature_input_selfA, len_feature_hidden_selfA, bias=False
        )
        self.W_q = nn.Linear(
            len_feature_input_selfA, len_feature_hidden_selfA, bias=False
        )
        self.W_v = nn.Linear(
            len_feature_input_selfA, len_feature_hidden_selfA, bias=False
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        K = self.W_k(X)
        Q = self.W_q(X)
        V = self.W_v(X)

        K_T = K.permute(0, 2, 1)
        scores = torch.matmul(Q, K_T) / math.sqrt(self.len_feature_hidden_selfA)
        scores = F.softmax(scores, -1)
        scores = self.dropout(scores)

        attention = torch.matmul(scores, V)
        return attention

