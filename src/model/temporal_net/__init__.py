from torch import nn

from .transformer import TransformerEncoder


class StreamTemporalGCN(nn.Module):
    def __init__(
        self,
        num_head,
        num_block,
        dropout,
        len_feature_new,
        input_size_temporal=(3, 300, 25, 2),
        **kargs
    ):
        super(StreamTemporalGCN, self).__init__()

        C, T, V, M = input_size_temporal
        channels = C
        num_frame = T

        self.ln0 = nn.LayerNorm(normalized_shape=(num_frame, channels))

        self.transformer = TransformerEncoder(
            input_size_transformer=(num_frame, channels),
            len_feature_new=len_feature_new,
            num_block=num_block,
            len_seq=num_frame,
            dropout=dropout,
            num_head=num_head,
        )

        self.linear1 = nn.Linear(len_feature_new[num_block - 1], 32)
        self.ln1 = nn.LayerNorm(normalized_shape=(300, 32))

        self.linear2 = nn.Linear(32, 32)
        self.ln2 = nn.LayerNorm(normalized_shape=(300, 32))

        self.linear3 = nn.Linear(32, 1)

    def forward(self, X):
        N_0, C_0, T_0, V_0, M_0 = X.size()
        X = X.permute(0, 4, 3, 2, 1).contiguous().view(N_0 * M_0, V_0, T_0, C_0)
        X = X.contiguous().view(-1, T_0, C_0)
        X = self.transformer(X)
        X = X.contiguous().view(N_0 * M_0 * V_0, T_0, -1).mean(-1).squeeze()
        X = X.contiguous().view(N_0 * M_0, -1, T_0).mean(1).squeeze()
        X = X.view(N_0, M_0, -1)
        X = X.mean(1)

        return X
