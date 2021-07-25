import torch.nn.functional as F
from torch import nn

from . import spatial_net, temporal_net


class SequentialNet(nn.Module):
    def __init__(self, num_class=60, **kargs):
        super(SequentialNet, self).__init__()

        self.spatial_net = spatial_net.StreamSpatialGCN(**kargs)

        self.temporal_net = temporal_net.StreamTemporalGCN(**kargs)

        self.fc1 = nn.Linear(300, 128)

        self.ln1 = nn.LayerNorm(normalized_shape=(128))

        self.fc2 = nn.Linear(128, 128)

        self.ln2 = nn.LayerNorm(normalized_shape=(128))

        self.fc3 = nn.Linear(128, 64)

        self.ln3 = nn.LayerNorm(normalized_shape=(64))

        self.fc4 = nn.Linear(64, 64)

        self.ln4 = nn.LayerNorm(normalized_shape=(64))

        self.fc5 = nn.Linear(64, num_class)

    def forward(self, x):

        output = self.spatial_net(x)

        output = self.temporal_net(output)

        output = F.relu(self.ln1(self.fc1(output)))

        output = F.relu(self.ln2(self.fc2(output)))

        output = F.relu(self.ln3(self.fc3(output)))

        output = F.relu(self.ln4(self.fc4(output)))

        output = self.fc5(output)

        return output


model_se_25 = SequentialNet(
    num_class=12,
    input_size_temporal=(16, 300, 25, 2),
    len_feature_new=[64, 32, 32],
    num_block=3,
    dropout=0.2,
    num_head=8,
)
