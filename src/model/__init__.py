from .sequential_net import SequentialNet

model_se_25 = SequentialNet(
    num_class=12,
    input_size_temporal=(16, 300, 25, 2),
    len_feature_new=[64, 32, 32],
    num_block=3,
    dropout=0.2,
    num_head=8,
)
