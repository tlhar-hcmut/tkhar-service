import util
import numpy as np
import torch
from torch import nn

from .ntu import NtuGraph


class StreamSpatialGCN(nn.Module):
    def __init__(self, name="spatial", in_channels=3, **kargs):
        super(StreamSpatialGCN, self).__init__()
        self.name = name
        self.graph = NtuGraph()

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(150)

        self.l1 = AAGCNBlock(in_channels, 8, A, residual=False)
        self.l2 = AAGCNBlock(8, 8, A)
        self.l3 = AAGCNBlock(8, 8, A)
        self.l4 = AAGCNBlock(8, 8, A)
        self.l5 = AAGCNBlock(8, 16, A)
        self.l6 = AAGCNBlock(16, 16, A)
        self.l7 = AAGCNBlock(16, 16, A)
        util.init_bn(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        return x.view(N, M, x.size(1), T, V).permute(0, 2, 3, 4, 1).contiguous()


class AAGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(AAGCNBlock, self).__init__()
        self.agcLayer = AGCLayer(in_channels, out_channels, A)
        self.tConv1 = TConv(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = ConvNorm(
                in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1)
            )

    def forward(self, x):
        return self.relu(self.tConv1(self.agcLayer(x)) + self.residual(x))


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)):
        super(ConvNorm, self).__init__()
        pad = tuple((s - 1) // 2 for s in kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        util.init_conv(self.conv)
        util.init_bn(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class TConv(ConvNorm):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
        )


class AGCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(AGCLayer, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()

        self.anti_zero = nn.Parameter(
            torch.from_numpy(A.astype(np.float32)), requires_grad=False
        )
        nn.init.constant_(self.anti_zero, 1e-6)

        self.A_adaptive = nn.Parameter(
            torch.from_numpy(A.astype(np.float32)), requires_grad=True
        )

        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels),
            )
        else:
            self.res = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                util.init_conv(m)
            elif isinstance(m, nn.BatchNorm2d):
                util.init_bn(m, 1)
        util.init_bn(self.bn, 1e-6)
        for i in range(self.num_subset):
            util.init_conv_branch(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A_adaptive + self.anti_zero

        fusion = None
        for i in range(self.num_subset):
            A1 = (
                # [-1, V, C, T] -> [-1, V, C_inter, T]
                self.conv_a[i](x)
                # -> [-1, V, C_inter, T]
                .permute(0, 3, 1, 2).contiguous()
                # -> [-1, V, C_inter*T]
                .view(N, V, self.inter_c * T)
            )
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)

            # N V V
            C_k = self.soft(torch.matmul(A1, A2) / A1.size(-1))
            BC_k = C_k + A[i]

            v = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(v, BC_k).view(N, C, T, V))
            fusion = (z + fusion) if fusion is not None else z

        fusion = self.bn(fusion)
        fusion += self.res(x)
        return self.relu(fusion)

