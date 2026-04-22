import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.tgcn import ConvTemporalGraphical
from .utils.graph import Graph


class ShiftTemporalGraphical(nn.Module):
    """Shift-based approximation for spatial graph convolution.

    It replaces explicit adjacency multiplication with feature shifts
    followed by a 1x1 projection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, shift_mode='spatial'):
        super().__init__()
        self.kernel_size = kernel_size
        self.shift_mode = shift_mode
        self.proj = nn.Conv2d(in_channels * kernel_size, out_channels, kernel_size=1, bias=True)

    def _roll(self, x, shift_index):
        if self.shift_mode == 'temporal':
            return torch.roll(x, shifts=shift_index, dims=2)
        if self.shift_mode == 'both':
            if shift_index % 2 == 0:
                return torch.roll(x, shifts=shift_index, dims=2)
            return torch.roll(x, shifts=shift_index, dims=3)
        return torch.roll(x, shifts=shift_index, dims=3)

    def forward(self, x, A):
        shifted = [x]
        for k in range(1, self.kernel_size):
            shifted.append(self._roll(x, k))
        x = torch.cat(shifted, dim=1)
        x = self.proj(x)
        return x.contiguous(), A

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        valid_stgcn_kwargs = [
            'dropout',
            'use_shift_gcn',
            'shift_mode',
            'use_bottleneck',
            'bottleneck_ratio',
            'bottleneck_layers',
        ]
        kwargs_filtered = {k: v for k, v in kwargs.items() if k in valid_stgcn_kwargs}

        default_channels = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        default_strides = [1, 1, 1, 1, 2, 1, 1, 2, 1, 1]
        self.channel_cfg = kwargs.get('channel_cfg', default_channels)
        if len(self.channel_cfg) != len(default_strides):
            raise ValueError('channel_cfg length must be 10, got {}'.format(len(self.channel_cfg)))
        self.strides = kwargs.get('stride_cfg', default_strides)
        if len(self.strides) != len(self.channel_cfg):
            raise ValueError('stride_cfg length must match channel_cfg length')

        bottleneck_layers = kwargs_filtered.get('bottleneck_layers', [8, 9, 10])
        self.bottleneck_layers = set(int(x) for x in bottleneck_layers)
        use_bottleneck = bool(kwargs_filtered.get('use_bottleneck', False))
        bottleneck_ratio = int(kwargs_filtered.get('bottleneck_ratio', 4))

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs_filtered.items() if k != 'dropout'}
        layers = []
        prev_channels = in_channels
        for i, out_channels in enumerate(self.channel_cfg):
            layer_id = i + 1
            stride = self.strides[i]
            block_kwargs = dict(kwargs_filtered)
            block_kwargs.pop('bottleneck_layers', None)
            if i == 0:
                block_kwargs = dict(kwargs0)
                block_kwargs.pop('bottleneck_layers', None)
            block_kwargs['use_bottleneck'] = use_bottleneck and (layer_id in self.bottleneck_layers)
            block_kwargs['bottleneck_ratio'] = bottleneck_ratio
            layers.append(
                st_gcn(
                    prev_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    residual=False if i == 0 else True,
                    **block_kwargs
                )
            )
            prev_channels = out_channels
        self.st_gcn_networks = nn.ModuleList(layers)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(self.channel_cfg[-1], num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

    def bn_l1_loss(self):
        reg = torch.tensor(0.0, device=self.A.device)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and module.weight is not None:
                reg = reg + module.weight.abs().sum()
        return reg

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 use_shift_gcn=False,
                 shift_mode='spatial',
                 use_bottleneck=False,
                 bottleneck_ratio=4):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.use_bottleneck = use_bottleneck
        if use_shift_gcn:
            self.gcn = ShiftTemporalGraphical(in_channels, out_channels, kernel_size[1], shift_mode=shift_mode)
        else:
            self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        if self.use_bottleneck:
            bottleneck_channels = max(out_channels // max(bottleneck_ratio, 1), 16)
            self.bottleneck_reduce = nn.Conv2d(out_channels, bottleneck_channels, kernel_size=1)
            self.bn_reduce = nn.BatchNorm2d(bottleneck_channels)
            self.bottleneck_t = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=(3, 1),
                stride=(stride, 1),
                padding=(1, 0),
                bias=False,
            )
            self.bn_t = nn.BatchNorm2d(bottleneck_channels)
            self.bottleneck_v = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
                bias=False,
            )
            self.bn_v = nn.BatchNorm2d(bottleneck_channels)
            self.bottleneck_expand = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
            self.bn_expand = nn.BatchNorm2d(out_channels)
            self.drop = nn.Dropout(dropout, inplace=True)
        else:
            self.tcn = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size[0], 1),
                    (stride, 1),
                    padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )

        if not residual:
            class Zero(nn.Module):
                def forward(self, x): return 0
            self.residual = Zero()

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        if self.use_bottleneck:
            x = self.bottleneck_reduce(x)
            x = self.bn_reduce(x)
            x = self.relu(x)
            x = self.bottleneck_t(x)
            x = self.bn_t(x)
            x = self.relu(x)
            x = self.bottleneck_v(x)
            x = self.bn_v(x)
            x = self.relu(x)
            x = self.bottleneck_expand(x)
            x = self.bn_expand(x)
            x = self.drop(x)
            x = x + res
        else:
            x = self.tcn(x) + res

        return self.relu(x), A