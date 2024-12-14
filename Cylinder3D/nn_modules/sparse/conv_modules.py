import spconv
import torch.nn as nn


class SparseConvModules:
    """Helper class containing all sparse convolution modules"""
    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
        return spconv.SubMConv3d(
            in_planes, out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            indice_key=indice_key
        )

    @staticmethod
    def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
        return spconv.SubMConv3d(
            in_planes, out_planes,
            kernel_size=(1, 3, 3),
            stride=stride,
            padding=(0, 1, 1),
            bias=False,
            indice_key=indice_key
        )

    @staticmethod
    def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
        return spconv.SubMConv3d(
            in_planes, out_planes,
            kernel_size=(3, 1, 3),
            stride=stride,
            padding=(1, 0, 1),
            bias=False,
            indice_key=indice_key
        )

    @staticmethod
    def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
        return spconv.SubMConv3d(
            in_planes, out_planes,
            kernel_size=(1, 1, 3),
            stride=stride,
            padding=(0, 0, 1),
            bias=False,
            indice_key=indice_key
        )

    @staticmethod
    def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
        return spconv.SubMConv3d(
            in_planes, out_planes,
            kernel_size=(1, 3, 1),
            stride=stride,
            padding=(0, 1, 0),
            bias=False,
            indice_key=indice_key
        )

    @staticmethod
    def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
        return spconv.SubMConv3d(
            in_planes, out_planes,
            kernel_size=(3, 1, 1),
            stride=stride,
            padding=(1, 0, 0),
            bias=False,
            indice_key=indice_key
        )


class Spconv1x3LeakyReLUBatchNorm(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, indice_key=None):
        super().__init__()
        self.conv = spconv.SubMConv3d(
            in_planes, out_planes,
            kernel_size=(1, 3, 3),
            stride=stride,
            padding=(0, 1, 1),
            bias=False,
            indice_key=indice_key
        )
        self.bn = nn.BatchNorm1d(out_planes)
        self.act = nn.LeakyReLU()

    def forward(self, x: spconv.SparseConvTensor):
        x = self.conv(x)
        x.features = self.act(x.features)
        x.features = self.bn(x.features)
        return x


class Spconv3x1LeakyReLUBatchNorm(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, indice_key=None):
        super().__init__()
        self.conv = SparseConvModules.conv3x1(
            in_planes, out_planes, stride, indice_key)
        self.bn = nn.BatchNorm1d(out_planes)
        self.act = nn.LeakyReLU()

    def forward(self, x: spconv.SparseConvTensor):
        x = self.conv(x)
        x.features = self.act(x.features)
        x.features = self.bn(x.features)
        return x


class Spconv3x3LeakyReLUBatchNorm(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, indice_key=None):
        super().__init__()
        self.conv = SparseConvModules.conv3x3(
            in_planes, out_planes, stride, indice_key)
        self.bn = nn.BatchNorm1d(out_planes)
        self.act = nn.LeakyReLU()

    def forward(self, x: spconv.SparseConvTensor):
        x = self.conv(x)
        x.features = self.act(x.features)
        x.features = self.bn(x.features)
        return x


class Spconv1x1x3SigmoidBatchNorm(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, indice_key=None):
        super().__init__()
        self.conv = SparseConvModules.conv1x1x3(
            in_planes, out_planes, stride, indice_key)
        self.bn = nn.BatchNorm1d(out_planes)
        self.act = nn.Sigmoid()

    def forward(self, x: spconv.SparseConvTensor):
        x = self.conv(x)
        x.features = self.bn(x.features)  # strange... why before activation?
        x.features = self.act(x.features)
        return x


class Spconv1x3x1SigmoidBatchNorm(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, indice_key=None):
        super().__init__()
        self.conv = SparseConvModules.conv1x3x1(
            in_planes, out_planes, stride, indice_key)
        self.bn = nn.BatchNorm1d(out_planes)
        self.act = nn.Sigmoid()

    def forward(self, x: spconv.SparseConvTensor):
        x = self.conv(x)
        x.features = self.bn(x.features)
        x.features = self.act(x.features)
        return x


class Spconv3x1x1SigmoidBatchNorm(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, indice_key=None):
        super().__init__()
        self.conv = SparseConvModules.conv3x1x1(
            in_planes, out_planes, stride, indice_key)
        self.bn = nn.BatchNorm1d(out_planes)
        self.act = nn.Sigmoid()

    def forward(self, x: spconv.SparseConvTensor):
        x = self.conv(x)
        x.features = self.bn(x.features)
        x.features = self.act(x.features)
        return x
