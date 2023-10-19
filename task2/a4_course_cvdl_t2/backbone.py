"""
Здесь находится backbone на основе resnet-18, в статье "Objects as Points" он описан в
5.Implementation details/Resnet и в Figure 6-b.
"""

import torch
from torch import nn

from torchvision.models import resnet18

from typing import List, Type, Union, Optional


class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 64,
    ) -> None:
        super(_BasicBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class _Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 64,
    ) -> None:
        super(_Bottleneck, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels

        channels = int(out_channels * (base_channels / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (stride, stride), (1, 1), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, int(out_channels * self.expansion), (1, 1), (1, 1), (0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_channels * self.expansion))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class HeadlessPretrainedResnet18Encoder(nn.Module):
    """
    Предобученная на imagenet версия resnet, у которой
    нет avg-pool и fc слоев.
    Принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(self):
        super().__init__()
        md = resnet18(pretrained=True)
        # все, кроме avgpool и fc
        self.md = nn.Sequential(
            md.conv1,
            md.bn1,
            md.relu,
            md.maxpool,
            md.layer1,
            md.layer2,
            md.layer3,
            md.layer4
        )

    def forward(self, x):
        return self.md(x)


class HeadlessResnet18Encoder(nn.Module):
    """
    Версия resnet, которую надо написать с нуля.
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(
            self,
            groups: int = 1,
            channels_per_group: int = 64,
    ) -> None:
        # полносверточная сеть, архитектуру можно найти в
        # https://arxiv.org/pdf/1512.03385.pdf, Table1
        super(HeadlessResnet18Encoder, self).__init__()
        self.in_channels = 64
        self.dilation = 1
        self.groups = groups
        self.base_channels = channels_per_group

        self.conv1 = nn.Conv2d(3, self.in_channels, (7, 7), (2, 2), (3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d((3, 3), (2, 2), (1, 1))

        self.layer1 = self._make_layer(2, _BasicBlock, 64, 1)
        self.layer2 = self._make_layer(2, _BasicBlock, 128, 2)
        self.layer3 = self._make_layer(2, _BasicBlock, 256, 2)
        self.layer4 = self._make_layer(2, _BasicBlock, 512, 2)


    def _make_layer(
            self,
            repeat_times: int,
            block: Type[Union[_BasicBlock, _Bottleneck]],
            channels: int,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, (1, 1), (stride, stride), (0, 0),
                          bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = [
            block(
                self.in_channels,
                channels,
                stride,
                downsample,
                self.groups,
                self.base_channels
            )
        ]
        self.in_channels = channels * block.expansion
        for _ in range(1, repeat_times):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    1,
                    None,
                    self.groups,
                    self.base_channels,
                )
            )

        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out




class UpscaleTwiceLayer(nn.Module):
    """
    Слой, повышающий height и width в 2 раза.
    В реализации из "Objects as Points" используются Transposed Convolutions с
    отсылкой по деталям к https://arxiv.org/pdf/1804.06208.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, output_padding=0):
        super().__init__()
        # dilation = (1 + 2 * padding - output_padding) // (kernel_size - 1)
        self.upscaling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, \
                                            padding=padding, output_padding=output_padding + 1, \
                                            stride=2)

    def forward(self, x):
        out = self.upscaling(x)

        return out


class ResnetBackbone(nn.Module):
    """
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, C, H/R, W/R], где R = 4.
    C может быть выбрано разным, в конструкторе ниже C = 64.
    """
    def __init__(self, pretrained: bool = True, out_channels=64):
        super().__init__()
        # downscale - fully-convolutional сеть, снижающая размерность в 32 раза
        if pretrained:
            self.downscale = HeadlessPretrainedResnet18Encoder()
        else:
            self.downscale = HeadlessResnet18Encoder()

        # upscale - fully-convolutional сеть из UpscaleTwiceLayer слоев, повышающая размерность в 2^3 раз
        downscale_channels = 512 # выход resnet
        channels = [downscale_channels, 256, 128, out_channels]
        layers_up = [
            UpscaleTwiceLayer(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ]
        self.upscale = nn.Sequential(*layers_up)

    def forward(self, x):
        x = self.downscale(x)
        x = self.upscale(x)
        return x

