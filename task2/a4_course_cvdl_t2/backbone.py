"""
Здесь находится backbone на основе resnet-18, в статье "Objects as Points" он описан в
5.Implementation details/Resnet и в Figure 6-b.
"""
from torch import nn
from torchvision.models import resnet18
from torchvision.models.inception import BasicConv2d


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


class ResnetBlock(nn.Module):

  def __init__(self, cin, cout, stride=1, k=3):
    super().__init__()
    self.fun = nn.Sequential(
      nn.Conv2d(cin, cout, k, stride, 1),
      nn.BatchNorm2d(cout),
      nn.ReLU(inplace=True),
      nn.Conv2d(cout, cout, k, 1, 1),
      nn.BatchNorm2d(cout),
      nn.ReLU(inplace=True)
    )
    if stride == 1:
      self.proj = nn.Identity()
    else:
      self.proj = nn.Conv2d(cin, cout, k, stride, 1)
  def forward(self, x):
    projed = self.proj(x)
    f = self.fun(x)
    return f + projed




class HeadlessResnet18Encoder(nn.Module):
    """
    Версия resnet, которую надо написать с нуля.
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(self):
        # полносверточная сеть, архитектуру можно найти в
        # https://arxiv.org/pdf/1512.03385.pdf, Table1
      super().__init__()
      self.fun = nn.Sequential(
        nn.Conv2d(3, 64, 7, 2, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2, 1),

        ResnetBlock(64, 64),
        ResnetBlock(64, 64),

        ResnetBlock(64, 128, 2),
        ResnetBlock(128, 128),

        ResnetBlock(128, 256, 2),
        ResnetBlock(256, 256),

        ResnetBlock(256, 512, 2),
        ResnetBlock(512, 512)
      )

    def forward(self, x):
        return self.fun(x)


class UpscaleTwiceLayer(nn.Module):
    """
    Слой, повышающий height и width в 2 раза.
    В реализации из "Objects as Points" используются Transposed Convolutions с
    отсылкой по деталям к https://arxiv.org/pdf/1804.06208.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, output_padding=1):
        super().__init__()
        self.fun = nn.Sequential(
          nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=output_padding),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fun(x)

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

