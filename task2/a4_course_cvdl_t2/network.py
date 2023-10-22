from torch import nn
import torch
from a4_course_cvdl_t2.head import CenterNetHead
from a4_course_cvdl_t2.backbone import ResnetBackbone
from a4_course_cvdl_t2.convert import PointsToObjects
from torch.nn import functional as F

class PointsNonMaxSuppression(nn.Module):
    """
    Описан в From points to bounding boxes, фильтрует находящиеся
    рядом объекты.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, points):
        heat = points[:, :-4].max(1).values.unsqueeze(1)
        kernel = self.kernel_size
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()

        return points * keep


class ScaleObjects(nn.Module):
    """
    Объекты имеют размеры в пикселях, и размер входа в сеть
    в несколько раз больше (R, output_stride), чем размер выхода.
    Из-за этого все объекты, полученные с помощью PointsToObjects
    имеют меньший размер.
    Чтобы это компенисровать, надо увеличить размеры объектов.
    """
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, objects):
        b, n, d6 = objects.shape
        objects[:, :, :4] *= self.scale
        return objects


class CenterNet(nn.Module):
    """
    Детектор объектов из статьи 'Objects as Points': https://arxiv.org/pdf/1904.07850.pdf
    """
    def __init__(self, pretrained=True, head_kwargs={}, nms_kwargs={}, points_to_objects_kwargs={}):
        super().__init__()
        self.backbone = ResnetBackbone(pretrained)
        self.head = CenterNetHead(**head_kwargs)
        self.return_objects = torch.nn.Sequential(
            PointsNonMaxSuppression(**nms_kwargs),
            PointsToObjects(**points_to_objects_kwargs),
            ScaleObjects()
        )

    def forward(self, input_t, return_objects=False):
        x = input_t
        x = self.backbone(x)
        x = self.head(x)
        if return_objects:
            x = self.return_objects(x)
        return x
