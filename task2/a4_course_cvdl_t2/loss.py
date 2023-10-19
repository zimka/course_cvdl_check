import torch
from torch import nn
import torch.nn.functional as F
from a4_course_cvdl_t2.convert import ObjectsToPoints


class CenterNetLoss(nn.Module):
    """
    Вычисляет 3 лосса для CenterNet, описанные в разделах 3-4 статьи https://arxiv.org/pdf/1904.07850.pdf
    1. Lk - focal loss между предсказанными точками и gt-объектами, "размазанными" в гауссианы.
    2. Loff - L1 лосс поправок dx, dy
    3. Lsize - L1 лосс размеров w, h
    Принимает на вход:
    - predict[B, C+4, W/R, H/R] - выходы CenterNet в виде heatmaps.
    - target[B, N, 5] - gt в виде объектов (xc, yc, wx, hy, class);
    """

    def __init__(self, l_size_lambda=0.1, l_offset_lambda=1, r_scale=4, obj_to_points=None, **kwargs):
        super().__init__()
        self.l_size_lambda = l_size_lambda
        self.l_offset_lambda = l_offset_lambda
        self.r_scale = r_scale
        if obj_to_points is None:
            obj_to_points = ObjectsToPoints(**kwargs)
        self.obj_to_points = obj_to_points

    def forward(self, pred_heatmaps, target_objects):
        """
        Получает на вход:
        pred_heatmaps: тензор[B, C+4, W/R, H/R] с вероятностями, поправками и размерами боксов
        target_objects: тензор[B, N, 6] c (xc, yc, w, h, c, confidence) детекциями.
        Оставлен реализованным для упрощения.
        """
        target_objects = target_objects.clone()
        target_objects[:, :, :-2] /= self.r_scale

        target_heatmaps = self.obj_to_points.forward(target_objects)
        assert target_heatmaps.shape == pred_heatmaps.shape, (target_heatmaps.shape, pred_heatmaps.shape)
        num_classes = pred_heatmaps.size(1) - 4

        target_probs = target_heatmaps[:, :num_classes]
        pred_probs = pred_heatmaps[:, :num_classes]

        # создаем маску, какие объекты являются реальными детекциями, а какие "фальшивыми" со значениями 0
        is_real_object = (target_probs.sum(axis=1, keepdims=True)>=1.0).byte()
        num_real_objects = is_real_object.sum(axis=(1, 2, 3))

        # считаем лоссы
        lk = self.loss_fl(pred_probs, target_probs) / (num_real_objects + 1)

        pred_offsets = pred_heatmaps[:, -4:-2]
        target_offsets = target_heatmaps[:, -4:-2]
        loff = self.loss_l1(pred_offsets, target_offsets, is_real_object) / (num_real_objects + 1)

        pred_sizes = pred_heatmaps[:, -2:]
        target_sizes = target_heatmaps[:, -2:]
        lsize = self.loss_l1(pred_sizes, target_sizes, is_real_object) / (num_real_objects + 1)
        return torch.stack([lk, self.l_offset_lambda * loff, lsize * self.l_size_lambda ], axis=-1)

    def loss_fl(self, predict_cyx, target_cyx, alpha=2, beta=4):
        """
        Focal loss между двумя heatmap. В статье параметры FL alpha=2, beta=4.
        """
        pos_mask = (target_cyx == 1).float()
        pos_elements = (((1 - predict_cyx) ** alpha) * torch.log(predict_cyx + 1e-8)) * pos_mask

        neg_mask = (target_cyx == 0).float()
        neg_elements = (((1 - target_cyx) ** beta) * (predict_cyx ** alpha) \
                            * torch.log(1 - predict_cyx + 1e-8)) * neg_mask

        loss = -torch.sum(pos_elements + neg_elements)

        return loss


    def loss_l1(self, predict, target, is_real_object):
        """
        L1 лосс между предсказаниями и ground truth .
        Некоторые ground truth - ненастоящие bbox
        (т.к. их для всех изображений генерируется по N, а детекций может быть меньше),
        и для объектов с is_real_object=False следует считать лосс как 0.
        """

        predict = predict * is_real_object
        target = target * is_real_object
        regr_loss = nn.functional.smooth_l1_loss(predict, target)

        return regr_loss