import numpy as np
from .base import BaseLayer


class ConvLayer(BaseLayer):
    """
    Слой, выполняющий 2D кросс-корреляцию (с указанными ниже ограничениями).
    y[B, k, h, w] = Sum[i, j, c] (x[B, c, h+i, w+j] * w[k, c, i, j]) + b[k]

    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.
    В тестах input также всегда квадратный, и H==W всегда нечетные.
    К свертке входа с ядром всегда надо прибавлять тренируемый параметр-вектор (bias).
    Ядро свертки не разреженное (~ dilation=1).
    Значение stride всегда 1.
    Всегда используется padding='same', т.е. входной тензор необходимо дополнять нулями, и
     результат .forward(input) должен всегда иметь [H, W] размерность, равную
     размерности input.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        assert(in_channels > 0)
        assert(out_channels > 0)
        assert(kernel_size % 2 == 1)
        super().__init__()
        self.parameters = [np.zeros((out_channels, in_channels, kernel_size, kernel_size)), np.zeros(out_channels)]
        self.k = kernel_size
      

    @property
    def kernel_size(self):
        return self.parameters[0].shape[-1]

    @property
    def out_channels(self):
        return self.parameters[0].shape[0]

    @property
    def in_channels(self):
        return self.parameters[0].shape[1]

    @staticmethod
    def _pad_zeros(tensor, one_size_pad, axis=[3, 4]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        val = 0
        shape = list(tensor.shape)
        idxs = []
        for i, s in enumerate(shape):
          oneshape = [1 for t in shape]
          oneshape[i] = s
          idx = np.arange(s).reshape(tuple(oneshape))
          if i in axis:
            idx += one_size_pad
          idxs.append((np.ones_like(tensor) * idx).astype(int))
        shape[axis[0]] += one_size_pad * 2
        shape[axis[1]] += one_size_pad * 2
        out = np.ones(tuple(shape)) * val

        out[tuple(idxs)] = tensor
        return out
  

    def _cross_correlate(self, tensor, kernel, s, p):
        """
        Вычисляет "valid" кросс-корреляцию input[B, C_in, H, W]
        и kernel[C_out, C_in, X, Y].
        Метод не проверяется в тестах -- можно релизовать слой и без
        использования этого метода.
        """
        assert kernel.shape[-1] == kernel.shape[-2]
        assert kernel.shape[-1] % 2 == 1
        b, cin, h, w, = tensor.shape
        cout, cink, k, __ = kernel.shape
        assert cin == cink 
        c = cin

        hn = (h - k + 2 * p) // s + 1
        wn = (w - k + 2 * p) // s + 1
        idxs_b = (np.ones((b, c, hn, wn, k, k)) * np.arange(b)[:, None, None,None,None,None]).astype(int)
        idxs_c = (np.ones((b, c, hn, wn, k, k)) * np.arange(c)[None, :, None,None,None,None]).astype(int)
        idxs_h = (np.ones((b, c, hn, wn, k, k)) * np.arange(hn)[None, None, :, None,None,None] * s  + np.arange(k)[None, None, None,None, :,None]).astype(int)
        idxs_w = (np.ones((b, c, hn, wn, k, k)) * np.arange(wn)[None, None, None, :,None,None] * s + np.arange(k)[None, None, None,None,None, :]).astype(int)
        
        tensor = self._pad_zeros(tensor, p, [2, 3])[idxs_b, idxs_c, idxs_h, idxs_w]
        tensor = (tensor[:, None] * kernel[None, :, :, None, None]).sum((2, -1, -2))
        return tensor

    def forward(self, input: np.ndarray) -> np.ndarray:
        p = (self.k - 1) // 2
        self.input = input
        out = self._cross_correlate(input, self.parameters[0], 1, p)
        out = out + self.parameters[1][None, :, None, None]
        return out

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        dI = self._cross_correlate(output_grad, self.parameters[0][:, :, ::-1, ::-1].transpose(1, 0, 2, 3), 1, (self.k - 1 ) // 2)
        
        db = output_grad.sum((0, 2, 3))
        dkernel = self._cross_correlate(output_grad.transpose(1, 0, 2, 3), self.input.transpose(1, 0, 2, 3), 1, (self.k - 1) // 2)
        self.parameters_grads = [dkernel[:, :, ::-1, ::-1], db]
        return dI

