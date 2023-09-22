import numpy as np
from .base import BaseLayer


class MaxPoolLayer(BaseLayer):
    """
    Слой, выполняющий 2D Max Pooling, т.е. выбор максимального значения в окне.
    y[B, c, h, w] = Max[i, j] (x[B, c, h+i, w+j])

    У слоя нет обучаемых параметров.
    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.

    В качестве значений padding используется -np.inf, т.е. добавленые pad-значения используются исключительно
     для корректности индексов в любом положении, и никогда не могут реально оказаться максимумом в
     своем пуле.
    Гарантируется, что значения padding, stride и kernel_size будут такие, что
     в input + padding поместится целое число kernel, т.е.:
     (D + 2*padding - kernel_size)  % stride  == 0, где D - размерность H или W.

    Пример корректных значений:
    - kernel_size = 3
    - padding = 1
    - stride = 2
    - D = 7
    Результат:
    (Pool[-1:2], Pool[1:4], Pool[3:6], Pool[5:(7+1)])
    """
    def __init__(self, kernel_size: int, stride: int, padding: int):
        assert(kernel_size % 2 == 1)
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @staticmethod
    def add_pad(tensor, one_size_pad, axis=[2, 3], val=-np.inf):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
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

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[-1] == input.shape[-2]
        assert (input.shape[-1] + 2 * self.padding - self.kernel_size) % self.stride  == 0
        b, c, h, w = input.shape
        self.h = h
        self.w = w
        k = self.kernel_size
        s = self.stride
        p = self.padding
        hn = (h - k + 2 * p) // s + 1
        wn = (w - k + 2 * p) // s + 1
        idxs_b = (np.ones((b, c, hn, wn, k, k)) * np.arange(b)[:, None, None,None,None,None]).astype(int)
        idxs_c =( np.ones((b, c, hn, wn, k, k)) * np.arange(c)[None, :, None,None,None,None]).astype(int)
        idxs_h = (np.ones((b, c, hn, wn, k, k)) * np.arange(hn)[None, None, :, None,None,None] * s + np.arange(k)[None, None, None,None, :,None]).astype(int)
        idxs_w = (np.ones((b, c, hn, wn, k, k)) * np.arange(wn)[None, None, None, :,None,None] * s + np.arange(k)[None, None, None,None,None, :]).astype(int)
        input = self.add_pad(input, p)
        out = input[idxs_b, idxs_c, idxs_h, idxs_w].reshape((b, c, hn, wn, -1))
        self.idxs_max = out.argmax(-1) 
        out = out.max(-1)
        # import torch.nn.functional as F
        # import torch
        # print(np.allclose(F.max_pool2d(torch.tensor(x), k, s, p).numpy(), out), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n\n\n\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n\n\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return out

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        b, c, hn, wn = output_grad.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        h = self.h
        w = self.w
        hiu, wiu = np.unravel_index(self.idxs_max, (k, k))
        idxs_bi = (np.ones((b, c, hn, wn)) * np.arange(b)[:, None, None,None]).astype(int)
        idxs_ci =( np.ones((b, c, hn, wn)) * np.arange(c)[None, :, None,None]).astype(int)
        idxs_hi = (np.ones((b, c, hn, wn)) * np.arange(hn)[None, None, :, None]).astype(int)
        idxs_wi = (np.ones((b, c, hn, wn)) * np.arange(wn)[None, None, None, :]).astype(int)
        g = np.zeros((b, c, hn, wn, k, k))
        g[idxs_bi, idxs_ci, idxs_hi, idxs_wi, hiu, wiu] = 1
        g = g * output_grad[:, :, :, :, None, None]
        h += 2 * p
        w += 2 * p
        idxs_b = (np.ones((b, c, h, w, k, k)) * np.arange(b)[:, None, None,None,None,None]).astype(int)
        idxs_c = (np.ones((b, c, h, w, k, k)) * np.arange(c)[None, :, None,None,None,None]).astype(int)
        idxs_h = (np.ones((b, c, h, w, k, k)) * np.arange(h)[None, None, :, None,None,None] - np.arange(k)[None, None, None,None, :,None]).astype(int)
        mask_h = ((idxs_h % s) == 0).astype(float)
        idxs_h = idxs_h // s + k
        idxs_w = (np.ones((b, c, h, w, k, k)) * np.arange(w)[None, None, None, :,None,None] - np.arange(k)[None, None, None,None,None, :]).astype(int)
        mask_w = ((idxs_w % s) == 0).astype(float)
        idxs_w = idxs_w // s + k
        idxs_kh = (np.ones((b, c, h, w, k, k)) * np.arange(k)[None, None,None,None,:, None]).astype(int)
        idxs_kw = (np.ones((b, c, h, w, k, k)) * np.arange(k)[None, None,None,None,None, :]).astype(int)
        g = self.add_pad(g, k, [2, 3], 0)[idxs_b, idxs_c, idxs_h, idxs_w, idxs_kh, idxs_kw]
        return (g * mask_h * mask_w).sum((-1, -2))[:, :, p:-p, p:-p]

