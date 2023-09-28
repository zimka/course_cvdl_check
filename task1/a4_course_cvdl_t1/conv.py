import numpy as np
import torch.nn.functional as F
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

        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.random.randn(out_channels)

        self.parameters.append(self.W)
        self.parameters.append(self.b)

        self.stride = 1
        self.padding = kernel_size // 2

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

        self.parameters_grads.append(self.gradW)
        self.parameters_grads.append(self.gradb)

        self.output = None
        self.input = None
        self.grad_input = None

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
    def _pad_zeros(tensor, one_side_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        N, C, H, W = tensor.shape

        target_shape = [N, C, H, W]
        for a in axis:
            target_shape[a] += 2 * one_side_pad

        for dim_in, dim_target in zip(tensor.shape, target_shape):
            assert dim_target >= dim_in

        pad_width = []
        for dim_in, dim_target in zip(tensor.shape, target_shape):
            if (dim_in - dim_target) % 2 == 0:
                pad_width.append((int(abs((dim_in - dim_target) / 2)), int(abs((dim_in - dim_target) / 2))))
            else:
                pad_width.append((int(abs((dim_in - dim_target) / 2)), (int(abs((dim_in - dim_target) / 2)) + 1)))

        return np.pad(tensor, pad_width, 'constant', constant_values=0)

    @staticmethod
    def _cross_correlate(input, kernel):
        """
        Вычисляет "valid" кросс-корреляцию input[B, C_in, H, W]
        и kernel[C_out, C_in, X, Y].
        Метод не проверяется в тестах -- можно релизовать слой и без
        использования этого метода.
        """
        assert kernel.shape[-1] == kernel.shape[-2]
        assert kernel.shape[-1] % 2 == 1
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        N, _, H, W = input.shape
        H = 1 + int((H + 2 * self.padding - self.kernel_size) // self.stride)
        W = 1 + int((W + 2 * self.padding - self.kernel_size) // self.stride)

        xpad = self._pad_zeros(input, self.padding)
        self.input = xpad
        self.output = np.zeros((N, self.out_channels, H, W))
        for xn in range(N):
            for fn in range(self.out_channels):
                for i in range(H):
                    h_start = i * self.stride
                    h_end = i * self.stride + self.kernel_size
                    for j in range(W):
                        w_start = j * self.stride
                        w_end = j * self.stride + self.kernel_size
                        self.output[xn, fn, i, j] = np.sum(xpad[xn, :, h_start: h_end, w_start:w_end] * self.parameters[0][fn]) + \
                                                    self.parameters[1][fn]

        del xpad

        return self.output

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        N, Cout, Hout, Wout = output_grad.shape
        self.grad_input = np.zeros(self.input.shape)
        self.parameters_grads[1] = np.sum(output_grad, axis=(0, 2, 3))
        for xn in range(N):
            for fn in range(self.out_channels):
                for i in range(Hout):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    for j in range(Wout):
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        self.grad_input[xn, :, h_start:h_end, w_start:w_end] += output_grad[xn, fn, i, j] * self.parameters[0][fn]
                        self.parameters_grads[0][fn, :] += self.input[xn, :, h_start:h_end, w_start:w_end] * output_grad[xn, fn, i, j]

        self.grad_input = self.grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return self.grad_input
