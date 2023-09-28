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
    def _pad_neg_inf(tensor, one_size_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        N, C, H, W = tensor.shape

        target_shape = [N, C, H, W]
        for a in axis:
            target_shape[a] += 2 * one_size_pad

        for dim_in, dim_target in zip(tensor.shape, target_shape):
            assert dim_target >= dim_in

        pad_width = []
        for dim_in, dim_target in zip(tensor.shape, target_shape):
            if (dim_in - dim_target) % 2 == 0:
                pad_width.append((int(abs((dim_in - dim_target) / 2)), int(abs((dim_in - dim_target) / 2))))
            else:
                pad_width.append((int(abs((dim_in - dim_target) / 2)), (int(abs((dim_in - dim_target) / 2)) + 1)))

        return np.pad(tensor, pad_width, 'constant', constant_values=-np.inf)

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[-1] == input.shape[-2]
        assert (input.shape[-1] + 2 * self.padding - self.kernel_size) % self.stride  == 0
        N, C, H, W = input.shape
        H = 1 + int((H + 2 * self.padding - self.kernel_size) // self.stride)
        W = 1 + int((W + 2 * self.padding - self.kernel_size) // self.stride)

        xpad = self._pad_neg_inf(input, self.padding)
        self.input = xpad
        self.output = np.zeros((N, C, H, W))
        self.mask = np.zeros((N, C, H, W))

        for xn in range(N):
            for fn in range(C):
                for i in range(H):
                    h_start = i * self.stride
                    h_end = i * self.stride + self.kernel_size
                    for j in range(W):
                        w_start = j * self.stride
                        w_end = j * self.stride + self.kernel_size
                        window = xpad[xn, fn, h_start: h_end, w_start:w_end]
                        max_ids = np.unravel_index(window.argmax(), window.shape)
                        self.output[xn, fn, i, j] = np.max(xpad[xn, fn, h_start: h_end, w_start:w_end])

        del xpad



        return self.output


    def backward(self, output_grad: np.ndarray)->np.ndarray:
        N, C, H, W = self.input.shape

        self.input_grad = np.zeros(self.input.shape)
        print(self.output.shape)
        print(output_grad.shape)

        for xn in range(N):
            for fn in range(C):
                for i in range(H):
                    h_start = i * self.stride
                    h_end = i * self.stride + self.kernel_size
                    for j in range(W):
                        w_start = j * self.stride
                        w_end = j * self.stride + self.kernel_size
                        window = self.input[xn, fn, h_start: h_end, w_start:w_end]
                        max_ids = np.unravel_index(window.argmax(), window.shape)

                        self.input_grad[xn, fn, h_start: h_end, w_start:w_end][max_ids] = output_grad[xn, fn, i, j]

        self.input_grad = self.input_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return self.input_grad