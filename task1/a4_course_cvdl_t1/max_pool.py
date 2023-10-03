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
        arr_pad = np.array([(0, 0)] * tensor.ndim)
        arr_pad[axis] = (one_size_pad, one_size_pad)
        return np.pad(tensor, arr_pad, mode='constant', constant_values=-np.inf)

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[-1] == input.shape[-2]
        assert (input.shape[-1] + 2 * self.padding - self.kernel_size) % self.stride  == 0

        self.input_shape = input.shape
        input_pad = self._pad_neg_inf(input, self.padding)

        res_shape = [
            input.shape[0],
            input.shape[1], 
            (input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1,
            (input.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        ]
        res = np.zeros(res_shape)

        self.max_list = []
        
        for b in range(res_shape[0]):
            for c in range(res_shape[1]):
                for h_ in range(res_shape[2]):
                    h = h_ * self.stride
                    for w_ in range(res_shape[3]):
                        w = w_ * self.stride
                        arg_max = np.argmax(input_pad[b, c, h:(h + self.kernel_size), w:(w + self.kernel_size)])
                        ind_max = np.unravel_index(arg_max, (self.kernel_size, self.kernel_size))
                        max_val = input_pad[b, c, h + ind_max[0], w + ind_max[1]]
                        res[b, c, h_, w_] = max_val
                        self.max_list.append(
                            (b, c, h + arg_max // self.kernel_size - self.padding, w + arg_max % self.kernel_size - self.padding)
                        )
        
        return res


    def backward(self, output_grad: np.ndarray)->np.ndarray:
        res = np.zeros(self.input_shape)
        i = 0
        for b in range(output_grad.shape[0]):
            for c in range(output_grad.shape[1]):
                for h in range(output_grad.shape[2]):
                    for w in range(output_grad.shape[3]):
                        res[self.max_list[i]] += output_grad[b][c][h][w] 
                        i += 1
        
        return res

