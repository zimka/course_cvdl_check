import numpy as np
from .base import BaseLayer


class ReluLayer(BaseLayer):
    """
    Слой, выполняющий Relu активацию y = max(x, 0).
    Не имеет параметров.
    """
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Принимает x, возвращает y(x)
        """
        self.input = input
        return np.maximum(input, np.zeros_like(input))


    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Принимат dL/dy, возвращает dL/dx.
        """
        gradInput = np.multiply(output_grad, self.input > 0)
        return gradInput

