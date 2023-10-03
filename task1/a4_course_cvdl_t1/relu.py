import numpy as np
from .base import BaseLayer


class ReluLayer(BaseLayer):
    """
    Слой, выполняющий Relu активацию y = max(x, 0).
    Не имеет параметров.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Принимает x, возвращает y(x)
        """
        self.is_positive = (input > 0).astype(float)
        return input * self.is_positive

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Принимат dL/dy, возвращает dL/dx.
        """
        return output_grad * self.is_positive

