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
        self.ispos = (input > 0)
        return input * self.ispos
        
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Принимат dL/dy, возвращает dL/dx.
        """
        return self.ispos * output_grad

