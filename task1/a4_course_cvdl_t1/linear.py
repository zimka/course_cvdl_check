import numpy as np
from .base import BaseLayer


class LinearLayer(BaseLayer):
    """
    Слой, выполняющий линейное преобразование y = x @ W.T + b.
    Параметры:
        parameters[0]: W;
        parameters[1]: b;
    Линейное преобразование выполняется для последней оси тензоров, т.е.
     y[B, ..., out_features] = LinearLayer(in_features, out_feautres)(x[B, ..., in_features].)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = np.random.randn(out_features, in_features)
        self.b = np.zeros(out_features)
        self.parameters.append(self.W)
        self.parameters.append(self.b)

    def forward(self, input: np.ndarray) -> np.ndarray:
        y = input @ self.parameters[0].T + self.parameters[1]
        self.input = input
        return y

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        dinput = output_grad @ self.parameters[0]
        dW = np.expand_dims(output_grad, -1) * np.expand_dims(self.input, -2)
        db = output_grad
        dW = np.apply_over_axes(np.sum, dW, range(dW.ndim - 2)).reshape(dW.shape[-2:])
        db = np.apply_over_axes(np.sum, db, range(db.ndim - 1)).reshape(db.shape[-1:])
        self.parameters_grads = [dW, db]
        return dinput
