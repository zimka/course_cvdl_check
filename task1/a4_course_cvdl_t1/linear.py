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

        stdv = 1. / np.sqrt(in_features)
        self.W = np.random.randn(out_features, in_features)
        self.b = np.random.uniform(-stdv, stdv, size=out_features)

        self.parameters.append(self.W)
        self.parameters.append(self.b)
        self.parameters_grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

        self.input = None
        self.grad_input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = input @ self.parameters[0].T + self.parameters[1]

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.grad_input = output_grad @ self.parameters[0]

        if len(self.input.shape) <= 2:
            self.gradW = np.transpose(self.input.T @ output_grad)
            self.gradb = np.sum(output_grad, axis=0)
        else:
            self.gradW = np.expand_dims(self.input, -2) * np.expand_dims(output_grad, -1)
            self.gradW = np.sum(np.expand_dims(self.input, -2) * np.expand_dims(output_grad, -1), axis=tuple(range(len(self.gradW.shape) - 2)))
            self.gradb = np.sum(output_grad, axis=tuple(range(len(output_grad.shape) - 1)))

        self.parameters_grads[0] = self.gradW
        self.parameters_grads[1] = self.gradb

        return self.grad_input
