import numpy as np
from abc import ABC, abstractmethod


class MathFunction(ABC):
    @abstractmethod
    def value(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def hessian(self, x: np.ndarray) -> np.ndarray:
        pass


# For LP constraints
class LPConstraint(MathFunction):
    def __init__(self, coeffs, offset):
        self.coeffs = np.array(coeffs)
        self.offset = offset

    def value(self, x):
        return np.dot(self.coeffs, x) + self.offset

    def gradient(self, x):
        return self.coeffs

    def hessian(self, x):
        return np.zeros((len(x), len(x)))


class TransportationObjective(MathFunction):
    def __init__(self, cost_matrix):
        self.c = cost_matrix.ravel()
        self.n = len(self.c)

    def value(self, x):
        return np.dot(self.c, x)

    def gradient(self, x):
        return self.c

    def hessian(self, x):
        return np.zeros((self.n, self.n))
