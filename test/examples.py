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


# ===== QP Example =====
class QPObjective(MathFunction):
    def value(self, x):
        return x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2

    def gradient(self, x):
        return np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])

    def hessian(self, x):
        return np.diag([2, 2, 2])


# ===== LP Example =====
class LPObjective(MathFunction):
    def value(self, x):
        return -np.sum(x)  # Maximize x+y → minimize -(x+y)

    def gradient(self, x):
        return np.array([-1, -1])

    def hessian(self, x):
        return np.zeros((2, 2))


class QPConstraint(MathFunction):
    def __init__(self, index: int):
        self.index = index

    def value(self, x):
        return -x[self.index]  # -x_i ≤ 0 → x_i ≥ 0

    def gradient(self, x):
        grad = np.zeros_like(x)
        grad[self.index] = -1
        return grad

    def hessian(self, x):
        return np.zeros((len(x), len(x)))


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