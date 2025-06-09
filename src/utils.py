import numpy as np
from test.examples import MathFunction


class BarrierFunction(MathFunction):
    def __init__(self, func, ineq_constraints, t):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.t = t

    def value(self, x):
        penalty = sum(-np.log(-c.value(x)) for c in self.ineq_constraints)
        return self.t * self.func.value(x) + penalty

    def gradient(self, x):
        grad = self.func.gradient(x)
        penalty_grad = sum(
            (1.0 / (-c.value(x))) * c.gradient(x) for c in self.ineq_constraints
        )
        return self.t * grad + penalty_grad

    def hessian(self, x):
        hessian = self.func.hessian(x)
        penalty_hess = np.zeros((x.size, x.size))

        for c in self.ineq_constraints:
            constraint_val = c.value(x)
            grad = c.gradient(x)
            outer_product = np.outer(grad, grad)
            penalty_hess += (
                outer_product / (constraint_val**2) - c.hessian(x) / constraint_val
            )

        return self.t * hessian + penalty_hess
