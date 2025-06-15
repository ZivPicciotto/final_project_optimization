import numpy as np
from test.test_utils import MathFunction
from .utils import BarrierFunction

# --------------------------- GLOBAL CONSTANTS --------------------------------
MU = 10.0
T0 = 1.0
INNER_EPSILON = 1e-8
OUTER_EPSILON = 1e-10
OUTER_MAX_ITER = 1000
RHO = 0.5
# -----------------------------------------------------------------------------


class InteriorPoint:
    def __init__(
        self,
        func: MathFunction,
        ineq_constraints: list[MathFunction],
        eq_constraints_mat: np.ndarray,
        eq_constraints_rhs: np.ndarray,
        x0: np.ndarray,
    ):
        self.f = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.t = T0
        self.current_x = x0
        self.current_fx = self.f.value(self.current_x)
        self.outer_loop_history = [self.current_x.copy()]
        self.obj_history = [self.current_fx.copy()]

    def backtracking(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        f_x: float,
        grad_f_x: np.ndarray,
        get_func_value,
        c1: float = 1e-2,
        rho: float = RHO,
        max_steps: int = 50,
    ) -> float:
        alpha = 1.0
        slope = np.dot(grad_f_x, direction)
        for _ in range(max_steps):
            if get_func_value(x + alpha * direction) <= f_x + c1 * alpha * slope:
                return alpha
            alpha *= rho
        return alpha

    def newton_unconstrained(self, max_iter: int = 1000) -> bool:
        barrier = BarrierFunction(self.f, self.ineq_constraints, self.t)
        for _ in range(max_iter):
            gradient = barrier.gradient(self.current_x)
            hessian = barrier.hessian(self.current_x)
            A = self.eq_constraints_mat
            n = self.current_x.size
            m = A.shape[0] if A.size else 0
            if m:
                KKT = np.block([[hessian, A.T], [A, np.zeros((m, m))]])
                rhs = np.concatenate([-gradient, np.zeros(m)])
            else:
                KKT = hessian
                rhs = -gradient
            try:
                step = np.linalg.solve(KKT, rhs)[:n]
            except np.linalg.LinAlgError:
                return False

            if 0.5 * step @ hessian @ step < INNER_EPSILON:
                return True

            fx_barrier = barrier.value(self.current_x)
            alpha = self.backtracking(
                self.current_x,
                step,
                fx_barrier,
                gradient,
                get_func_value=barrier.value,
            )

            self.current_x += alpha * step
        return False

    def minimize(self):
        m = len(self.ineq_constraints)
        for _ in range(OUTER_MAX_ITER):
            if not self.newton_unconstrained():
                break
            self.outer_loop_history.append(self.current_x.copy())
            self.obj_history.append(self.f.value(self.current_x))
            if m / self.t < OUTER_EPSILON:
                return True
            self.t *= MU
        return False
