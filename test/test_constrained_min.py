import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.constrained_min import InteriorPoint
from examples import LPConstraint, TransportationObjective
from src.utils import plot_cost_breakdown, plot_flow_network, plot_utilization


class TestConstrainedMin(unittest.TestCase):
    def test_transportation(self):
        print("\nRunning transportation problem test...")

        # Problem parameters
        n_factories = 2
        n_outlets = 12
        a = [100, 200]  # Factory capacities
        b = [10]*6 + [20]*6  # Outlet demands
        costs = np.ones((n_factories, n_outlets))
        costs[1, :] = 2  # Factory 2 has higher shipping costs

        total_vars = n_factories * n_outlets

        # Objective function
        obj = TransportationObjective(costs)

        # Inequality constraints
        constraints = []

        # Capacity constraints (2)
        for i in range(n_factories):
            coeffs = np.zeros(total_vars)
            coeffs[i*n_outlets:(i+1)*n_outlets] = 1
            constraints.append(LPConstraint(coeffs, -a[i]))  # Σx_ij ≤ a_i

        # Demand constraints (12)
        for j in range(n_outlets):
            coeffs = np.zeros(total_vars)
            coeffs[j] = -1  # Factory 1
            coeffs[j + n_outlets] = -1  # Factory 2
            constraints.append(LPConstraint(coeffs, b[j]))  # x1j + x2j ≥ b_j

        # Non-negativity constraints (24)
        for k in range(total_vars):
            coeffs = np.zeros(total_vars)
            coeffs[k] = -1
            constraints.append(LPConstraint(coeffs, 0))  # x_k ≥ 0

        # No equality constraints
        eq_mat = np.zeros((0, total_vars))
        eq_rhs = np.zeros(0)

        # Better initial feasible point (strictly satisfies all constraints)
        x0 = np.array([
            # Factory1 to first 6 outlets (10 demand)
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            # Factory1 to next 6 outlets (20 demand)
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0,    # Factory2 to first 6 outlets
            12.0, 12.0, 12.0, 12.0, 12.0, 12.0  # Factory2 to next 6 outlets
        ])

        # Verify initial feasibility more carefully
        x0_mat = x0.reshape((n_factories, n_outlets))

        # Check factory capacities
        cap1 = x0_mat[0, :].sum()  # Should be < 100
        cap2 = x0_mat[1, :].sum()  # Should be < 200
        print(f"Initial Factory 1 usage: {cap1}/{a[0]}")
        print(f"Initial Factory 2 usage: {cap2}/{a[1]}")
        assert cap1 < a[0], "Factory1 capacity exceeded in initial point"
        assert cap2 < a[1], "Factory2 capacity exceeded in initial point"

        # Check outlet demands
        for j in range(n_outlets):
            demand = x0_mat[0, j] + x0_mat[1, j]
            required = b[j]
            print(f"Outlet {j}: {demand} (needs {required})")
            assert demand > required, f"Demand not met at outlet {j} in initial point"

        # Check non-negativity
        assert np.all(x0 > 0), "Initial point has non-positive values"

        # Solve with adjusted parameters
        solver = InteriorPoint(
            obj,
            constraints,
            eq_mat,
            eq_rhs,
            x0,
        )

        # Adjust solver parameters for better stability
        global MU, T0, INNER_EPSILON, OUTER_EPSILON
        MU = 5.0  # Smaller multiplicative factor
        T0 = 0.1  # Smaller initial t value
        INNER_EPSILON = 1e-6
        OUTER_EPSILON = 1e-8

        success = solver.minimize()

        # Restore global parameters
        MU = 10.0
        T0 = 1.0
        INNER_EPSILON = 1e-8
        OUTER_EPSILON = 1e-10

        # Results
        final_x = solver.current_x
        final_x_mat = final_x.reshape((n_factories, n_outlets))
        total_cost = obj.value(final_x)

        print("\nTransportation Results:")
        print(f"Solver success: {success}")
        print(f"Final cost: {total_cost:.4f}")
        print(f"Factory 1 usage: {final_x_mat[0, :].sum():.2f}/{a[0]}")
        print(f"Factory 2 usage: {final_x_mat[1, :].sum():.2f}/{a[1]}")

        # Verify constraints with some tolerance
        self.assertTrue(np.all(final_x >= -1e-4), "Non-negativity violated")
        self.assertLessEqual(
            final_x_mat[0, :].sum(), a[0] + 1e-4, "Factory1 capacity")
        self.assertLessEqual(
            final_x_mat[1, :].sum(), a[1] + 1e-4, "Factory2 capacity")
        for j in range(n_outlets):
            total_ship = final_x_mat[0, j] + final_x_mat[1, j]
            self.assertGreaterEqual(
                total_ship, b[j] - 1e-4, f"Demand not met at outlet {j}")

        # Verify optimal cost (should be 260)
        self.assertAlmostEqual(total_cost, 260, delta=1e-3)

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(solver.obj_history, 'b-o', markersize=4)
        plt.title('Transportation Problem: Objective Value vs Outer Iteration')
        plt.xlabel('Outer Iteration')
        plt.ylabel('Total Shipping Cost')
        plt.grid(True)
        plt.savefig('transport_objective.png', dpi=300)
        plt.close()

        plot_flow_network(solver.current_x, costs, a, b)
        plot_utilization(solver.current_x, a)
        plot_cost_breakdown(solver.current_x, costs)


if __name__ == "__main__":
    unittest.main()
