import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.constrained_min import InteriorPoint
from examples import QPObjective, QPConstraint, LPObjective, LPConstraint, TransportationObjective
from src.utils import plot_cost_breakdown, plot_flow_network, plot_utilization


class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        print("\nRunning QP test...")

        obj = QPObjective()
        ineq_constraints = [QPConstraint(i) for i in range(3)]
        eq_mat = np.array([[1, 1, 1]])
        eq_rhs = np.array([1])
        x0 = np.array([0.1, 0.2, 0.7])

        # Solve
        solver = InteriorPoint(
            obj,
            ineq_constraints,
            eq_mat,
            eq_rhs,
            x0,
        )
        success = solver.minimize()

        # Convert to numpy array
        path = np.array(solver.outer_loop_history)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot path
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            "b-o",
            markersize=4,
            label="Central Path",
        )

        # Plot solution point
        final_point = path[-1]
        ax.scatter(
            final_point[0],
            final_point[1],
            final_point[2],
            c="r",
            s=100,
            marker="*",
            label="Solution",
        )

        # Plot feasible region (triangle)
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            c="g",
            s=50,
            label="Vertices",
        )

        # Plot triangle edges
        for i in range(3):
            edge = np.array([vertices[i], vertices[(i + 1) % 3]])
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], "g--", alpha=0.5)

        ax.set_title("QP Central Path (3D)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.tight_layout()
        plt.savefig("qp_central_path.png", dpi=300)
        plt.close()

        # Plot objective history
        plt.figure(figsize=(8, 5))
        plt.plot(solver.obj_history, "b-o", markersize=4)
        plt.title("QP Objective Value vs Outer Iteration")
        plt.xlabel("Outer Iteration")
        plt.ylabel("Objective Value")
        plt.grid(True)
        plt.savefig("qp_objective.png", dpi=300)
        plt.close()

        final_x = solver.current_x
        x = final_x[0]
        y = final_x[1]
        z = final_x[2]
        print("\nQP Results:")
        print(f"Solver success: {success}")
        print(f"Final point: {final_x}")
        print(f"Objective: {obj.value(final_x):.6f}")
        print("Constraint checks:")
        print(f"  x = {x:.6f} ≥ 0: {x >= 0}")
        print(f"  y = {y:.6f} ≥ 0: {y >= 0}")
        print(f"  z = {z:.6f} ≥ 0: {z >= 0}")
        print(f"  x + y + z == 1: {np.isclose(np.sum(final_x), 1)}")

    def test_lp(self):
        print("\nRunning LP test...")

        # Problem setup
        obj = LPObjective()
        ineq_constraints = [
            LPConstraint([-1, -1], 1),  # -x -y + 1 ≤ 0
            LPConstraint([0, 1], -1),  # y - 1 ≤ 0
            LPConstraint([1, 0], -2),  # x - 2 ≤ 0
            LPConstraint([0, -1], 0),  # -y ≤ 0
        ]
        eq_mat = np.zeros((0, 2))  # No equality constraints
        eq_rhs = np.zeros(0)
        x0 = np.array([0.5, 0.75])

        # Solve
        solver = InteriorPoint(
            obj,
            ineq_constraints,
            eq_mat,
            eq_rhs,
            x0,
        )
        success = solver.minimize()

        # Convert to numpy array
        path = np.array(solver.outer_loop_history)

        # Plot central path (2D)
        plt.figure(figsize=(8, 6))

        # Plot path
        plt.plot(path[:, 0], path[:, 1], "b-o", markersize=4, label="Central Path")

        # Plot solution point
        final_point = path[-1]
        plt.scatter(
            final_point[0], final_point[1], c="r", s=100, marker="*", label="Solution"
        )

        # Plot feasible region (polygon)
        vertices = np.array([[0, 1], [1, 0], [2, 0], [2, 1]])
        plt.plot(vertices[:, 0], vertices[:, 1], "g--", label="Feasible Region")
        plt.fill(vertices[:, 0], vertices[:, 1], "g", alpha=0.1)

        plt.title("LP Central Path (2D)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.savefig("lp_central_path.png", dpi=300)
        plt.close()

        # Plot objective history
        obj_history = [-p for p in solver.obj_history]
        plt.figure(figsize=(8, 5))
        plt.plot(obj_history, "b-o", markersize=4)
        plt.title("LP Objective Value vs Outer Iteration")
        plt.xlabel("Outer Iteration")
        plt.ylabel("Objective Value")
        plt.grid(True)
        plt.savefig("lp_objective.png", dpi=300)
        plt.close()

        final_x = solver.current_x
        x = final_x[0]
        y = final_x[1]
        print("\nLP Results:")
        print(f"Solver success: {success}")
        print(f"Final point: {final_x}")
        print(f"Objective: {-obj.value(final_x):.6f}")
        print("Constraint checks:")
        print(f"  x+y ≥ 1: {np.sum(final_x):.6f} ≥ 1: {np.sum(final_x) >= 1}")
        print(f"  y ≤ 1: {y:.6f} ≤ 1: {y <= 1}")
        print(f"  x ≤ 2: {x:.6f} ≤ 2: {x <= 2}")
        print(f"  y ≥ 0: {y:.6f} ≥ 0: {y >= 0}")

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
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0,    # Factory1 to first 6 outlets (10 demand)
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, # Factory1 to next 6 outlets (20 demand)
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
        self.assertLessEqual(final_x_mat[0, :].sum(), a[0] + 1e-4, "Factory1 capacity")
        self.assertLessEqual(final_x_mat[1, :].sum(), a[1] + 1e-4, "Factory2 capacity")
        for j in range(n_outlets):
            total_ship = final_x_mat[0, j] + final_x_mat[1, j]
            self.assertGreaterEqual(total_ship, b[j] - 1e-4, f"Demand not met at outlet {j}")
        
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
