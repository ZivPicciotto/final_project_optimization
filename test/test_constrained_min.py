import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.constrained_min import InteriorPoint
from examples import QPObjective, QPConstraint, LPObjective, LPConstraint


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


if __name__ == "__main__":
    unittest.main()
