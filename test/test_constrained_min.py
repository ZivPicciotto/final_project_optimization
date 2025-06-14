import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from src.constrained_min import InteriorPoint
from examples import LPConstraint, TransportationObjective
from src.utils import plot_cost_breakdown, plot_flow_network, plot_utilization
import networkx as nx
from time import time


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

    def test_compare_all_methods(self):
        print("\n" + "="*60)
        print("COMPREHENSIVE METHOD COMPARISON")
        print("="*60)

        # Problem setup - same for all methods
        n_factories = 2
        n_outlets = 12
        capacities = [100, 200]  # Factory capacities
        demands = [10]*6 + [20]*6  # Outlet demands
        costs = np.ones((n_factories, n_outlets))
        costs[1, :] = 2  # Factory 2 has higher shipping costs

        results = {}

        # ============================================
        # Method 1: Interior Point
        # ============================================
        print("\n1. Solving with Interior Point method...")
        ip_start = time()
        ip_solution, ip_cost = self.solve_with_interior_point(
            costs, capacities, demands)
        ip_time = time() - ip_start
        results['Interior Point'] = {
            'solution': ip_solution,
            'cost': ip_cost,
            'time': ip_time,
            'success': True  # Assume success if no exception
        }

        # ============================================
        # Method 2: NetworkX Min-Cost Max Flow
        # ============================================
        print("\n2. Solving with NetworkX Min-Cost Max Flow...")
        mf_start = time()
        mf_solution, mf_cost, _ = solve_with_maxflow_balanced(
            costs, capacities, demands)
        mf_time = time() - mf_start
        results['NetworkX Min-Cost Flow'] = {
            'solution': mf_solution,
            'cost': mf_cost,
            'time': mf_time,
            'success': True
        }

        # ============================================
        # Method 3: SciPy Linear Programming
        # ============================================
        print("\n3. Solving with SciPy Linear Programming...")
        scipy_start = time()
        scipy_solution, scipy_cost, scipy_success = self.solve_with_scipy_linprog(
            costs, capacities, demands)
        scipy_time = time() - scipy_start
        results['SciPy LinProg'] = {
            'solution': scipy_solution,
            'cost': scipy_cost,
            'time': scipy_time,
            'success': scipy_success
        }

        # ============================================
        # Comprehensive Comparison
        # ============================================
        self.compare_all_results(results, costs, capacities, demands)

    def solve_with_interior_point(self, costs, capacities, demands):
        """Solve using your Interior Point implementation"""
        n_factories, n_outlets = costs.shape
        total_vars = n_factories * n_outlets

        obj = TransportationObjective(costs)

        # Inequality constraints (just capacity and non-negativity)
        constraints = []

        # Capacity constraints (≤)
        for i in range(n_factories):
            coeffs = np.zeros(total_vars)
            coeffs[i*n_outlets:(i+1)*n_outlets] = 1
            constraints.append(LPConstraint(coeffs, -capacities[i]))

        # Non-negativity constraints (≥ 0)
        for k in range(total_vars):
            coeffs = np.zeros(total_vars)
            coeffs[k] = -1
            constraints.append(LPConstraint(coeffs, 0))

        # Equality constraints for demand satisfaction
        eq_constraints = []
        for j in range(n_outlets):
            coeffs = np.zeros(total_vars)
            coeffs[j] = 1  # From factory 1
            coeffs[j + n_outlets] = 1  # From factory 2
            eq_constraints.append(coeffs)

        eq_mat = np.array(eq_constraints)
        eq_rhs = np.array(demands)

        # Initial feasible point
        x0 = np.array([*[5.0]*6, *[10.0]*6, *[5.0]*6, *[10.0]*6])

        # Solve
        solver = InteriorPoint(obj, constraints, eq_mat, eq_rhs, x0)
        solver.minimize()

        solution = solver.current_x.reshape((n_factories, n_outlets))
        cost = obj.value(solver.current_x)

        return solution, cost

    def solve_with_scipy_linprog(self, costs, capacities, demands):
        """Solve using SciPy's linear programming solver"""
        n_factories, n_outlets = costs.shape
        total_vars = n_factories * n_outlets

        # Objective: minimize cost (linprog minimizes c^T x)
        c = costs.flatten()

        # Inequality constraints: A_ub @ x <= b_ub
        A_ub = []
        b_ub = []

        # Capacity constraints: sum of shipments from each factory <= capacity
        for i in range(n_factories):
            constraint_row = np.zeros(total_vars)
            constraint_row[i*n_outlets:(i+1)*n_outlets] = 1
            A_ub.append(constraint_row)
            b_ub.append(capacities[i])

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Equality constraints: A_eq @ x = b_eq
        # Demand constraints: sum of shipments to each outlet = demand
        A_eq = []
        b_eq = []

        for j in range(n_outlets):
            constraint_row = np.zeros(total_vars)
            constraint_row[j] = 1  # From factory 1
            constraint_row[j + n_outlets] = 1  # From factory 2
            A_eq.append(constraint_row)
            b_eq.append(demands[j])

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        # Bounds: all variables >= 0 (default in linprog)
        bounds = [(0, None) for _ in range(total_vars)]

        # Solve using different methods for robustness
        methods = ['highs', 'highs-ds', 'highs-ipm']

        for method in methods:
            try:
                print(f"   Trying SciPy method: {method}")
                result = linprog(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds,
                    method=method,
                    options={'disp': False, 'presolve': True}
                )

                if result.success:
                    print(f"   ✓ Success with method: {method}")
                    solution = result.x.reshape((n_factories, n_outlets))
                    cost = result.fun

                    # Verify the solution
                    self.verify_scipy_solution(
                        solution, costs, capacities, demands)

                    return solution, cost, True
                else:
                    print(
                        f"   ✗ Failed with method: {method} - {result.message}")

            except Exception as e:
                print(f"   ✗ Exception with method: {method} - {str(e)}")
                continue

        # If all methods failed
        print("   ✗ All SciPy methods failed")
        return np.zeros((n_factories, n_outlets)), np.inf, False

    def verify_scipy_solution(self, solution, costs, capacities, demands):
        """Verify that SciPy solution satisfies all constraints"""
        n_factories, n_outlets = solution.shape

        print(f"   SciPy Solution Verification:")

        # Check capacity constraints
        for i in range(n_factories):
            used = solution[i, :].sum()
            print(f"     Factory {i+1}: {used:.2f}/{capacities[i]} capacity")
            assert used <= capacities[i] + \
                1e-6, f"Capacity exceeded for factory {i+1}"

        # Check demand constraints
        for j in range(n_outlets):
            supplied = solution[:, j].sum()
            print(f"     Outlet {j+1}: {supplied:.2f}/{demands[j]} demand")
            assert abs(
                supplied - demands[j]) <= 1e-6, f"Demand not met for outlet {j+1}"

        # Check non-negativity
        assert np.all(solution >= -1e-6), "Negative shipments found"

        # Calculate and verify cost
        total_cost = np.sum(solution * costs)
        print(f"     Total cost: {total_cost:.4f}")

    def compare_all_results(self, results, costs, capacities, demands):
        """Compare results from all three methods"""
        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)

        # Summary table
        print(f"{'Method':<25} {'Cost':<12} {'Time(s)':<10} {'Success':<8}")
        print("-" * 55)

        for method_name, result in results.items():
            print(
                f"{method_name:<25} {result['cost']:<12.4f} {result['time']:<10.4f} {result['success']:<8}")

        # Detailed comparison
        successful_methods = {k: v for k, v in results.items() if v['success']}

        if len(successful_methods) >= 2:
            print(f"\nDETAILED COMPARISON (Successful methods only):")

            methods = list(successful_methods.keys())
            costs_list = [successful_methods[m]['cost'] for m in methods]

            # Cost differences
            print(f"\nCost Analysis:")
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i < j:
                        diff = abs(costs_list[i] - costs_list[j])
                        print(
                            f"  {method1} vs {method2}: difference = {diff:.6f}")

            # Solution differences
            print(f"\nSolution Analysis:")
            base_method = methods[0]
            base_solution = successful_methods[base_method]['solution']

            for method in methods[1:]:
                other_solution = successful_methods[method]['solution']
                max_diff = np.max(np.abs(base_solution - other_solution))
                rms_diff = np.sqrt(
                    np.mean((base_solution - other_solution)**2))
                print(f"  {base_method} vs {method}:")
                print(f"    Max difference: {max_diff:.6f}")
                print(f"    RMS difference: {rms_diff:.6f}")

        # Create comprehensive comparison plot
        self.plot_comprehensive_comparison(results, costs, capacities, demands)

    def plot_comprehensive_comparison(self, results, costs, capacities, demands):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Method Comparison',
                     fontsize=16, fontweight='bold')

        successful_results = {k: v for k, v in results.items() if v['success']}
        methods = list(successful_results.keys())
        colors = ['blue', 'green', 'red', 'orange', 'purple']

        # 1. Time Comparison
        ax = axes[0]
        times_list = [successful_results[m]['time'] for m in methods]
        bars = ax.bar(methods, times_list, color=colors[:len(methods)])
        ax.set_title('Computation Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')  # Log scale for time differences

        # Add value labels
        for bar, time_val in zip(bars, times_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.4f}', ha='center', va='bottom')

        # 2. Factory Utilization Comparison
        ax = axes[1]
        width = 0.8 / len(methods)
        x = np.arange(2)  # Two factories

        for i, method in enumerate(methods):
            solution = successful_results[method]['solution']
            utilization = [solution[j, :].sum() / capacities[j]
                           * 100 for j in range(2)]
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, utilization, width,
                   label=method, color=colors[i])

        ax.set_title('Factory Capacity Utilization')
        ax.set_ylabel('Utilization (%)')
        ax.set_xlabel('Factory')
        ax.set_xticks(x)
        ax.set_xticklabels(['Factory 1', 'Factory 2'])
        ax.legend()

        # 3. Method Characteristics Summary
        ax = axes[2]
        ax.axis('off')

        summary_text = "METHOD CHARACTERISTICS\n" + "="*25 + "\n\n"

        for method, result in successful_results.items():
            summary_text += f"{method}:\n"
            summary_text += f"  Cost: ${result['cost']:.2f}\n"
            summary_text += f"  Time: {result['time']:.4f}s\n"
            summary_text += f"  Success: {result['success']}\n\n"

        # Add algorithm characteristics
        summary_text += "\nALGORITHM TYPES:\n" + "-"*15 + "\n"
        summary_text += "Interior Point: Barrier Method\n"
        summary_text += "NetworkX: Min-Cost Max Flow\n"
        summary_text += "SciPy: Simplex/Interior Point\n"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig('comprehensive_method_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(
            "\nSaved comprehensive comparison plot to 'comprehensive_method_comparison.png'")

    def test_compare_methods(self):
        """Legacy test method - now calls the comprehensive comparison"""
        self.test_compare_all_methods()


def solve_with_maxflow_balanced(costs, capacities, demands):
    """Max-flow formulation that correctly calculates costs"""
    G = nx.DiGraph()

    # Add nodes
    G.add_node('source', demand=-sum(demands))
    G.add_node('sink', demand=sum(demands))

    # Add factory nodes
    for i, cap in enumerate(capacities):
        G.add_node(f'F{i+1}')
        # Connect source to factories with available capacity
        G.add_edge('source', f'F{i+1}', capacity=cap, weight=0)

    # Add retail nodes
    for j, dem in enumerate(demands):
        G.add_node(f'R{j+1}')
        # Connect retailers to sink with demand as capacity
        G.add_edge(f'R{j+1}', 'sink', capacity=dem, weight=0)

    # Add factory-retail edges with costs
    for i in range(len(capacities)):
        for j in range(len(demands)):
            # Set capacity to min of factory capacity and outlet demand
            edge_capacity = min(capacities[i], demands[j])
            G.add_edge(f'F{i+1}', f'R{j+1}',
                       capacity=edge_capacity,
                       # This is crucial for cost calculation
                       weight=costs[i, j])

    # Solve min cost flow
    flow_dict = nx.min_cost_flow(G)

    # Extract solution and compute cost
    solution = np.zeros_like(costs)
    total_cost = 0

    for i in range(len(capacities)):
        factory_node = f'F{i+1}'
        for j in range(len(demands)):
            retail_node = f'R{j+1}'
            flow = flow_dict[factory_node].get(retail_node, 0)
            solution[i, j] = flow
            total_cost += flow * costs[i, j]

    # Verification
    print("NetworkX Min-Cost Flow Verification:")
    print("Total shipped from F1:", solution[0].sum())
    print("Total shipped from F2:", solution[1].sum())
    print("Total received by outlets:", solution.sum(axis=0))
    print("Actual demands:", demands)
    print("Total calculated cost:", total_cost)

    return solution, total_cost, 0


if __name__ == "__main__":
    unittest.main()
