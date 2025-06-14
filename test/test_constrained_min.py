import unittest
import numpy as np
import matplotlib.pyplot as plt
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

    def test_compare_methods(self):
        print("\nRunning method comparison test with matched formulations...")
        
        # Problem setup - same for both methods
        n_factories = 2
        n_outlets = 12
        capacities = [100, 200]  # Factory capacities
        demands = [10]*6 + [20]*6  # Outlet demands
        costs = np.ones((n_factories, n_outlets))
        costs[1, :] = 2  # Factory 2 has higher shipping costs

        # ============================================
        # Revised Interior Point Formulation (matches max-flow)
        # ============================================
        print("\nSolving with revised Interior Point method...")
        
        total_vars = n_factories * n_outlets
        obj = TransportationObjective(costs)
        
        # Inequality constraints (just capacity and non-negativity)
        constraints = []
        
        # Capacity constraints (≤)
        for i in range(n_factories):
            coeffs = np.zeros(total_vars)
            coeffs[i*n_outlets:(i+1)*n_outlets] = 1
            constraints.append(LPConstraint(coeffs, -capacities[i]))  # Σx_ij ≤ capacity_i
        
        # Non-negativity constraints (≥ 0)
        for k in range(total_vars):
            coeffs = np.zeros(total_vars)
            coeffs[k] = -1
            constraints.append(LPConstraint(coeffs, 0))  # x_k ≥ 0

        # Equality constraints for demand satisfaction (=)
        eq_constraints = []
        for j in range(n_outlets):
            coeffs = np.zeros(total_vars)
            coeffs[j] = 1  # From factory 1
            coeffs[j + n_outlets] = 1  # From factory 2
            eq_constraints.append(coeffs)
        
        eq_mat = np.array(eq_constraints)
        eq_rhs = np.array(demands)

        # Initial feasible point (must satisfy Ax = b)
        x0 = np.array([
            # Factory 1 ships exactly half of each demand
            *[5.0]*6, *[10.0]*6,
            # Factory 2 ships the other half
            *[5.0]*6, *[10.0]*6
        ])

        # Verify initial point satisfies equality constraints
        assert np.allclose(eq_mat @ x0, eq_rhs), "Initial point doesn't satisfy demand constraints"
        
        # Solve
        ip_start = time()
        solver = InteriorPoint(obj, constraints, eq_mat, eq_rhs, x0)
        success = solver.minimize()
        ip_time = time() - ip_start
        
        ip_solution = solver.current_x.reshape((n_factories, n_outlets))
        ip_cost = obj.value(solver.current_x)

        # ============================================
        # Max-Flow Formulation
        # ============================================
        print("\nSolving with Max-Flow method...")
        mf_start = time()
        mf_solution, mf_cost, _ = solve_with_maxflow_balanced(costs, capacities, demands)
        mf_time = time() - mf_start

        # ============================================
        # Comparison
        # ============================================
        print("\nComparison Results:")
        print(f"{'Metric':<20} {'Interior Point':<15} {'Max-Flow':<15}")
        print(f"{'Total Cost':<20} {ip_cost:<15.2f} {mf_cost:<15.2f}")
        print(f"{'Solve Time (s)':<20} {ip_time:<15.4f} {mf_time:<15.4f}")
        
        # Solution difference
        diff = ip_solution - mf_solution
        max_diff = np.max(np.abs(diff))
        print(f"\nMaximum solution difference: {max_diff:.4f}")

        # Plot comparison
        self.plot_comparison(ip_solution, mf_solution, ip_cost, mf_cost, ip_time, mf_time)

    def plot_comparison(self, ip_solution, mf_solution, ip_cost, mf_cost, ip_time, mf_time):
        """Plot comparison between the two methods"""
        plt.figure(figsize=(15, 10))
        
        # Cost and Time comparison
        plt.subplot(2, 2, 1)
        bars = plt.bar(['Interior Point', 'Max-Flow'], [ip_cost, mf_cost], color=['blue', 'green'])
        plt.title('Total Cost Comparison')
        plt.ylabel('Cost ($)')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.subplot(2, 2, 2)
        bars = plt.bar(['Interior Point', 'Max-Flow'], [ip_time, mf_time], color=['red', 'orange'])
        plt.title('Computation Time Comparison')
        plt.ylabel('Time (seconds)')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        # Solution difference heatmap
        plt.subplot(2, 2, 3)
        diff = ip_solution - mf_solution
        plt.imshow(diff, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Difference (IP - MF)')
        plt.title("Solution Difference Heatmap")
        plt.xlabel("Retail Outlet")
        plt.ylabel("Factory")
        plt.xticks(np.arange(12), np.arange(12)+1)
        plt.yticks([0, 1], ['Factory 1', 'Factory 2'])
        
        # Solution values side by side
        plt.subplot(2, 2, 4)
        plt.plot(ip_solution.flatten(), 'bo', label='Interior Point', alpha=0.6)
        plt.plot(mf_solution.flatten(), 'r+', label='Max-Flow', markersize=10)
        plt.title("Solution Values Comparison")
        plt.xlabel("Variable Index")
        plt.ylabel("Shipping Amount")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('method_comparison.png', dpi=300)
        plt.close()
        print("\nSaved comparison plot to 'method_comparison.png'")

def solve_with_maxflow(costs, capacities, demands):
    """Solve using max-flow min-cost with proper supply/demand balance"""
    G = nx.DiGraph()
    
    total_supply = sum(capacities)
    total_demand = sum(demands)
    
    # Add source and sink with balanced supply/demand
    G.add_node('source', demand=-(total_supply - total_demand))
    G.add_node('sink', demand=0)  # All demand handled through retail nodes
    
    # Add factory nodes with their actual capacities
    for i in range(len(capacities)):
        G.add_node(f'F{i+1}')
        G.add_edge('source', f'F{i+1}', capacity=capacities[i], weight=0)
    
    # Add retail nodes with their actual demands
    for j in range(len(demands)):
        G.add_node(f'R{j+1}')
        G.add_edge(f'R{j+1}', 'sink', capacity=demands[j], weight=0)
    
    # Add factory-retail edges with infinite capacity (or large enough)
    for i in range(len(capacities)):
        for j in range(len(demands)):
            G.add_edge(f'F{i+1}', f'R{j+1}', 
                      capacity=min(capacities[i], demands[j])*100,  # Large capacity
                      weight=costs[i,j])
    
    try:
        # Solve min-cost flow
        flow_dict = nx.min_cost_flow(G)
        
        # Extract solution
        solution = np.zeros_like(costs)
        for i in range(len(capacities)):
            for j in range(len(demands)):
                solution[i,j] = flow_dict[f'F{i+1}'].get(f'R{j+1}', 0)
        
        total_cost = (solution * costs).sum()
        
        return solution, total_cost, 0  # 0 is placeholder for time
        
    except nx.NetworkXUnfeasible:
        print("Warning: Max-flow problem is infeasible. Trying alternative formulation...")
        return solve_with_maxflow_alternative(costs, capacities, demands)

def solve_with_maxflow_alternative(costs, capacities, demands):
    """Alternative formulation when exact balance isn't possible"""
    G = nx.DiGraph()
    
    # Add source and sink
    G.add_node('source')
    G.add_node('sink')
    
    # Add factories with edges from source
    for i, cap in enumerate(capacities):
        G.add_node(f'F{i+1}')
        G.add_edge('source', f'F{i+1}', capacity=cap, weight=0)
    
    # Add retailers with edges to sink
    for j, dem in enumerate(demands):
        G.add_node(f'R{j+1}')
        G.add_edge(f'R{j+1}', 'sink', capacity=dem, weight=0)
    
    # Add factory-retail edges
    for i in range(len(capacities)):
        for j in range(len(demands)):
            G.add_edge(f'F{i+1}', f'R{j+1}', 
                      capacity=min(capacities[i], demands[j]),
                      weight=costs[i,j])
    
    # Solve as maximum flow with minimum cost
    flow_value, flow_dict = nx.maximum_flow(G, 'source', 'sink')
    min_cost = 0
    solution = np.zeros_like(costs)
    
    for i in range(len(capacities)):
        for j in range(len(demands)):
            flow = flow_dict[f'F{i+1}'].get(f'R{j+1}', 0)
            solution[i,j] = flow
            min_cost += flow * costs[i,j]
    
    return solution, min_cost, 0

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
                      weight=costs[i,j])  # This is crucial for cost calculation
    
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
            solution[i,j] = flow
            total_cost += flow * costs[i,j]
    
    # Verification
    print("\nMax-Flow Verification:")
    print("Total shipped from F1:", solution[0].sum())
    print("Total shipped from F2:", solution[1].sum())
    print("Total received by outlets:", solution.sum(axis=0))
    print("Actual demands:", demands)
    print("Flow costs:", solution * costs)
    print("Total calculated cost:", total_cost)
    
    return solution, total_cost, 0
if __name__ == "__main__":
    unittest.main()
