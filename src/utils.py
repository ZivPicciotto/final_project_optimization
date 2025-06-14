import numpy as np
from test.examples import MathFunction
import matplotlib.pyplot as plt


class BarrierFunction(MathFunction):
    def __init__(self, func, ineq_constraints, t):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.t = t

    # In utils.py, modify BarrierFunction methods:
    def value(self, x):
        constraint_vals = [-c.value(x) for c in self.ineq_constraints]
        min_val = min(constraint_vals)
        if min_val <= 1e-10:  # Too close to boundary
            return np.inf
        penalty = sum(-np.log(val) for val in constraint_vals)
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
                outer_product / (constraint_val**2) -
                c.hessian(x) / constraint_val
            )

        return self.t * hessian + penalty_hess


def plot_flow_network(solution, costs, a, b):
    plt.figure(figsize=(16, 10))
    pos = {}

    # Factory positions (left side)
    pos['F1'] = (0, 1.5)
    pos['F2'] = (0, 0.5)

    # Outlet positions (right side) - spread them out more
    for j in range(12):
        pos[f'R{j+1}'] = (3, 2.2 - j*0.2)  # Increased spacing

    # Draw factories and outlets
    for node, (x, y) in pos.items():
        if node.startswith('F'):
            plt.scatter(x, y, s=3000, c='lightblue', edgecolor='black')
            plt.text(x-0.3, y, f"{node}\nCapacity: {a[0] if node == 'F1' else a[1]}",
                     ha='center', va='center', fontweight='bold')
        else:
            plt.scatter(x, y, s=1000, c='salmon', edgecolor='black')
            outlet_num = int(node[1:])
            plt.text(x+0.3, y, f"{node}\nDemand: {b[outlet_num-1]}",
                     ha='center', va='center')

    # Draw flows with labels positioned near demand labels
    solution_mat = solution.reshape((2, 12))
    max_width = 8

    for j in range(12):  # For each outlet
        outlet_pos = pos[f'R{j+1}']
        label_x = outlet_pos[0] + 0.6
        label_y = outlet_pos[1]

        # Combine both factory flows for this outlet
        factory_flows = []
        for i in range(2):
            flow = solution_mat[i, j]
            if flow > 0.1:  # Only show meaningful flows
                width = max(0.5, flow/max(solution_mat.flatten())*max_width)

                # Draw the arrow
                plt.annotate("",
                             xy=outlet_pos, xytext=pos[f'F{i+1}'],
                             arrowprops=dict(arrowstyle="->",
                                             linewidth=width,
                                             color='green' if i == 0 else 'purple',
                                             alpha=0.7))

                factory_flows.append(f"F{i+1}: {flow:.1f} (${costs[i, j]})")

        # Create combined label for both factories
        if factory_flows:
            combined_label = "\n".join(factory_flows)
            plt.text(label_x, label_y, combined_label,
                     ha='center', va='center', fontsize=8,
                     bbox=dict(facecolor='lightyellow',
                               alpha=0.9, boxstyle="round,pad=0.2"))

    plt.title("Optimal Transportation Flow Network",
              fontsize=16, fontweight='bold')
    plt.xlim(-0.8, 4.5)
    plt.ylim(-0.5, 2.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('transport_network.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_utilization(solution, a):
    plt.figure(figsize=(12, 6))

    solution_mat = solution.reshape((2, 12))
    util1 = solution_mat[0, :].sum()/a[0]*100
    util2 = solution_mat[1, :].sum()/a[1]*100

    plt.subplot(1, 2, 1)
    plt.pie([util1, 100-util1],
            labels=[f'Used ({util1:.1f}%)', 'Remaining'],
            colors=['#66b3ff', '#99ccff'],
            autopct='%1.1f%%')
    plt.title("Factory 1 Capacity Utilization")

    plt.subplot(1, 2, 2)
    plt.pie([util2, 100-util2],
            labels=[f'Used ({util2:.1f}%)', 'Remaining'],
            colors=['#ff9999', '#ffcccc'],
            autopct='%1.1f%%')
    plt.title("Factory 2 Capacity Utilization")

    plt.tight_layout()
    plt.savefig('capacity_utilization.png', dpi=300)
    plt.close()


def plot_cost_breakdown(solution, costs):
    solution_mat = solution.reshape((2, 12))
    cost_by_factory = (solution_mat * costs).sum(axis=1)
    cost_by_outlet = (solution_mat * costs).sum(axis=0)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.bar(['Factory 1', 'Factory 2'], cost_by_factory,
            color=['skyblue', 'lightcoral'])
    plt.title("Total Cost by Factory")
    plt.ylabel("Cost ($)")
    for i, v in enumerate(cost_by_factory):
        plt.text(i, v+5, f"${v:.1f}", ha='center')

    plt.subplot(1, 2, 2)
    outlets = [f'Outlet {i+1}' for i in range(12)]
    plt.bar(outlets, cost_by_outlet, color='lightgreen')
    plt.title("Total Cost by Retail Outlet")
    plt.ylabel("Cost ($)")
    plt.xticks(rotation=45)
    for i, v in enumerate(cost_by_outlet):
        plt.text(i, v+2, f"${v:.1f}", ha='center', rotation=45, fontsize=8)

    plt.tight_layout()
    plt.savefig('cost_breakdown.png', dpi=300)
    plt.close()
