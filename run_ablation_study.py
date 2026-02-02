# file: run_ablation_study.py

import pandas as pd
import matplotlib.pyplot as plt
import os
from ribdp_core import RIBDPConfig, CyberSecurityPOMDP, RIBDPSolver, BaselineSolver, evaluate_policy, set_seed, RESULTS_DIR

def main():
    """Run an ablation study to validate the contribution of each component."""
    print("Running ablation study...")
    set_seed(42)
    config = RIBDPConfig(belief_grid_size=25)
    pomdp = CyberSecurityPOMDP(config)
    kl_budget = 0.1
    
    # 1. Full RIBDP
    solver_full = RIBDPSolver(pomdp, kl_budget)
    solver_full.value_iteration()
    reward_full = evaluate_policy(solver_full, episodes=300)
    
    # 2. Baseline (Non-robust planner)
    # The baseline policy is trained assuming no adversary (kl=0)
    baseline_policy_solver = BaselineSolver(pomdp, kl_budget=0.0)
    baseline_policy_solver.value_iteration()
    # But it must be evaluated under adversarial conditions (kl=0.1)
    eval_solver = RIBDPSolver(pomdp, kl_budget)
    eval_solver.policy = baseline_policy_solver.policy # Use the non-robust policy
    reward_baseline = evaluate_policy(eval_solver, episodes=300)

    ablation_results = [
        {'Method': 'Full RIBDP', 'Reward': reward_full},
        {'Method': 'Baseline (Non-Robust)', 'Reward': reward_baseline}
    ]
    
    ablation_df = pd.DataFrame(ablation_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(7, 6))
    colors = ['#4c72b0', '#c44e52']
    bars = plt.bar(ablation_df['Method'], ablation_df['Reward'], color=colors, alpha=0.8)
    
    plt.ylabel('Average Cumulative Reward')
    plt.title(f'Ablation Study (Adversary Budget ε={kl_budget})')
    plt.xticks(rotation=10)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')
        
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figure_ablation_study.png", dpi=300)
    plt.close()
    
    print(f"Ablation study complete. Graph saved to '{RESULTS_DIR}/'.")

if __name__ == "__main__":
    main()