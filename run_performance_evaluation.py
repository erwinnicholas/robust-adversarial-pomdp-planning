# file: run_performance_evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
# Reserved for future parallel value-iteration experiments
from concurrent.futures import ProcessPoolExecutor
import os
from ribdp_core import RIBDPConfig, CyberSecurityPOMDP, RIBDPSolver, BaselineSolver, set_seed, evaluate_policy, RESULTS_DIR

def run_single_experiment(args):
    """Run a single RIBDP vs Baseline comparison for a given KL budget and seed."""
    kl_budget, config, seed = args
    set_seed(seed)
    
    pomdp = CyberSecurityPOMDP(config)
    
    # Solve and evaluate RIBDP
    ribdp_solver = RIBDPSolver(pomdp, kl_budget)
    ribdp_solver.value_iteration(verbose=False)
    ribdp_reward = evaluate_policy(ribdp_solver, episodes=200)
    
    # Create a baseline policy (equivalent to RIBDP with kl_budget=0)
    baseline_solver = BaselineSolver(pomdp, kl_budget=0.0) # Evaluated under same adversarial conditions
    baseline_policy_solver = BaselineSolver(pomdp, kl_budget=0.0)
    baseline_policy_solver.value_iteration(verbose=False)
    
    # We need to evaluate the baseline policy under the *adversarial* conditions of the current experiment
    # So we create a temporary solver with the baseline policy but the current experiment's kl_budget for evaluation
    eval_solver = RIBDPSolver(pomdp, kl_budget)
    eval_solver.policy = baseline_policy_solver.policy
    baseline_reward = evaluate_policy(eval_solver, episodes=200)

    return {
        'kl_budget': kl_budget,
        'ribdp_reward': ribdp_reward,
        'baseline_reward': baseline_reward,
        'advantage': ribdp_reward - baseline_reward
    }

def main():
    """Run the comprehensive performance evaluation and generate the main plot."""
    print("Running comprehensive performance evaluation...")
    
    config = RIBDPConfig(
        belief_grid_size=30,
        kl_budgets=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    )
    seeds = range(5)
    
    args_list = [(kl, config, seed) for kl in config.kl_budgets for seed in seeds]
    
    all_results = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        results = list(executor.map(run_single_experiment, args_list))
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    summary = df.groupby('kl_budget').agg(
        ribdp_reward_mean=('ribdp_reward', 'mean'),
        ribdp_reward_std=('ribdp_reward', 'std'),
        baseline_reward_mean=('baseline_reward', 'mean'),
        baseline_reward_std=('baseline_reward', 'std'),
        advantage_mean=('advantage', 'mean'),
        advantage_std=('advantage', 'std')
    ).round(4)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    kl_vals = summary.index
    
    # Plot 1: Reward Comparison
    ax1.plot(kl_vals, summary['ribdp_reward_mean'], 'o-', label='RIBDP', linewidth=2)
    ax1.fill_between(kl_vals, 
                     summary['ribdp_reward_mean'] - summary['ribdp_reward_std'],
                     summary['ribdp_reward_mean'] + summary['ribdp_reward_std'],
                     alpha=0.2)
    ax1.plot(kl_vals, summary['baseline_reward_mean'], 's--', label='Baseline', linewidth=2)
    ax1.fill_between(kl_vals,
                     summary['baseline_reward_mean'] - summary['baseline_reward_std'],
                     summary['baseline_reward_mean'] + summary['baseline_reward_std'],
                     alpha=0.2)
    ax1.set_xlabel('KL Budget ε')
    ax1.set_ylabel('Average Cumulative Reward')
    ax1.legend()
    ax1.set_title('RIBDP vs. Baseline Performance')
    
    # Plot 2: Advantage
    ax2.plot(kl_vals, summary['advantage_mean'], 'o-', color='green', linewidth=2)
    ax2.fill_between(kl_vals,
                     summary['advantage_mean'] - summary['advantage_std'],
                     summary['advantage_mean'] + summary['advantage_std'],
                     alpha=0.2, color='green')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax2.set_xlabel('KL Budget ε')
    ax2.set_ylabel('Advantage (RIBDP Reward - Baseline Reward)')
    ax2.set_title('Robustness Advantage of RIBDP')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figure_performance_vs_baseline.png", dpi=300)
    plt.close()
    
    # ... (after the plotting section)
    print("\n--- Data for Table 1 ---")
    # Select and rename columns for the paper
    table_data = summary[['ribdp_reward_mean', 'ribdp_reward_std', 'baseline_reward_mean', 'baseline_reward_std']].copy()
    table_data.columns = ["RIBDP Mean Reward", "RIBDP Std Dev", "Baseline Mean Reward", "Baseline Std Dev"]
    # Print as Markdown for easy copying
    print(table_data.to_markdown())
    table_data.to_csv(f"{RESULTS_DIR}/table_performance_evaluation.csv", index=False)
    print(f"Performance evaluation complete. Graph saved to '{RESULTS_DIR}/'.")

if __name__ == "__main__":
    main()