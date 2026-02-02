# file: run_adversary_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ribdp_core import RIBDPConfig, CyberSecurityPOMDP, exponential_tilt, RESULTS_DIR, EPS_SAFE

def main():
    """Analyze and plot the behavior of the exponential-tilt adversary."""
    print("Running adversary behavior analysis...")
    config = RIBDPConfig()
    pomdp = CyberSecurityPOMDP(config)
    
    test_belief = np.array([0.1, 0.6, 0.25, 0.05]) # Belief focused on 'Reconnaissance'
    kl_budgets = config.kl_budgets
    
    adversary_results = []
    
    for action in range(config.n_actions):
        nominal_obs = np.sum(test_belief[:, None] * pomdp.obs_model, axis=0)
        
        loss = np.zeros(pomdp.X)
        for x in range(pomdp.X):
            likelihood = pomdp.obs_model[:, x]
            post_unnorm = test_belief * likelihood
            normalizer = np.sum(post_unnorm)
            if normalizer > EPS_SAFE:
                posterior = post_unnorm / normalizer
                loss[x] = np.sum(posterior * pomdp.cost_matrix[:, action])
        
        for kl_budget in kl_budgets:
            adv_obs = exponential_tilt(nominal_obs, loss, kl_budget)
            tv_distance = 0.5 * np.sum(np.abs(adv_obs - nominal_obs))
            max_ratio = np.max(adv_obs / np.clip(nominal_obs, EPS_SAFE, None))
            
            adversary_results.append({
                'action': f'Action {action}',
                'kl_budget': kl_budget,
                'tv_distance': tv_distance,
                'max_likelihood_ratio': max_ratio,
            })
    
    adv_df = pd.DataFrame(adversary_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for action_name in adv_df['action'].unique():
        subset = adv_df[adv_df['action'] == action_name]
        ax1.plot(subset['kl_budget'], subset['tv_distance'], 'o-', label=action_name, linewidth=2)
        ax2.plot(subset['kl_budget'], subset['max_likelihood_ratio'], 'o-', label=action_name, linewidth=2)

    ax1.set_xlabel('KL Budget ε')
    ax1.set_ylabel('Total Variation Distance')
    ax1.set_title('Observation Distribution Shift')
    ax1.legend()
    
    ax2.set_xlabel('KL Budget ε')
    ax2.set_ylabel('Max Likelihood Ratio q(x)/p(x)')
    ax2.set_title('Maximum Adversarial Amplification')
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figure_adversary_behavior.png", dpi=300)
    plt.close()

    print(f"Adversary analysis complete. Graph saved to '{RESULTS_DIR}/'.")

if __name__ == "__main__":
    main()