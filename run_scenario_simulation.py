# file: run_scenario_simulation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ribdp_core import RESULTS_DIR

def main():
    """Simulate a realistic cybersecurity detection scenario."""
    print("Running realistic dataset simulation...")
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Scikit-learn not found. Please run 'pip install scikit-learn'.")
        return
        
    np.random.seed(42)
    n_samples, n_features = 5000, 20
    
    normal_features = np.random.multivariate_normal(mean=np.zeros(n_features), cov=np.eye(n_features), size=n_samples // 2)
    attack_features = np.random.multivariate_normal(mean=2 * np.ones(n_features), cov=1.5 * np.eye(n_features), size=n_samples // 2)
    
    X = np.vstack([normal_features, attack_features])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    
    classifier = LogisticRegression(random_state=42).fit(X_train_s, y_train)
    test_probs = classifier.predict_proba(X_test_s)[:, 1]

    detection_results = []
    for kl_budget in [0.0, 0.05, 0.1, 0.2, 0.4]:
        corrupted_probs = test_probs.copy()
        if kl_budget > 0:
            # Simulate adversarial perturbation
            noise = np.random.normal(0, kl_budget * 0.5, len(corrupted_probs))
            corrupted_probs += noise
            corrupted_probs = np.clip(corrupted_probs, 0, 1)
        
        predictions = (corrupted_probs > 0.5).astype(int)
        tp = np.sum((predictions == 1) & (y_test == 1))
        fp = np.sum((predictions == 1) & (y_test == 0))
        tn = np.sum((predictions == 0) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        detection_results.append({'kl_budget': kl_budget, 'f1_score': f1, 'accuracy': (tp + tn) / len(y_test)})
        
    detection_df = pd.DataFrame(detection_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    plt.plot(detection_df['kl_budget'], detection_df['f1_score'], 'o-', label='F1 Score', linewidth=2)
    plt.plot(detection_df['kl_budget'], detection_df['accuracy'], 's--', label='Accuracy', linewidth=2)
    plt.xlabel('Adversarial Perturbation Strength (Simulated ε)')
    plt.ylabel('Performance Metric')
    plt.legend()
    plt.title('Detection Performance Under Adversarial Perturbation')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=min(0.5, detection_df['f1_score'].min() - 0.05))
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figure_realistic_scenario.png", dpi=300)
    plt.close()
    
    print(f"Scenario simulation complete. Graph saved to '{RESULTS_DIR}/'.")

if __name__ == "__main__":
    main()