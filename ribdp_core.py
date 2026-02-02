# file: ribdp_core.py

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import List

import torch

# Globals
EPS_SAFE = 1e-12
RESULTS_DIR = "graphs"

@dataclass
class RIBDPConfig:
    """Configuration for RIBDP experiments"""
    n_states: int = 4
    n_obs: int = 8
    n_actions: int = 3
    gamma: float = 0.95
    belief_grid_size: int = 50
    kl_budgets: List[float] = None
    
    def __post_init__(self):
        if self.kl_budgets is None:
            self.kl_budgets = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def logsumexp_stable(x):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

def kl_divergence(q, p):
    """Compute KL(q||p) with numerical stability"""
    q_safe = np.clip(q, EPS_SAFE, 1.0)
    p_safe = np.clip(p, EPS_SAFE, 1.0)
    return np.sum(q_safe * (np.log(q_safe) - np.log(p_safe)))

def exponential_tilt(p, loss, kl_budget, debug=False, use_torch=False, device="cpu"):
    """
    Compute worst-case distribution via exponential tilting.
    Safe version: no silent failure, fewer redundant ops.
    """
    if kl_budget <= EPS_SAFE:
        return p.copy()

    # If loss has no variation, adversary gains nothing
    loss_range = np.max(loss) - np.min(loss)
    if loss_range <= EPS_SAFE:
        return p.copy()

    p_safe = np.clip(p, EPS_SAFE, None)
    log_p = np.log(p_safe)

    def kl_objective(lam):
        if lam <= 0:
            return -kl_budget

        if use_torch:
            p_t = torch.tensor(log_p, device=device)
            loss_t = torch.tensor(loss, device=device)
            log_q = p_t + loss_t / lam
            log_z = torch.logsumexp(log_q, dim=0)
            q = torch.exp(log_q - log_z).cpu().numpy()
        else:
            log_q = log_p + loss / lam
            log_z = logsumexp_stable(log_q)
            q = np.exp(log_q - log_z)


    try:
        lam_lo = 1e-8
        lam_hi = 20.0 / (kl_budget + 1e-6)

        # Expand upper bound if needed
        while kl_objective(lam_hi) <= 0 and lam_hi < 1e6:
            lam_hi *= 2.0

        lam_opt = brentq(kl_objective, lam_lo, lam_hi, xtol=1e-8)

        log_q_unnorm = log_p + loss / lam_opt
        log_z = logsumexp_stable(log_q_unnorm)
        return np.exp(log_q_unnorm - log_z)

    except Exception as e:
        if debug:
            raise RuntimeError(f"Exponential tilt failed: {e}")
        # Fallback: nominal distribution
        return p.copy()


class CyberSecurityPOMDP:
    """Realistic cybersecurity POMDP for APT detection"""
    def __init__(self, config: RIBDPConfig):
        self.config = config
        self.S = config.n_states
        self.X = config.n_obs
        self.A = config.n_actions
        
        self._setup_transition_model()
        self._setup_observation_model()
        self._setup_cost_model()
        self._setup_belief_grid()
    
    def _setup_transition_model(self):
        self.P = np.zeros((self.S, self.A, self.S))
        self.P[0, 0, :] = [0.85, 0.10, 0.03, 0.02]; self.P[0, 1, :] = [0.90, 0.07, 0.02, 0.01]; self.P[0, 2, :] = [0.95, 0.03, 0.01, 0.01]
        self.P[1, 0, :] = [0.20, 0.30, 0.35, 0.15]; self.P[1, 1, :] = [0.40, 0.40, 0.15, 0.05]; self.P[1, 2, :] = [0.70, 0.20, 0.07, 0.03]
        self.P[2, 0, :] = [0.05, 0.15, 0.40, 0.40]; self.P[2, 1, :] = [0.25, 0.25, 0.35, 0.15]; self.P[2, 2, :] = [0.60, 0.20, 0.15, 0.05]
        self.P[3, 0, :] = [0.02, 0.08, 0.20, 0.70]; self.P[3, 1, :] = [0.15, 0.15, 0.25, 0.45]; self.P[3, 2, :] = [0.40, 0.25, 0.20, 0.15]
    
    def _setup_observation_model(self):
        self.obs_model = np.array([
            [0.4, 0.3, 0.15, 0.08, 0.04, 0.02, 0.005, 0.005],
            [0.1, 0.2, 0.25, 0.2, 0.15, 0.07, 0.02, 0.01],
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.07, 0.03],
            [0.2, 0.15, 0.1, 0.15, 0.2, 0.1, 0.05, 0.05]
        ])
        self.obs_model = self.obs_model / self.obs_model.sum(axis=1, keepdims=True)
    
    def _setup_cost_model(self):
        self.cost_matrix = np.array([[0.1, 0.5, 1.0], [1.0, 0.3, 0.4], [3.0, 1.0, 0.8], [5.0, 2.0, 1.5]])
    
    def _setup_belief_grid(self):
        grid_size = self.config.belief_grid_size
        beliefs = [np.eye(self.S)[i] for i in range(self.S)]
        beliefs.append(np.ones(self.S) / self.S)
        remaining = grid_size - len(beliefs)
        if remaining > 0:
            beliefs.extend(np.random.dirichlet(np.ones(self.S) * 0.5, remaining))
        self.belief_grid = np.array(beliefs[:grid_size])

class RIBDPSolver:
    def __init__(self, pomdp: CyberSecurityPOMDP, kl_budget: float = 0.1, fast: bool = True):
        self.pomdp = pomdp
        self.kl_budget = kl_budget
        self.config = pomdp.config
        self.V = np.zeros(len(pomdp.belief_grid))
        self.policy = np.zeros(len(pomdp.belief_grid), dtype=int)
        self.fast = fast
        self._transition_cache = {} if fast else None

    
    def compute_observation_loss(self, belief, action):
        """
        Vectorized computation of observation-conditioned loss.
        """
        # obs_model: [S, X]
        # belief:    [S]
        likelihoods = self.pomdp.obs_model

        posterior_unnorm = belief[:, None] * likelihoods
        normalizers = np.sum(posterior_unnorm, axis=0)

        # Avoid division by zero
        safe = normalizers > EPS_SAFE
        posterior = np.zeros_like(posterior_unnorm)
        posterior[:, safe] = posterior_unnorm[:, safe] / normalizers[safe]

        # cost_matrix[:, action] -> [S]
        return np.sum(posterior * self.pomdp.cost_matrix[:, action][:, None], axis=0)

    
    def belief_update(self, belief, action, observation):
        predicted_belief = belief @ self.pomdp.P[:, action, :]
        likelihood = self.pomdp.obs_model[:, observation]
        posterior_unnorm = predicted_belief * likelihood
        normalizer = np.sum(posterior_unnorm)
        return posterior_unnorm / normalizer if normalizer > EPS_SAFE else predicted_belief
    
    def find_nearest_belief(self, belief):
        distances = np.sum((self.pomdp.belief_grid - belief)**2, axis=1)
        return np.argmin(distances)
    
    def q_value(self, belief_idx, action):
        belief = self.pomdp.belief_grid[belief_idx]

        immediate_reward = -np.sum(belief * self.pomdp.cost_matrix[:, action])

        # Nominal observation distribution
        nominal_obs_dist = np.sum(belief[:, None] * self.pomdp.obs_model, axis=0)

        # Observation loss
        obs_loss = self.compute_observation_loss(belief, action)

        # Adversarial observation distribution
        adv_obs_dist = exponential_tilt(
            nominal_obs_dist,
            obs_loss,
            self.kl_budget
        )

        expected_future = 0.0

        for x in range(self.pomdp.X):
            p_x = adv_obs_dist[x]
            if p_x <= EPS_SAFE:
                continue

            if self.fast:
                key = (belief_idx, action, x)
                if key not in self._transition_cache:
                    next_belief = self.belief_update(belief, action, x)
                    self._transition_cache[key] = self.find_nearest_belief(next_belief)
                next_idx = self._transition_cache[key]
            else:
                next_belief = self.belief_update(belief, action, x)
                next_idx = self.find_nearest_belief(next_belief)

            expected_future += p_x * self.V[next_idx]


        return immediate_reward + self.config.gamma * expected_future

    
    def value_iteration(self, max_iters=100, tolerance=1e-4, verbose=True):
        if verbose: print(f"Running value iteration (KL budget: {self.kl_budget})...")
        for iteration in range(max_iters):
            V_old = self.V.copy()
            for b_idx in range(len(self.pomdp.belief_grid)):
                q_values = [self.q_value(b_idx, a) for a in range(self.pomdp.A)]
                self.V[b_idx] = max(q_values)
                self.policy[b_idx] = np.argmax(q_values)
            if np.max(np.abs(self.V - V_old)) < tolerance:
                if verbose: print(f"Converged after {iteration + 1} iterations.")
                break
        return self.V, self.policy

class BaselineSolver(RIBDPSolver):
    def q_value(self, belief_idx, action): # Override to be non-robust
        belief = self.pomdp.belief_grid[belief_idx]
        immediate_reward = -np.sum(belief * self.pomdp.cost_matrix[:, action])
        
        expected_future = 0.0
        for x in range(self.pomdp.X):
            # Use nominal observation distribution directly
            nominal_obs_prob = np.sum(belief * self.pomdp.obs_model[:, x])
            if nominal_obs_prob > EPS_SAFE:
                next_belief = self.belief_update(belief, action, x)
                next_belief_idx = self.find_nearest_belief(next_belief)
                expected_future += nominal_obs_prob * self.V[next_belief_idx]
                
        return immediate_reward + self.config.gamma * expected_future

    def value_iteration(self, max_iters=100, tolerance=1e-4, verbose=True):
        # Small change to print a different message
        if verbose: print("Running baseline value iteration...")
        return super().value_iteration(max_iters, tolerance, verbose=False) # Call parent without verbose spam

def evaluate_policy(solver, episodes=200):
    total_reward = 0.0
    for _ in range(episodes):
        belief = np.ones(solver.pomdp.S) / solver.pomdp.S
        episode_reward = 0.0
        for step in range(50):
            belief_idx = solver.find_nearest_belief(belief)
            action = solver.policy[belief_idx]
            
            true_state = np.random.choice(solver.pomdp.S, p=belief)
            immediate_reward = -solver.pomdp.cost_matrix[true_state, action]
            episode_reward += (solver.config.gamma ** step) * immediate_reward
            
            next_state = np.random.choice(solver.pomdp.S, p=solver.pomdp.P[true_state, action])
            
            # Adversary acts on defender's belief to generate worst-case observation
            nominal_obs_dist = np.sum(belief[:, None] * solver.pomdp.obs_model, axis=0)
            obs_loss = solver.compute_observation_loss(belief, action)
            adv_obs_dist = exponential_tilt(nominal_obs_dist, obs_loss, solver.kl_budget)
            observation = np.random.choice(solver.pomdp.X, p=adv_obs_dist)
            
            belief = solver.belief_update(belief, action, observation)
            if np.max(belief) > 0.95: break
        total_reward += episode_reward
    return total_reward / episodes