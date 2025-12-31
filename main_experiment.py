import torch
import numpy as np
from agf_core.manifold_attention import RiemannianAttention
from sim_uav.drone_env import UAVAxiomaticEnv

def run_scientific_benchmark():
    print("--- Running Axiomatic Gravity Framework (AGF) Rigorous Evaluation ---")
    env = UAVAxiomaticEnv()
    # Configuration matches the paper: beta=2.5, k=2.0
    ram = RiemannianAttention(d_model=12, beta=2.5, k_rigidity=2.0)
    ram.theta_a.data[2] = 5.0 # Set the Yaqin anchor at 5m
    
    noise_levels = [0.5, 1.5, 3.0, 5.0] # Scaling 'Shakk'
    n_trials = 100 # Scientific significance

    print(f"{'Noise (σ)':<10} | {'Vanilla Error':<15} | {'AGF Error':<15} | {'Persistence (%)'}")
    print("-" * 65)

    for sigma in noise_levels:
        v_errors = []
        a_errors = []
        
        for _ in range(n_trials):
            # Drone is at safe altitude, but sensors are failing
            true_state = torch.zeros(12)
            true_state[2] = 5.0
            
            obs = env.get_stochastic_observation(true_state, sigma=sigma)
            
            # 1. Vanilla (Standard AI): directly follows the noise
            v_err = env.evaluate_safety_violation(obs)
            v_errors.append(v_err)
            
            # 2. AGF (Our Framework): anchors the state to the manifold
            anchored_obs = ram(obs.unsqueeze(0)).squeeze(0)
            a_err = env.evaluate_safety_violation(anchored_obs)
            a_errors.append(a_err)
            
        m_v = np.mean(v_errors)
        m_a = np.mean(a_errors)
        persistence = (1 - m_a/m_v) * 100
        
        print(f"{sigma:<10.1f} | {m_v:<15.4f} | {m_a:<15.4f} | {persistence:.2f}%")

if __name__ == "__main__":
    run_scientific_benchmark()
