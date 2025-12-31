import torch
import numpy as np
from agf_core.manifold_attention import RiemannianAttention
from sim_uav.drone_env import UAVAxiomaticEnv

def run_rigorous_benchmark(n_trials=100):
    env = UAVAxiomaticEnv()
    ram = RiemannianAttention(d_model=12)
    ram.theta_a.data[2] = 5.0 # Safety Target
    
    noise_levels = [1.0, 2.0, 3.0, 5.0]
    results = {}

    for sigma in noise_levels:
        vanilla_errors = []
        agf_errors = []
        
        for _ in range(n_trials):
            true_state = torch.tensor([0., 0., 5.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            obs = env.get_stochastic_observation(true_state, noise_level=sigma)
            
            # Vanilla Performance
            vanilla_errors.append(torch.norm(obs[2] - true_state[2]).item())
            
            # AGF Performance
            anchored = ram(obs.unsqueeze(0)).squeeze(0)
            agf_errors.append(torch.norm(anchored[2] - true_state[2]).item())
            
        results[sigma] = {
            "vanilla_mean": np.mean(vanilla_errors),
            "agf_mean": np.mean(agf_errors),
            "improvement": (1 - np.mean(agf_errors)/np.mean(vanilla_errors)) * 100
        }

    # Print a professional scientific table
    print(f"{'Noise (σ)':<10} | {'Vanilla Error':<15} | {'AGF Error':<15} | {'Improvement'}")
    print("-" * 60)
    for sigma, data in results.items():
        print(f"{sigma:<10.1f} | {data['vanilla_mean']:<15.4f} | {data['agf_mean']:<15.4f} | {data['improvement']:.2f}%")

if __name__ == "__main__":
    run_rigorous_benchmark()
