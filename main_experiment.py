import torch
from agf_core.manifold_attention import RiemannianAttention, AxiomaticMetricOptimizer
from sim_uav.drone_env import UAVAxiomaticEnv

def run_benchmarks():
    print("--- Starting AGF vs Vanilla Benchmark ---")
    env = UAVAxiomaticEnv()
    d_model = 12
    
    # Initialize AGF Components
    ram_layer = RiemannianAttention(d_model=d_model, beta=3.5, k_rigidity=2.0)
    # Set the gravity well at the safe altitude
    ram_layer.theta_a.data[2] = 5.0 
    
    # Test Scenarios
    true_state = torch.tensor([0.1, 0.1, 4.8, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # Near target
    noise_scenarios = [0.5, 1.5, 3.0, 5.0] # Increasing Shakk
    
    for sigma in noise_scenarios:
        print(f"\nTesting with Noise Level (Sigma): {sigma}")
        obs = env.get_stochastic_observation(true_state, noise_level=sigma)
        
        # 1. Vanilla Processing (No Gravity Well)
        vanilla_error = torch.norm(obs[2] - true_state[2]).item()
        
        # 2. AGF Processing (Riemannian Anchoring)
        with torch.no_grad():
            anchored_obs = ram_layer(obs.unsqueeze(0)).squeeze(0)
            agf_error = torch.norm(anchored_obs[2] - true_state[2]).item()
            
        print(f"Vanilla Deviation: {vanilla_error:.4f}")
        print(f"AGF Deviation:     {agf_error:.4f}")
        improvement = (1 - agf_error/vanilla_error)*100
        print(f"Safety Persistence: +{improvement:.2f}%")

if __name__ == "__main__":
    run_benchmarks()
