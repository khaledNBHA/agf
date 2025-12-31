import torch
import numpy as np

class UAVAxiomaticEnv:
    """
    UAV Simulation Environment for Safety-Critical Navigation.
    The goal is to maintain a stable hover despite severe sensor noise (Shakk).
    """
    def __init__(self, state_dim=12, safety_threshold=1.0):
        self.state_dim = state_dim
        self.safety_threshold = safety_threshold
        # The Axiomatic Safe State (Perfect Hover)
        self.safe_state = torch.zeros(state_dim)
        self.safe_state[2] = 5.0  # Target altitude: 5 meters
        
    def get_stochastic_observation(self, true_state, noise_level=2.0):
        """
        Adds high-entropy Gaussian noise to the state.
        Represents 'Shakk' (Doubt/Uncertainty).
        """
        noise = torch.randn(self.state_dim) * noise_level
        return true_state + noise

    def compute_hjb_loss(self, state):
        """
        Simplified Hamilton-Jacobi-Bellman penalty for safety violation.
        """
        altitude = state[2]
        if altitude < self.safety_threshold:
            return torch.tensor(100.0, requires_grad=True)
        return torch.norm(state - self.safe_state, p=2)
