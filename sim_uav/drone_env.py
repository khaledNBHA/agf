import torch

class UAVAxiomaticEnv:
    """
    UAV Simulation Environment for Safety-Critical Navigation.
    Designed to test Epistemic Drift (Shakk) under extreme sensor noise.
    """
    def __init__(self, state_dim=12):
        self.state_dim = state_dim
        # Target: Stable hover at 5 meters altitude
        self.safe_state = torch.zeros(state_dim)
        self.safe_state[2] = 5.0 
        
    def get_stochastic_observation(self, true_state, sigma=1.0):
        """
        Generates a perturbed observation representing 'Shakk'.
        """
        noise = torch.randn_like(true_state) * sigma
        return true_state + noise

    def evaluate_safety_violation(self, state):
        """
        Measures the deviation from the Axiomatic safety zone.
        """
        altitude_error = torch.abs(state[2] - self.safe_state[2])
        return altitude_error.item()
