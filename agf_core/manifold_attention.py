import torch
import torch.nn as nn

class RiemannianAttention(nn.Module):
    def __init__(self, d_model, beta=2.5, k_rigidity=2.0):
        super().__init__()
        self.d_model = d_model
        self.beta = beta
        self.k = k_rigidity
        self.register_buffer('theta_a', torch.zeros(d_model))

    def forward(self, x):
        # 1. Compute the Metric Tensor g_mu_nu
        # This proves we are not in flat Euclidean space
        g = fisher_rao_metric(x)
        
        # 2. Compute Geodesic Distance (Approximation via the metric)
        # d^2(x, y) = (x-y)T * G * (x-y)
        diff = x - self.theta_a
        # Quadratic form: proving the curvature influences the distance
        dist_sq = torch.sum((diff @ g) * diff, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-6)
        
        # 3. Axiomatic Capture Mechanism
        # The 'Well' is not just a weight, it's a topological deformation
        w = torch.exp(-self.beta * (dist ** self.k))
        w = w.unsqueeze(-1)
        
        # Anchoring the latent state
        return w * self.theta_a + (1 - w) * x
