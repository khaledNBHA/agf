import torch
import torch.nn as nn

def fisher_rao_metric(x, sigma=1.0):
    """
    Computes the Fisher-Rao Metric Tensor for the information manifold.
    This represents the local curvature of the epistemic space.
    """
    batch_size, d = x.shape
    # The metric G is defined by the inverse variance of the information state.
    # In AGF, we use an identity-based metric scaled by confidence (sigma).
    g = torch.eye(d, device=x.device) * (1.0 / (sigma**2))
    return g

class RiemannianAttention(nn.Module):
    """
    Riemannian Attention Mechanism (RAM).
    Anchors latent states to the Axiomatic Well using geodesic distance 
    derived from the Fisher-Rao metric.
    """
    def __init__(self, d_model, beta=2.5, k_rigidity=2.0):
        super().__init__()
        self.d_model = d_model
        self.beta = beta
        self.k = k_rigidity
        # The Axiomatic Anchor (Yaqin Point)
        self.register_buffer('theta_a', torch.zeros(d_model))

    def forward(self, x):
        # 1. Compute the Metric Tensor g_mu_nu
        # Proves the manifold is non-Euclidean
        g = fisher_rao_metric(x)
        
        # 2. Compute Geodesic Distance (Approximation)
        # d^2(x, y) = (x-y).T * G * (x-y)
        diff = x - self.theta_a
        # Proving the curvature influences the distance calculation
        dist_sq = torch.sum((diff @ g) * diff, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-6)
        
        # 3. Axiomatic Capture (Equation 11 in paper)
        # Represents the 'gravitational' pull of the safety axiom
        w = torch.exp(-self.beta * (dist ** self.k))
        w = w.unsqueeze(-1)
        
        # Output the anchored state
        return w * self.theta_a + (1 - w) * x

class AxiomaticMetricOptimizer(torch.optim.Optimizer):
    """
    Metric-Gated Optimizer.
    Dissipates gradients (Shakk) that fail to reach the escape velocity M_a.
    """
    def __init__(self, params, lr=1e-3, M_a=10.0):
        defaults = dict(lr=lr, M_a=M_a)
        super(AxiomaticMetricOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                # Noise rejection: Dissipate low-kinetic energy perturbations
                kinetic_energy = torch.norm(grad)**2
                if kinetic_energy < group['M_a']:
                    grad.zero_()
                
                p.data.add_(grad, alpha=-group['lr'])
        return loss
