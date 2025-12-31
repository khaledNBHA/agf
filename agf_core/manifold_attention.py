import torch
import torch.nn as nn
import torch.nn.functional as F

class RiemannianAttention(nn.Module):
    """
    Implementation of the Riemannian Attention Mechanism (RAM) as described 
    in 'Axiomatic Gravity'. Anchors latent states using singular potentials.
    """
    def __init__(self, d_model, beta=2.5, k_rigidity=2.0):
        super(RiemannianAttention, self).__init__()
        self.d_model = d_model
        self.beta = beta  # Damping parameter
        self.k = k_rigidity # Rigidity index
        
        # Axiomatic Anchor (theta_A) - Initialize as zeros or pre-defined safety state
        self.register_buffer('theta_a', torch.zeros(d_model))

    def geodesic_distance(self, x, y):
        """
        Computes an approximation of the Fisher-Rao distance 
        on the latent manifold.
        """
        return torch.norm(x - y, p=2, dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input latent state of shape (batch, d_model)
        Returns:
            Tensor: Attenuated state anchored by the Axiomatic Well
        """
        # Calculate distance to the safety anchor
        dist = self.geodesic_distance(x, self.theta_a)
        
        # Equation (11): Metric-Gated Attention Weights
        # Weights decay exponentially as they drift from theta_A
        w = torch.exp(-self.beta * (dist ** self.k))
        
        # Reshape for broadcasting
        w = w.unsqueeze(-1)
        
        # The anchored state: a linear interpolation forced by the metric
        # Ensuring the state persists near the Axiomatic Gravity Well
        anchored_x = w * self.theta_a + (1 - w) * x
        
        return anchored_x

class AxiomaticMetricOptimizer(torch.optim.Optimizer):
    """
    Riemannian Gradient Descent optimized for the Axiomatic Gravity Manifold.
    Implements the Metric-Gated Gradient described in Section V-B.
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
                
                # Apply the metric tensor g_mu_nu influence
                # Here we simulate the 'well' effect on the gradient flow
                grad = p.grad.data
                
                # If gradient kinetic energy < Axiomatic Mass M_a, 
                # the gradient is dissipated (Noise rejection)
                kinetic_energy = torch.norm(grad)**2
                if kinetic_energy < group['M_a']:
                    # Dissipate Shakk (Doubt/Noise)
                    grad.zero_()
                
                p.data.add_(grad, alpha=-group['lr'])
        return loss
