import torch

def compute_fisher_metric(x, sigma=1.0):
    """
    Computes the Fisher-Rao Metric Tensor G.
    In the AGF framework, this defines the local density of the information manifold.
    """
    d = x.shape[-1]
    # G is the identity matrix scaled by the inverse variance (precision)
    g = torch.eye(d, device=x.device) * (1.0 / (sigma**2))
    return g

def geodesic_distance(x1, x2, G):
    """
    Computes the Riemannian distance between x1 and x2 under metric G.
    d(x1, x2) = sqrt((x1-x2)^T * G * (x1-x2))
    """
    diff = x1 - x2
    # Quadratic form: represents the curvature-aware distance
    dist_sq = torch.sum((diff @ G) * diff, dim=-1)
    return torch.sqrt(dist_sq + 1e-6)
