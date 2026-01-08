import torch
import numpy as np

class SVGD:
    def __init__(self, kernel, gamma=10.0):
        self.kernel = kernel  
        if gamma == 0:
            raise ValueError("gamma must be non-zero.")
        self.gamma = float(gamma)
        self.last_kernel_stats = None

    def phi(self, thetas, score):
        """
        B => Nombre d'instances
        N => Nombre de variables
        M => Nombre de particules (agents)
        theta : (B, M, N)   particles per batch (B instances, M agents, N dims)
        score : (B, M, N)   ∇_theta log p(theta) supplied by RL agents

        Standard SVGD update:
            φ_i = (1/M) * [ Σ_j k(θ_j, θ_i) * score_j  +  Σ_j ∇_θ_j k(θ_j, θ_i) ]
        """
        B, M, N = thetas.shape

        # Gram matrix & gradients produced by the kernel itself
        # K[b, i, j] = k(θ_i, θ_j)
        # grad_first[b, i, j, :] = ∇_{θ_i} k(θ_i, θ_j)
        K, grad_term = self.kernel(thetas)  # (B, M, M), (B, M, N)
        if torch.isnan(K).any() or torch.isinf(K).any():
            K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(grad_term).any() or torch.isinf(grad_term).any():
            grad_term = torch.nan_to_num(grad_term, nan=0.0, posinf=0.0, neginf=0.0)


        # First SVGD term: Σ_j k(θ_j, θ_i) * score_j
        # matmul: (B, M, M) @ (B, M, N) -> (B, M, N)

        score_term = torch.matmul(K, score)
        # Average over M particles


        phi = (score_term / self.gamma + grad_term) / M  # (B, M, N)

        # phi = score


        if torch.isnan(phi).any() or torch.isinf(phi).any():
            phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_kernel_stats = {
            "avg_kernel_value": float(K.mean().item()),
            "avg_kernel_grad": float(grad_term.mean().item()),
        }
        return phi

    def get_last_kernel_stats(self):
        return self.last_kernel_stats or None
