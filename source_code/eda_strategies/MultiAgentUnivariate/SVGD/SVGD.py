import torch
import numpy as np

class SVGD:
    def __init__(self, kernel, gamma=10.0, no_repulsion=False):
        self.kernel = kernel  
        if gamma == 0:
            raise ValueError("gamma must be non-zero.")
        self.gamma = float(gamma)
        self.no_repulsion = bool(no_repulsion)
        self.last_kernel_stats = None
        self.last_force_stats = None

    def phi(self, thetas, score, probs=None):
        """
        B => Nombre d'instances
        N => Nombre de variables
        M => Nombre de particules (agents)
        theta : (B, M, N)   particles per batch (B instances, M agents, N dims)
        score : (B, M, N)   ∇_theta log p(theta) supplied by RL agents

        Standard SVGD update:
            φ_i = (1/M) * [ Σ_j k(θ_j, θ_i) * score_j  +  Σ_j ∇_θ_j k(θ_j, θ_i) ]
        """
        if thetas.dim() == 4:
            B, M, N, _ = thetas.shape
        else:
            B, M, N = thetas.shape
        if M == 1:
            return score / self.gamma
        K, grad_term = self.kernel(thetas, probs=probs)                        
        if torch.isnan(K).any() or torch.isinf(K).any():
            K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(grad_term).any() or torch.isinf(grad_term).any():
            grad_term = torch.nan_to_num(grad_term, nan=0.0, posinf=0.0, neginf=0.0)
        if score.dim() == 4:
            score_term = (K.unsqueeze(-1).unsqueeze(-1) * score.unsqueeze(1)).sum(dim=2)
        else:
            score_term = torch.matmul(K, score)
        attraction = (score_term / self.gamma) / M
        repulsion = torch.zeros_like(attraction) if self.no_repulsion else grad_term / M
        phi = attraction + repulsion

        if torch.isnan(phi).any() or torch.isinf(phi).any():
            phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_kernel_stats = {
            "avg_kernel_value": float(K.mean().item()),
            "avg_kernel_grad": float(grad_term.mean().item()),
        }
        with torch.no_grad():
            attraction_flat = attraction.reshape(B, M, -1)
            repulsion_flat = repulsion.reshape(B, M, -1)
            self.last_force_stats = {
                "attraction_per_agent": attraction_flat.norm(dim=-1).mean(dim=0).cpu().tolist(),
                "repulsion_per_agent": repulsion_flat.norm(dim=-1).mean(dim=0).cpu().tolist(),
            }
        return phi

    def get_last_kernel_stats(self):
        return self.last_kernel_stats or None

    def get_last_force_stats(self):
        return self.last_force_stats or None
