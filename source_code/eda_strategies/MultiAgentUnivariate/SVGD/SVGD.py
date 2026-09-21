import torch
import numpy as np

class SVGD:
    def __init__(self, kernel, gamma=10.0, no_repulsion=False):
        self.kernel = kernel
        self.gamma = float(gamma)
        self.no_repulsion = bool(no_repulsion)
        self._last_K = None
        self._last_grad_term = None
        self._last_attraction = None
        self._last_repulsion = None

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
        M = thetas.shape[1]
        if M == 1:
            return score / self.gamma
        K, grad_term = self.kernel(thetas, probs=probs)
        K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        grad_term = torch.nan_to_num(grad_term, nan=0.0, posinf=0.0, neginf=0.0)
        if score.dim() == 4:
            score_term = (K.unsqueeze(-1).unsqueeze(-1) * score.unsqueeze(1)).sum(dim=2)
        else:
            score_term = torch.matmul(K, score)
        attraction = (score_term / self.gamma) / M
        repulsion = torch.zeros_like(attraction) if self.no_repulsion else grad_term / M
        phi = torch.nan_to_num(attraction + repulsion, nan=0.0, posinf=0.0, neginf=0.0)

        self._last_K = K.detach()
        self._last_grad_term = grad_term.detach()
        self._last_attraction = attraction.detach()
        self._last_repulsion = repulsion.detach()
        return phi

    def get_last_kernel_stats(self):
        if self._last_K is None:
            return None
        return {
            "avg_kernel_value": float(self._last_K.mean().item()),
            "avg_kernel_grad": float(self._last_grad_term.mean().item()),
        }

    def get_last_force_stats(self):
        if self._last_attraction is None:
            return None
        B, M = self._last_attraction.shape[:2]
        attraction_flat = self._last_attraction.reshape(B, M, -1)
        repulsion_flat = self._last_repulsion.reshape(B, M, -1)
        return {
            "attraction_per_agent": attraction_flat.norm(dim=-1).mean(dim=0).cpu().tolist(),
            "repulsion_per_agent": repulsion_flat.norm(dim=-1).mean(dim=0).cpu().tolist(),
        }
