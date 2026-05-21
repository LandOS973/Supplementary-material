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

    def phi(self, thetas, score, probs=None, support_thetas=None, support_probs=None, support_mask=None):
        """
        B => Nombre d'instances
        N => Nombre de variables
        M => Nombre de particules (agents)
        theta : (B, M, N)   particles per batch (B instances, M agents, N dims)
        score : (B, M, N)   ∇_theta log p(theta) supplied by RL agents

        Standard SVGD update:
            φ_i = (1/M) * [ Σ_j k(θ_j, θ_i) * score_j  +  Σ_j ∇_θ_j k(θ_j, θ_i) ]

        support_mask : (B, support_count) bool — True = particule éligible.
                       Les particules masquées sont exclues des deux termes.
                       La normalisation utilise le compte d'éligibles par instance.
        """
        if support_thetas is not None:
            if support_thetas.dim() < thetas.dim() + 1:
                raise ValueError("support_thetas must add one support dimension to thetas.")
            support_count = support_thetas.size(2)
            if support_count <= 0:
                raise ValueError("support_thetas must contain at least one support particle.")
            K, grad_term = self.kernel(
                thetas,
                probs=probs,
                support_thetas=support_thetas,
                support_probs=support_probs,
            )
            if torch.isnan(K).any() or torch.isinf(K).any():
                K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(grad_term).any() or torch.isinf(grad_term).any():
                grad_term = torch.nan_to_num(grad_term, nan=0.0, posinf=0.0, neginf=0.0)

            if support_mask is not None:
                # support_mask : (B, M) — grad_term non-sommé : (B, l_active, M, N[, D])
                mask_f = support_mask.float().unsqueeze(1)  # (B, 1, M)
                K = K * mask_f
                if score.dim() == 5:
                    grad_term = (grad_term * mask_f.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
                else:
                    grad_term = (grad_term * mask_f.unsqueeze(-1)).sum(dim=2)
                norm = support_mask.float().sum(dim=1).clamp(min=1.0)  # (B,)
                if score.dim() == 5:
                    norm = norm[:, None, None, None]
                else:
                    norm = norm[:, None, None]
            else:
                grad_term = grad_term.sum(dim=2)
                norm = float(support_count)

            if score.dim() == 5:
                score_term = (K.unsqueeze(-1).unsqueeze(-1) * score).sum(dim=2)
            else:
                score_term = (K.unsqueeze(-1) * score).sum(dim=2)
            if self.no_repulsion:
                phi = (score_term / self.gamma) / norm
            else:
                phi = (score_term / self.gamma + grad_term) / norm
            if torch.isnan(phi).any() or torch.isinf(phi).any():
                phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
            self.last_kernel_stats = {
                "avg_kernel_value": float(K.mean().item()),
                "avg_kernel_grad": float(grad_term.mean().item()),
            }
            return phi

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
        if self.no_repulsion:
            phi = (score_term / self.gamma) / M             
        else:
            phi = (score_term / self.gamma + grad_term) / M             



        if torch.isnan(phi).any() or torch.isinf(phi).any():
            phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_kernel_stats = {
            "avg_kernel_value": float(K.mean().item()),
            "avg_kernel_grad": float(grad_term.mean().item()),
        }
        return phi

    def get_last_kernel_stats(self):
        return self.last_kernel_stats or None
