import torch


class SVGD:
    def __init__(self, kernel, alpha=10.0):
        self.kernel = kernel  # ex: RBF()
        if alpha == 0:
            raise ValueError("alpha must be non-zero.")
        self.alpha = float(alpha)

    def phi(self, theta, score):
        """
        B => Nombre d'instances
        N => Nombre de variables
        M => Nombre de particules (agents)
        theta : (B, M, N)   particles per batch (B instances, M agents, N dims)
        score : (B, M, N)   ∇_theta log p(theta) supplied by RL agents

        Standard SVGD update:
            φ_i = (1/M) * [ Σ_j k(θ_j, θ_i) * score_j  +  Σ_j ∇_θ_j k(θ_j, θ_i) ]
        """
        B, M, N = theta.shape

        # Gram matrix & gradients produced by the kernel itself
        # K[b, i, j] = k(θ_i, θ_j)
        # grad_first[b, i, j, :] = ∇_{θ_i} k(θ_i, θ_j)
        K, grad_first = self.kernel(theta, theta)  # (B, M, M), (B, M, M, N)
        if torch.isnan(K).any() or torch.isinf(K).any():
            K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(grad_first).any() or torch.isinf(grad_first).any():
            grad_first = torch.nan_to_num(grad_first, nan=0.0, posinf=0.0, neginf=0.0)

        # K_transpose[b, i, j] = k(θ_j, θ_i)
        K_transpose = K.transpose(-2, -1)  # (B, M, M)

        # First SVGD term: Σ_j k(θ_j, θ_i) * score_j
        # matmul: (B, M, M) @ (B, M, N) -> (B, M, N)
        score_term = torch.matmul(K_transpose, score)
        # Second SVGD term: Σ_j ∇_{θ_j} k(θ_j, θ_i)
        grad_term = torch.sum(
            grad_first.transpose(-3, -2),
            dim=2  # sum over j
        )

        # Average over M particles
        phi = (score_term / self.alpha + grad_term) / M  # (B, M, N)
        if torch.isnan(phi).any() or torch.isinf(phi).any():
            phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        return phi
