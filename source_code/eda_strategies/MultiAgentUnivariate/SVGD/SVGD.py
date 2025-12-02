import torch


class SVGD:
    def __init__(self, kernel):
        self.kernel = kernel  # ex: RBF()

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

        # Gram matrix: K[b, i, j] = k(θ_i, θ_j)
        K = self.kernel(theta, theta)  # (B, M, M)

        # γ = 1 / (2 σ²) for RBF kernel parameterization k(x, y) = exp(-γ ||x - y||²)
        gamma = self.kernel.last_gamma.to(
            device=theta.device,
            dtype=theta.dtype
        )

        # K_transpose[b, i, j] = k(θ_j, θ_i)
        K_transpose = K.transpose(-2, -1)  # (B, M, M)

        # First SVGD term: Σ_j k(θ_j, θ_i) * score_j
        # matmul: (B, M, M) @ (B, M, N) -> (B, M, N)
        score_term = torch.matmul(K_transpose, score)

        # Second SVGD term: Σ_j ∇_{θ_j} k(θ_j, θ_i)
        # For RBF: ∇_{θ_j} k(θ_j, θ_i) = 2γ k(θ_j, θ_i) (θ_i - θ_j)

        # Build θ_i - θ_j for all i, j:
        # theta[:, :, None, :] -> (B, M, 1, N)  index i
        # theta[:, None, :, :] -> (B, 1, M, N)  index j
        # Result: theta_diff[b, i, j, :] = θ_i - θ_j
        theta_diff = theta[:, :, None, :] - theta[:, None, :, :]  # (B, M, M, N)

        # K_transpose.unsqueeze(-1): (B, M, M, 1)
        # product: (B, M, M, N)  with entries k(θ_j, θ_i) * (θ_i - θ_j)
        # sum over j (dim=2) -> (B, M, N)
        # analytic gradient of RBF kernel = 2γ k(θ_j, θ_i) (θ_i - θ_j)
        # gamma = 1 / (2 σ²) 

        # For the RBF kernel:
        #   k(θ_j, θ_i) = exp(-gamma * ||θ_j - θ_i||^2)
        # and using d/dx e^{f(x)} = e^{f(x)} f'(x), we get
        #
        #   ∇_{θ_j} k(θ_j, θ_i)
        #       = k(θ_j, θ_i) * ∇_{θ_j}(-gamma ||θ_j - θ_i||^2)
        #       = -2 gamma (θ_j - θ_i) k(θ_j, θ_i)
        #       =  2 gamma (θ_i - θ_j) k(θ_j, θ_i).
        grad_term = 2.0 * gamma.view(1, 1, 1) * torch.sum(
            K_transpose.unsqueeze(-1) * theta_diff,
            dim=2  # sum over j
        )

        # Average over M particles
        phi = (score_term/10 + grad_term) / M  # (B, M, N)
        return phi
