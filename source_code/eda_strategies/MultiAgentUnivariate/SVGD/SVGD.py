import torch


class SVGD:
    def __init__(self, kernel):
        # kernel is typically an RBF; it provides k(x, y) and exposes last_gamma.
        self.kernel = kernel

    def phi(self, theta, score):
        """
        theta : (B, M, N)   particles per batch (B instances, M agents, N dims)
        score : (B, M, N)   ∇_theta log p(theta) supplied by RL agents

        Implements standard SVGD update:
            φ_i = (1/M) * [ Σ_j k(θ_j, θ_i) * score_j  +  Σ_j ∇_θ_j k(θ_j, θ_i) ]
        """
        theta_req = theta.detach().clone().requires_grad_(True)

        # Gram matrix of kernel evaluations k(θ_b_i, θ_b_j) for each batch b.
        K = self.kernel(theta_req, theta_req)  # (B, M, M)

        # Reorder axes so matmul matches Σ_j k_j,i * score_j.
        K_transpose = K.transpose(-2, -1)  # (B, M_dest, M_src)

        # First SVGD term: (Kᵗ @ score) -> B × M × N.
        score_term = torch.matmul(K_transpose, score)

        # Second SVGD term computed via autograd:
        #   grad_term[b, i, :] = Σ_j ∇_{θ_j} k(θ_j, θ_i).
        grad_batches = []
        for b in range(theta_req.size(0)):
            grads_per_i = []
            for i in range(theta_req.size(1)):
                grad_outputs = torch.zeros_like(K_transpose[b])
                grad_outputs[i, :].fill_(1.0)
                grad_theta = torch.autograd.grad(
                    K_transpose[b],
                    theta_req[b],
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad_theta is None:
                    grad_theta = torch.zeros_like(theta_req[b])
                grads_per_i.append(grad_theta.sum(dim=0))
            grad_batches.append(torch.stack(grads_per_i, dim=0))
        grad_term = torch.stack(grad_batches, dim=0).detach()

        # Average over M particles to obtain φ.
        phi = (score_term + grad_term) / theta.size(1)

        return phi
