import torch


class SVGD:
    def __init__(self, kernel):
        self.kernel = kernel  # ex: RBF()

    def phi(self, theta, score):
        """
        theta : (M, N) ou (B, M, N)
        score : (M, N) ou (B, M, N)
        """
        squeeze_batch = False
        if theta.dim() == 2:
            theta = theta.unsqueeze(0)
            score = score.unsqueeze(0)
            squeeze_batch = True

        K = self.kernel(theta, theta)  # (B, M, M)
        if K.dim() == 2:
            K = K.unsqueeze(0)

        gamma = getattr(self.kernel, "last_gamma", None)
        if gamma is None:
            gamma = torch.tensor(1.0, device=theta.device, dtype=theta.dtype)
        elif not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma, device=theta.device, dtype=theta.dtype)
        else:
            gamma = gamma.to(device=theta.device, dtype=theta.dtype)

        K_transpose = K.transpose(-2, -1)

        score_term = torch.matmul(K_transpose, score)

        theta_diff = theta[:, None, :, :] - theta[:, :, None, :]
        grad_term = -2.0 * gamma.view(1) * torch.sum(K_transpose.unsqueeze(-1) * theta_diff, dim=2)

        phi = (score_term + grad_term) / theta.size(1)

        if squeeze_batch:
            return phi.squeeze(0)
        return phi
