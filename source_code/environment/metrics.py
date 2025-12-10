import torch
from torch.distributions import Bernoulli, kl_divergence


class MetricsCalculator:
    """Utility class responsible for computing cross-agent metrics."""

    def __init__(self, normalization_factor: float = 1.0):
        self.normalization_factor = normalization_factor

    @staticmethod
    def agent_theta_tensor(agent):
        return agent.theta if hasattr(agent, "theta") else agent

    def compute_average_kl(self, agents):
        if agents is None or len(agents) < 2:
            return 0.0

        eps = 1e-8
        total_pairwise_kl = 0.0
        comparisons = 0
        with torch.no_grad():
            agent_probs = [torch.sigmoid(self.agent_theta_tensor(agent)).detach() for agent in agents]

        for i in range(len(agent_probs)):
            for j in range(i + 1, len(agent_probs)):
                p = torch.clamp(agent_probs[i], eps, 1 - eps)
                q = torch.clamp(agent_probs[j], eps, 1 - eps)
                dist_p = Bernoulli(probs=p)
                dist_q = Bernoulli(probs=q)
                kl_pq_inst = kl_divergence(dist_p, dist_q).mean(dim=1)
                kl_qp_inst = kl_divergence(dist_q, dist_p).mean(dim=1)
                kl_pair_inst = 0.5 * (kl_pq_inst + kl_qp_inst)
                total_pairwise_kl += kl_pair_inst.mean().item()
                comparisons += 1

        return (total_pairwise_kl / comparisons) if comparisons > 0 else 0.0

    def compute_average_hamming(self, agents):
        """Compute pairwise theoretical Hamming diversity from Bernoulli policies."""
        if agents is None or len(agents) < 2:
            return 0.0

        M = len(agents)
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(self.agent_theta_tensor(agent)).detach() for agent in agents], dim=0)

        theta_i = probs.unsqueeze(1)  # (M, 1, B, N)
        theta_j = probs.unsqueeze(0)  # (1, M, B, N)
        pairwise = theta_i + theta_j - 2 * theta_i * theta_j  # (M, M, B, N)
        distances = pairwise.sum(dim=-1).mean(dim=-1)  # (M, M)

        off_diag_sum = distances.sum() - torch.diagonal(distances).sum()
        num_pairs = M * (M - 1)
        return (off_diag_sum / num_pairs).item() if num_pairs > 0 else 0.0

    def compute_fitness(self, scores: torch.Tensor):
        value = torch.mean(scores).item()
        return value / self.normalization_factor
