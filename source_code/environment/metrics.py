import torch


class MetricsCalculator:
    """Utility class responsible for computing cross-agent metrics."""

    def __init__(self, normalization_factor: float = 1.0):
        self.normalization_factor = normalization_factor

    @staticmethod
    def agent_theta_tensor(agent):
        return agent.theta if hasattr(agent, "theta") else agent

    def compute_average_js(self, agents):
        if agents is None or len(agents) < 2:
            return 0.0, None

        eps = 1e-6
        with torch.no_grad():
            probs = torch.stack(
                [torch.sigmoid(self.agent_theta_tensor(agent)).detach() for agent in agents],
                dim=0
            )
            probs = torch.clamp(probs, eps, 1 - eps)

        def _clamp(x):
            return torch.clamp(x, eps, 1 - eps)

        p = _clamp(probs.unsqueeze(1))  # (M,1,B,N)
        q = _clamp(probs.unsqueeze(0))  # (1,M,B,N)
        m = _clamp(0.5 * (p + q))

        def _kl(a, b):
            a = _clamp(a)
            b = _clamp(b)
            val = (a * (torch.log(a) - torch.log(b)) + (1 - a) * (torch.log1p(-a) - torch.log1p(-b))).sum(dim=-1)
            if torch.isnan(val).any() or torch.isinf(val).any():
                val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
            return val

        js = 0.5 * (_kl(p, m) + _kl(q, m))  # (M,M,B)
        if torch.isnan(js).any() or torch.isinf(js).any():
            js = torch.nan_to_num(js, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise_mean = js.mean(dim=-1)  # (M,M)

        num_agents = probs.shape[0]
        total = pairwise_mean.sum() - torch.diagonal(pairwise_mean).sum()
        num_pairs = num_agents * (num_agents - 1)
        avg = (total / num_pairs).item() if num_pairs > 0 else 0.0
        return avg, pairwise_mean.cpu().numpy()

    def compute_average_hamming(self, agents):
        """Compute pairwise theoretical Hamming diversity from Bernoulli policies."""
        if agents is None or len(agents) < 2:
            return 0.0, None

        M = len(agents)
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(self.agent_theta_tensor(agent)).detach() for agent in agents], dim=0)

        theta_i = probs.unsqueeze(1)  # (M, 1, B, N)
        theta_j = probs.unsqueeze(0)  # (1, M, B, N)
        pairwise = theta_i + theta_j - 2 * theta_i * theta_j  # (M, M, B, N)
        distances = pairwise.sum(dim=-1).mean(dim=-1)  # (M, M)

        off_diag_sum = distances.sum() - torch.diagonal(distances).sum()
        num_pairs = M * (M - 1)
        avg = (off_diag_sum / num_pairs).item() if num_pairs > 0 else 0.0
        return avg, distances.detach().cpu().numpy()

    def compute_entropy(self, agents):
        if agents is None or len(agents) == 0:
            return 0.0, None

        eps = 1e-8
        with torch.no_grad():
            theta = torch.stack([self.agent_theta_tensor(agent).detach() for agent in agents], dim=0)
            if theta.dim() == 4:
                # Categorical entropy: -sum_c p log p per variable
                p = torch.softmax(theta, dim=-1)
                p = torch.nan_to_num(p, nan=1.0 / float(p.size(-1)))
                p = torch.clamp(p, eps, 1.0 - eps)
                p = p / p.sum(dim=-1, keepdim=True)
                ent = -(p * torch.log(p)).sum(dim=-1)  # (M,B,N)
            else:
                # Bernoulli entropy
                p = torch.sigmoid(theta)
                p = torch.nan_to_num(p, nan=0.5, posinf=1.0 - eps, neginf=eps)
                p = torch.clamp(p, eps, 1 - eps)
                ent = -(p * torch.log(p) + (1 - p) * torch.log1p(-p))  # (M,B,N)
            if torch.isnan(ent).any() or torch.isinf(ent).any():
                ent = torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)
            ent = ent.sum(dim=-1).mean(dim=-1)  # (M,)

        entropies = ent.cpu().tolist()
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        return avg_entropy, entropies

    def compute_fitness(self, scores: torch.Tensor):
        value = torch.mean(scores).item()
        return value / self.normalization_factor

    def compute_l2_distance(self, agents):
        if agents is None or len(agents) < 2:
            return 0.0, None

        with torch.no_grad():
            theta = torch.stack([torch.sigmoid(self.agent_theta_tensor(agent).detach()) for agent in agents], dim=0)  # (M, B, N)

        theta_i = theta.unsqueeze(1)  # (M,1,B,N)
        theta_j = theta.unsqueeze(0)  # (1,M,B,N)
        diff = theta_i - theta_j  # (M,M,B,N)
        pairwise = torch.sqrt(torch.sum(diff * diff, dim=-1))  # (M,M,B)
        pairwise_mean = pairwise.mean(dim=-1)  # (M,M)

        num_agents = theta.shape[0]
        total = pairwise_mean.sum() - torch.diagonal(pairwise_mean).sum()
        num_pairs = num_agents * (num_agents - 1)
        avg = (total / num_pairs).item() if num_pairs > 0 else 0.0
        return avg, pairwise_mean.cpu().numpy()

    def compute_l1_distance(self, agents):
        if agents is None or len(agents) < 2:
            return 0.0, None

        with torch.no_grad():
            theta = torch.stack([torch.sigmoid(self.agent_theta_tensor(agent).detach()) for agent in agents], dim=0)  # (M, B, N)

        theta_i = theta.unsqueeze(1)  # (M,1,B,N)
        theta_j = theta.unsqueeze(0)  # (1,M,B,N)
        diff = torch.abs(theta_i - theta_j)  # (M,M,B,N)
        pairwise = torch.sum(diff, dim=-1)  # (M,M,B)
        pairwise_mean = pairwise.mean(dim=-1)  # (M,M)

        num_agents = theta.shape[0]
        total = pairwise_mean.sum() - torch.diagonal(pairwise_mean).sum() 
        num_pairs = num_agents * (num_agents - 1)
        avg = (total / num_pairs).item() if num_pairs > 0 else 0.0
        return avg, pairwise_mean.cpu().numpy()
