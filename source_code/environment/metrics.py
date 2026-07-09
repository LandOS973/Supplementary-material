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

        def _clamp(x):
            return torch.clamp(x, eps, 1 - eps)

        with torch.no_grad():
            theta = torch.stack([self.agent_theta_tensor(agent).detach() for agent in agents], dim=0)
            theta_i = theta.unsqueeze(1)
            theta_j = theta.unsqueeze(0)

            if theta.dim() == 4:  # catégoriel : (M, B, N, D)
                p = _clamp(torch.softmax(theta_i, dim=-1))
                q = _clamp(torch.softmax(theta_j, dim=-1))
                m = _clamp(0.5 * (p + q))

                def _kl(a, b):
                    a = _clamp(a)
                    b = _clamp(b)
                    val = (a * (torch.log(a) - torch.log(b))).sum(dim=-1)          # somme sur D -> (M, M, B, N)
                    return torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

                js_per_var = 0.5 * (_kl(p, m) + _kl(q, m))                          # (M, M, B, N)
                js_total = js_per_var.sum(dim=-1)                                    # somme sur N -> (M, M, B)
            else:  # binaire : (M, B, N)
                p = _clamp(torch.sigmoid(theta_i))
                q = _clamp(torch.sigmoid(theta_j))
                m = _clamp(0.5 * (p + q))

                def _kl(a, b):
                    a = _clamp(a)
                    b = _clamp(b)
                    val = (a * (torch.log(a) - torch.log(b)) + (1 - a) * (torch.log1p(-a) - torch.log1p(-b))).sum(dim=-1)
                    return torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

                js_total = 0.5 * (_kl(p, m) + _kl(q, m))                            # déjà sommé sur N -> (M, M, B)

            pairwise_mean = js_total.mean(dim=-1)                                    # moyenne sur B -> (M, M)

        num_agents = theta.shape[0]
        total = pairwise_mean.sum() - torch.diagonal(pairwise_mean).sum()
        num_pairs = num_agents * (num_agents - 1)
        avg = (total / num_pairs).item() if num_pairs > 0 else 0.0
        return avg, pairwise_mean.cpu().numpy()

    def compute_average_hamming(self, agents):
        """Compute pairwise theoretical Hamming diversity from Bernoulli/Categorical policies."""
        if agents is None or len(agents) < 2:
            return 0.0, None

        M = len(agents)
        with torch.no_grad():
            theta = torch.stack([self.agent_theta_tensor(agent).detach() for agent in agents], dim=0)
            theta_i = theta.unsqueeze(1)
            theta_j = theta.unsqueeze(0)
            if theta.dim() == 4:  # catégoriel : (M, B, N, D)
                probs_i = torch.softmax(theta_i, dim=-1)
                probs_j = torch.softmax(theta_j, dim=-1)
                pairwise = 1.0 - (probs_i * probs_j).sum(dim=-1)          # (M, M, B, N) — P(différent) par variable
            else:  # binaire : (M, B, N)
                probs_i = torch.sigmoid(theta_i)
                probs_j = torch.sigmoid(theta_j)
                pairwise = probs_i + probs_j - 2 * probs_i * probs_j       # (M, M, B, N)
            distances = pairwise.sum(dim=-1).mean(dim=-1)                  # somme sur N, moyenne sur B -> (M, M)

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
                p = torch.softmax(theta, dim=-1)
                p = torch.nan_to_num(p, nan=1.0 / float(p.size(-1)))
                p = torch.clamp(p, eps, 1.0 - eps)
                p = p / p.sum(dim=-1, keepdim=True)
                ent = -(p * torch.log(p)).sum(dim=-1)           
            else:
                p = torch.sigmoid(theta)
                p = torch.nan_to_num(p, nan=0.5, posinf=1.0 - eps, neginf=eps)
                p = torch.clamp(p, eps, 1 - eps)
                ent = -(p * torch.log(p) + (1 - p) * torch.log1p(-p))           
            if torch.isnan(ent).any() or torch.isinf(ent).any():
                ent = torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)
            ent = ent.sum(dim=-1).mean(dim=-1)        

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
            theta = torch.stack([self.agent_theta_tensor(agent).detach() for agent in agents], dim=0)
            probs = torch.softmax(theta, dim=-1) if theta.dim() == 4 else torch.sigmoid(theta)

            theta_i = probs.unsqueeze(1)
            theta_j = probs.unsqueeze(0)
            diff = theta_i - theta_j
            per_elem = torch.sqrt(torch.sum(diff * diff, dim=-1))           # somme sur D (catégoriel) ou N (binaire)
            if theta.dim() == 4:  # catégoriel : per_elem est (M, M, B, N) -> il reste N à agréger
                pairwise = per_elem.sum(dim=-1)                              # somme sur N -> (M, M, B)
            else:
                pairwise = per_elem                                          # déjà (M, M, B)
            pairwise_mean = pairwise.mean(dim=-1)                            # moyenne sur B -> (M, M)

        num_agents = theta.shape[0]
        total = pairwise_mean.sum() - torch.diagonal(pairwise_mean).sum()
        num_pairs = num_agents * (num_agents - 1)
        avg = (total / num_pairs).item() if num_pairs > 0 else 0.0
        return avg, pairwise_mean.cpu().numpy()

    def compute_l1_distance(self, agents):
        if agents is None or len(agents) < 2:
            return 0.0, None

        with torch.no_grad():
            theta = torch.stack([self.agent_theta_tensor(agent).detach() for agent in agents], dim=0)
            probs = torch.softmax(theta, dim=-1) if theta.dim() == 4 else torch.sigmoid(theta)

            theta_i = probs.unsqueeze(1)
            theta_j = probs.unsqueeze(0)
            diff = torch.abs(theta_i - theta_j)
            per_elem = torch.sum(diff, dim=-1)                               # somme sur D (catégoriel) ou N (binaire)
            if theta.dim() == 4:  # catégoriel : per_elem est (M, M, B, N) -> il reste N à agréger
                pairwise = per_elem.sum(dim=-1)                              # somme sur N -> (M, M, B)
            else:
                pairwise = per_elem                                          # déjà (M, M, B)
            pairwise_mean = pairwise.mean(dim=-1)                            # moyenne sur B -> (M, M)

        num_agents = theta.shape[0]
        total = pairwise_mean.sum() - torch.diagonal(pairwise_mean).sum()
        num_pairs = num_agents * (num_agents - 1)
        avg = (total / num_pairs).item() if num_pairs > 0 else 0.0
        return avg, pairwise_mean.cpu().numpy()
