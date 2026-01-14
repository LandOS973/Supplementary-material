import torch


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = int(buffer_size)  # taille max par instance
        self.batch_size = int(batch_size)  # nb d'anciens echantillons tires par instance
        self.samples = None  # (B, buffer_size, N) solutions stockees
        self.scores = None  # (B, buffer_size) fitness stockees
        self.logps = None  # (B, buffer_size) log-proba au moment du tirage
        self.counts = None  # (B,) nb d'elements valides stockes
        self.pos = None  # (B,) curseur d'ecriture du ring-buffer
        self.nb_instances = 0  # B
        self.N = 0  # nb de variables par solution

    def reset(self, nb_instances, N, device, dtype):
        self.nb_instances = int(nb_instances)
        self.N = int(N)
        if self.buffer_size <= 0 or self.batch_size <= 0:
            self.samples = None
            self.scores = None
            self.logps = None
            self.counts = None
            self.pos = None
            return
        # Preallocation du ring-buffer par instance sur le device cible.
        self.samples = torch.empty(
            (nb_instances, self.buffer_size, N),
            device=device,
            dtype=dtype,
        )
        self.scores = torch.empty(
            (nb_instances, self.buffer_size),
            device=device,
            dtype=dtype,
        )
        self.logps = torch.empty(
            (nb_instances, self.buffer_size),
            device=device,
            dtype=dtype,
        )
        self.counts = torch.zeros(nb_instances, device=device, dtype=torch.long)
        self.pos = torch.zeros(nb_instances, device=device, dtype=torch.long)

    def sample(self, num_agents):
        if self.samples is None:
            return None, None, None
        if self.counts.min().item() == 0:
            return None, None, None

        K = self.batch_size
        B = self.nb_instances
        device = self.samples.device

        counts = self.counts.to(device=device, dtype=torch.float32)
        indices = (torch.rand((B, K), device=device) * counts.unsqueeze(1)).long()
        batch_idx = torch.arange(B, device=device).unsqueeze(1)

        samples = self.samples[batch_idx, indices]
        scores = self.scores[batch_idx, indices]
        logps = self.logps[batch_idx, indices]

        # Partage du meme batch replay entre agents pour une instance donnee.
        samples = samples.unsqueeze(1).expand(-1, num_agents, -1, -1)
        scores = scores.unsqueeze(1).expand(-1, num_agents, -1)
        logps = logps.unsqueeze(1).expand(-1, num_agents, -1)

        return samples, scores, logps # Return tensors of shape (B, M, K, N), (B, M, K), (B, M, K)

    def add(self, indivduals, fitness, log_probs):
        if self.samples is None:
            return
        B, M, L, N = indivduals.shape
        device = self.samples.device

        # Aplatit agents + echantillons en un seul flux par instance.
        samples = indivduals.reshape(B, -1, N).to(device=device, dtype=self.samples.dtype)
        scores = fitness.reshape(B, -1).to(device=device, dtype=self.scores.dtype)
        logps = log_probs.reshape(B, -1).to(device=device, dtype=self.logps.dtype)

        num_new = samples.shape[1]
        if num_new == 0:
            return
        if num_new > self.buffer_size:
            samples = samples[:, -self.buffer_size :, :]
            scores = scores[:, -self.buffer_size :]
            logps = logps[:, -self.buffer_size :]
            num_new = self.buffer_size

        positions = (self.pos.unsqueeze(1) + torch.arange(num_new, device=device)) % self.buffer_size
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, num_new)

        self.samples[batch_idx, positions] = samples
        self.scores[batch_idx, positions] = scores
        self.logps[batch_idx, positions] = logps

        self.counts = torch.minimum(
            self.counts + num_new,
            torch.full_like(self.counts, self.buffer_size),
        )
        self.pos = (self.pos + num_new) % self.buffer_size
