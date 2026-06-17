import numpy as np

from eda.optimizer.eda_base import EDABase


class PPBIL(EDABase):
    """
    Parallel (multi-population) PBIL with two independent probability vectors.

    Each PV is updated from its own sub-population's best solution.
    The sub-population sizes are adapted each generation: the winning PV's
    share grows by delta=round(lr*lam), clamped to [pop_min, pop_max].
    Paper defaults: lr=0.1, pop_min=0.4*lam, pop_max=0.6*lam.
    """
    def __init__(self, categories, lr=0.1, lam=64,
                 mut_prob=0.02, mut_shift=0.05, theta_init=None):
        super().__init__(categories, lam=lam, theta_init=theta_init)
        assert self.Cmax == 2
        assert 0.0 < lr < 1.0
        assert 0.0 <= mut_prob <= 1.0

        self.lr = lr
        self.mut_prob = mut_prob
        self.mut_shift = mut_shift

        # PV1 is self.theta (from EDABase, initialised to 0.5).
        # PV2 is a separate copy also initialised to 0.5.
        self.theta2 = self.theta.copy()

        # Population split, kept within [pop_min, pop_max]
        self.pop_min = max(1, round(0.4 * lam))
        self.pop_max = lam - self.pop_min          # == round(0.6 * lam)
        self.delta   = max(1, round(lr * lam))
        self.pop1    = lam // 2
        self.pop2    = lam - self.pop1

        # Counter used by sampling() to route calls to the right PV
        self._sample_count = 0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sampling(self):
        """
        Draw from PV1 for the first pop1 calls per generation,
        then from PV2 for the remaining pop2 calls.
        The counter resets automatically after lam total calls.
        """
        theta = self.theta if self._sample_count < self.pop1 else self.theta2
        self._sample_count += 1
        if self._sample_count >= self.lam:
            self._sample_count = 0

        rand = np.random.rand(self.d, 1)
        cum = theta.cumsum(axis=1)
        return (cum - theta <= rand) & (rand < cum)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, x, evals, range_restriction=False):
        x     = np.array(x)
        evals = np.array(evals)
        self.num_evals += x.shape[0]

        # Split according to the population sizes used for this generation
        x1, ev1 = x[:self.pop1],  evals[:self.pop1]
        x2, ev2 = x[self.pop1:],  evals[self.pop1:]

        has1 = len(ev1) > 0
        has2 = len(ev2) > 0

        if has1:
            i1 = np.argmin(ev1);  best1, bev1 = x1[i1], ev1[i1]
            if self.best_eval > bev1:
                self.best_eval, self.best_indiv = bev1, best1
            self.theta[:, -1] = (1.0 - self.lr) * self.theta[:, -1] + self.lr * best1[:, -1]

        if has2:
            i2 = np.argmin(ev2);  best2, bev2 = x2[i2], ev2[i2]
            if self.best_eval > bev2:
                self.best_eval, self.best_indiv = bev2, best2
            self.theta2[:, -1] = (1.0 - self.lr) * self.theta2[:, -1] + self.lr * best2[:, -1]

        # Adapt population sizes only when both sub-pops were evaluated
        if has1 and has2:
            if bev1 < bev2:
                self.pop1 = min(self.pop1 + self.delta, self.pop_max)
            elif bev2 < bev1:
                self.pop1 = max(self.pop1 - self.delta, self.pop_min)
            self.pop2 = self.lam - self.pop1

        # Mutate both PVs (forgetting factor toward 0.5)
        for theta in (self.theta, self.theta2):
            mut_idx = np.random.rand(self.d) < self.mut_prob
            mut_num = mut_idx.sum()
            if mut_num:
                theta[mut_idx, -1] = (1.0 - self.mut_shift) * theta[mut_idx, -1] \
                                     + np.random.randint(0, 2, mut_num) * self.mut_shift

        # Keep the two columns consistent (binary: p0 = 1 - p1)
        self.theta[:, 0]  = 1.0 - self.theta[:, -1]
        self.theta2[:, 0] = 1.0 - self.theta2[:, -1]

        # Clip PV1 via the base-class helper, then clip PV2 the same way
        self.clipping(range_restriction)
        self.theta, self.theta2 = self.theta2, self.theta
        self.clipping(range_restriction)
        self.theta, self.theta2 = self.theta2, self.theta

    # ------------------------------------------------------------------
    # Convergence (average of both PVs)
    # ------------------------------------------------------------------

    def convergence(self):
        conv1 = self.theta.max(axis=1).mean()
        conv2 = self.theta2.max(axis=1).mean()
        return (conv1 + conv2) / 2.0

    # ------------------------------------------------------------------

    def __str__(self):
        sup_str = "    " + super().__str__().replace("\n", "\n    ")
        return (
            'PPBIL(\n'
            '{}\n'
            '    lr: {}\n'
            '    mutation prob: {}\n'
            '    mutation shift: {}\n'
            '    delta: {}\n'
            '    pop_min/pop_max: {}/{}\n'
            '    current pop1/pop2: {}/{}\n'
            ')'
        ).format(sup_str, self.lr, self.mut_prob, self.mut_shift,
                 self.delta, self.pop_min, self.pop_max, self.pop1, self.pop2)


class PPBILGpu:
    """
    GPU-batched PPBIL: B = nb_instances * nb_restarts runs in parallel.

    Binary only (Cmax=2). Two PVs with adaptive split, PBIL-style EMA update
    and probabilistic mutation toward 0.5.  The split adaptation is per-batch-item.

    Sampling pads both sub-populations to pop_max so that the operation is
    fully vectorised; invalid slots are masked to -inf before taking the best.
    """

    def __init__(self, B, d, lr=0.1, lam=64, mut_prob=0.02, mut_shift=0.05,
                 device='cuda'):
        import torch
        self._torch    = torch
        self.B         = B
        self.d         = d
        self.lr        = lr
        self.lam       = lam
        self.mut_prob  = mut_prob
        self.mut_shift = mut_shift
        self.device    = torch.device(device)

        self.pop_min = max(1, round(0.4 * lam))
        self.pop_max = lam - self.pop_min
        self.delta   = max(1, round(lr * lam))

        # Adaptive split per batch item: (B,) int64
        self.pop1 = torch.full((B,), lam // 2, dtype=torch.int64, device=self.device)
        self.pop2 = lam - self.pop1

        # PV1 and PV2: (B, d, 2)  — column 1 = p(category = 1)
        self.theta1 = torch.full((B, d, 2), 0.5, dtype=torch.float32, device=self.device)
        self.theta2 = torch.full((B, d, 2), 0.5, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_pv(self, theta, n):
        """Sample n individuals from PV theta (B, d, 2) → (B, n, d) int64."""
        torch = self._torch
        B, d, C = theta.shape
        theta_flat = theta.reshape(B * d, C)
        samples = torch.multinomial(theta_flat, n, replacement=True)  # (B*d, n)
        return samples.reshape(B, d, n).permute(0, 2, 1).contiguous()  # (B, n, d)

    def sample(self):
        """
        Returns (pop1_samples, pop2_samples): each (B, pop_max, d) int64.
        Valid slots for batch item b: [0, pop1[b]) and [0, pop2[b]) respectively.
        """
        return (
            self._sample_pv(self.theta1, self.pop_max),
            self._sample_pv(self.theta2, self.pop_max),
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, pop1_samples, score1, pop2_samples, score2):
        """
        Parameters
        ----------
        pop1_samples : (B, pop_max, d) int64  — samples from PV1
        score1       : (B, pop_max) float32   — scores to maximise
        pop2_samples : (B, pop_max, d) int64  — samples from PV2
        score2       : (B, pop_max) float32   — scores to maximise
        """
        torch    = self._torch
        arange_B = torch.arange(self.B, device=self.device)
        pos      = torch.arange(self.pop_max, device=self.device).unsqueeze(0)  # (1, pop_max)

        # Mask slots beyond the actual split size
        mask1 = pos < self.pop1.unsqueeze(1)  # (B, pop_max)
        mask2 = pos < self.pop2.unsqueeze(1)
        s1 = score1.masked_fill(~mask1, float('-inf'))
        s2 = score2.masked_fill(~mask2, float('-inf'))

        best_idx1 = s1.argmax(dim=1)                               # (B,)
        best_idx2 = s2.argmax(dim=1)
        best_sol1 = pop1_samples[arange_B, best_idx1, :].float()   # (B, d) ∈ {0., 1.}
        best_sol2 = pop2_samples[arange_B, best_idx2, :].float()
        best_score1 = s1[arange_B, best_idx1]                      # (B,)
        best_score2 = s2[arange_B, best_idx2]

        # PBIL EMA update on p(category = 1)
        lr = self.lr
        self.theta1[:, :, 1] = (1.0 - lr) * self.theta1[:, :, 1] + lr * best_sol1
        self.theta2[:, :, 1] = (1.0 - lr) * self.theta2[:, :, 1] + lr * best_sol2

        # Mutation: probabilistic shift toward {0, 1} (forgetting factor toward 0.5)
        for theta in (self.theta1, self.theta2):
            mut_mask = torch.rand(self.B, self.d, device=self.device) < self.mut_prob
            mut_vals = torch.randint(0, 2, (self.B, self.d), device=self.device,
                                     dtype=torch.float32)
            theta[:, :, 1] = torch.where(
                mut_mask,
                (1.0 - self.mut_shift) * theta[:, :, 1] + mut_vals * self.mut_shift,
                theta[:, :, 1],
            )

        # Keep columns consistent and clamp
        for theta in (self.theta1, self.theta2):
            theta[:, :, 1].clamp_(0.0, 1.0)
            theta[:, :, 0] = 1.0 - theta[:, :, 1]

        # Adapt split per batch item
        pv1_wins = (best_score1 > best_score2).long()
        pv2_wins = (best_score2 > best_score1).long()
        self.pop1 = (
            self.pop1 + pv1_wins * self.delta - pv2_wins * self.delta
        ).clamp(self.pop_min, self.pop_max)
        self.pop2 = self.lam - self.pop1
