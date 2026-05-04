import numpy as np


class PEDA:
    """
    Parallel Island-based EDA (PEDA) with UMDA-style update and MDR migration.

    Each island maintains an independent probability vector (PV).
    Every `epo` generations, the best individual of each island migrates
    to each of its neighbours in the MDR torus topology (Best/Random strategy:
    the immigrant overwrites a randomly chosen slot in the destination island).
    Migration snapshot is taken before any replacement so that all islands use
    the pre-migration bests simultaneously.

    PV update: direct frequency count on the truncated elite
        PV_i[j] = mean(elite[:, j])   (no learning-rate smoothing)

    Paper defaults: lam=1280, sub_num=8, p_select=0.7, epo=4
    """

    def __init__(self, categories, lam=1280, sub_num=8, p_select=0.7, epo=4):
        assert lam % sub_num == 0, "lam must be divisible by sub_num"

        self.d            = len(categories)
        self.C            = categories
        self.Cmax         = int(np.max(categories))
        self.lam          = lam
        self.sub_num      = sub_num
        self.sub_pop_size = lam // sub_num
        self.p_select     = p_select
        self.n_select     = max(1, int(self.sub_pop_size * p_select))
        self.epo          = epo

        # Each island: PV of shape (d, Cmax), initialised to uniform 1/Cmax
        self.thetas = np.full((sub_num, self.d, self.Cmax), 1.0 / self.Cmax)

        # MDR torus topology — arrange sub_num islands in the most square grid
        rows = int(np.sqrt(sub_num))
        while sub_num % rows != 0:
            rows -= 1
        self._grid_rows = rows
        self._grid_cols = sub_num // rows
        # Pre-compute neighbour lists (deduplicated, self excluded)
        self._neighbours = [self._compute_neighbours(i) for i in range(sub_num)]

        self.generation = 0
        self.best_eval  = np.inf   # minimum cost seen (cost = -fitness)
        self.best_indiv = None
        self.num_evals  = 0

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _compute_neighbours(self, i):
        r, c = divmod(i, self._grid_cols)
        nbrs = {
            ((r - 1) % self._grid_rows) * self._grid_cols + c,
            ((r + 1) % self._grid_rows) * self._grid_cols + c,
            r * self._grid_cols + (c - 1) % self._grid_cols,
            r * self._grid_cols + (c + 1) % self._grid_cols,
        }
        nbrs.discard(i)
        return list(nbrs)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_island(self, island_idx):
        """
        Draw one individual from island `island_idx`.
        Returns a (d, Cmax) boolean one-hot array (same format as EDABase).
        """
        theta = self.thetas[island_idx]          # (d, Cmax)
        rand  = np.random.rand(self.d, 1)
        cum   = theta.cumsum(axis=1)
        return (cum - theta <= rand) & (rand < cum)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, populations, evals_list):
        """
        Parameters
        ----------
        populations : list of sub_num arrays, each (n_i, d, 2)
            Sampled individuals per island (n_i may be < sub_pop_size on last batch).
        evals_list : list of sub_num arrays, each (n_i,)
            Costs to minimise (pass -fitness for maximisation problems).
        """
        self.generation += 1
        self.num_evals  += sum(len(e) for e in evals_list)

        # --- Track global best (pre-migration state) ---
        for i in range(self.sub_num):
            if len(evals_list[i]) == 0:
                continue
            bi = int(np.argmin(evals_list[i]))
            if self.best_eval > evals_list[i][bi]:
                self.best_eval  = evals_list[i][bi]
                self.best_indiv = populations[i][bi].copy()

        # --- Migration every epo generations (Best / Random, MDR) ---
        if self.generation % self.epo == 0:
            # Snapshot best individual AND its cost BEFORE modifying any population
            bests = []
            for i in range(self.sub_num):
                if len(evals_list[i]) == 0:
                    bests.append(None)
                else:
                    bi = int(np.argmin(evals_list[i]))
                    bests.append((populations[i][bi].copy(), evals_list[i][bi]))

            # Apply migrations simultaneously
            for i, snapshot in enumerate(bests):
                if snapshot is None:
                    continue
                best_indiv, best_cost = snapshot
                for j in self._neighbours[i]:
                    n_j = len(populations[j])
                    if n_j == 0:
                        continue
                    rand_slot = np.random.randint(0, n_j)
                    populations[j][rand_slot]  = best_indiv
                    evals_list[j][rand_slot]   = best_cost

        # --- Selection + UMDA frequency update per island ---
        for i in range(self.sub_num):
            pop  = populations[i]   # (n_i, d, 2)
            fits = evals_list[i]    # (n_i,)
            if len(fits) == 0:
                continue

            n_sel = min(self.n_select, len(fits))
            idx   = np.argsort(fits)
            elite = pop[idx[:n_sel]]              # (n_sel, d, 2)

            # PV = frequency of each category in elite (direct count, no lr smoothing)
            for c in range(self.Cmax):
                self.thetas[i, :, c] = elite[:, :, c].mean(axis=0)

    # ------------------------------------------------------------------
    # Convergence helpers
    # ------------------------------------------------------------------

    def convergence(self):
        """Average max-probability across all islands and bits."""
        return float(self.thetas.max(axis=2).mean())

    def is_convergence(self, eps=1e-6):
        return 1.0 - self.convergence() < eps

    # ------------------------------------------------------------------

    def __str__(self):
        return (
            'PEDA(\n'
            '    dim: {}\n'
            '    lam (total): {}  sub_num: {}  sub_pop_size: {}\n'
            '    p_select: {}  n_select: {}\n'
            '    epo: {}\n'
            '    grid: {}x{}  (MDR torus)\n'
            '    neighbours: {}\n'
            ')'
        ).format(
            self.d, self.lam, self.sub_num, self.sub_pop_size,
            self.p_select, self.n_select, self.epo,
            self._grid_rows, self._grid_cols,
            {i: self._neighbours[i] for i in range(self.sub_num)},
        )
