"""Compute which (algo, t, seed) task indices for the QUBO n=512 nevergrad
campaign still need to run, by scanning existing complete result files and
excluding them from the full 0..4859 range.

Reuses the EXACT same encoding as expe_nevergrad_qubo512.slurm's decode logic
(ALGO_IDX = task_id // 60, T = remainder // 10, SEED = remainder % 10), so the
computed indices line up with what the .slurm script would compute for the
same task_id.

Usage (from repo root, e.g. on Jean Zay after `git pull`):
    python jeanzay/compute_missing_tasks.py

Prints a Slurm --array value (comma-separated indices / compressed ranges)
covering only the NOT-yet-completed (algo, t, seed) combos.
"""
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGO_FILE = os.path.join(REPO_ROOT, "jeanzay", "nevergrad_discrete_algos.txt")
RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "nevergrad")

DIM = 512
NB_T = 6
NB_SEEDS = 10
BUDGET = 50000
NB_INSTANCES = 10

with open(ALGO_FILE) as f:
    algos = [line.strip() for line in f if line.strip()]
algo_index = {name: i for i, name in enumerate(algos)}

FNAME_RE = re.compile(
    r"^results_nevergrad_(?P<algo>.+)_QUBO_(?P<dim>\d+)_(?P<t>\d+)_(?P<nb>\d+)_budget_(?P<budget>\d+)_.+_(?P<seed>\d+)\.txt$"
)

done_task_ids = set()

for algo in algos:
    for t in range(NB_T):
        d = os.path.join(RESULTS_ROOT, algo, "QUBO", str(DIM), str(t))
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            m = FNAME_RE.match(fname)
            if not m:
                continue
            if int(m.group("dim")) != DIM or int(m.group("budget")) != BUDGET or int(m.group("nb")) != NB_INSTANCES:
                continue
            if int(m.group("t")) != t:
                continue
            seed = int(m.group("seed"))
            path = os.path.join(d, fname)
            # completeness check: more than just the header line
            try:
                with open(path) as fh:
                    nlines = sum(1 for _ in fh)
            except OSError:
                continue
            if nlines <= 1:
                continue
            task_id = algo_index[algo] * (NB_T * NB_SEEDS) + t * NB_SEEDS + seed
            done_task_ids.add(task_id)

total = len(algos) * NB_T * NB_SEEDS
all_ids = set(range(total))
missing = sorted(all_ids - done_task_ids)

print(f"# total={total} done={len(done_task_ids)} missing={len(missing)}", flush=True)

# compress consecutive runs into ranges a-b for a shorter --array spec
def compress(ids):
    if not ids:
        return ""
    parts = []
    start = prev = ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)

spec = compress(missing)
print(spec)
