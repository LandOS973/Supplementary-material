import numpy as np
import os
import sys
import datetime
import argparse
import random

# Anchor paths relative to this script so the script works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from environment.nk import problem_NKlandscape
from utils.walsh_expansion import WalshExpansion
from eda.optimizer.ppbil import PPBIL


parser = argparse.ArgumentParser(
    description='PPBIL experiments — nevergrad output format. NK3 not supported (binary-only).\n'
                'Omit --dim / --type-instance to run all available configurations.'
)
parser.add_argument('type_problem',   type=str, choices=['NK', 'QUBO'])
parser.add_argument('dim',            type=int, nargs='?', default=None,
                    help='Problem dimension (omit to run all available)')
parser.add_argument('type_instance',  type=int, nargs='?', default=None,
                    help='K for NK / distribution type for QUBO (omit to run all available)')
parser.add_argument('--seed',         type=int, default=0)
parser.add_argument('--budget',       type=int, default=50000)
parser.add_argument('--nb-instances', type=int, default=10, dest='nb_instances')
parser.add_argument('--nb-restarts',  type=int, default=10, dest='nb_restarts')
parser.add_argument('--step-record',  type=int, default=100, dest='step_record')

args = parser.parse_args()

type_problem  = args.type_problem
budget        = args.budget
nb_instances  = args.nb_instances
nb_restarts   = args.nb_restarts
step_record   = args.step_record

np.random.seed(args.seed)
random.seed(args.seed)


# ------------------------------------------------------------------
# Discover available (dim, type_instance) configs
# ------------------------------------------------------------------
def discover_nk_configs():
    base = os.path.join('instances', 'nk')
    configs = []
    for dim_name in sorted(os.listdir(base)):
        dim_path = os.path.join(base, dim_name)
        if not os.path.isdir(dim_path) or not dim_name.isdigit():
            continue
        for k_name in sorted(os.listdir(dim_path)):
            k_path = os.path.join(dim_path, k_name)
            if os.path.isdir(k_path) and k_name.isdigit():
                configs.append((int(dim_name), int(k_name)))
    return configs


def discover_qubo_configs():
    base = os.path.join('instances', 'QUBO')
    seen = set()
    for fname in os.listdir(base):
        parts = fname.replace('.json', '').split('_')
        try:
            n_idx = parts.index('n') + 1
            t_idx = parts.index('t') + 1
            seen.add((int(parts[n_idx]), int(parts[t_idx])))
        except (ValueError, IndexError):
            continue
    return sorted(seen)


all_configs = discover_nk_configs() if type_problem == 'NK' else discover_qubo_configs()

# Filter by --dim / --type-instance if provided
if args.dim is not None:
    all_configs = [(d, k) for d, k in all_configs if d == args.dim]
if args.type_instance is not None:
    all_configs = [(d, k) for d, k in all_configs if k == args.type_instance]


if not all_configs:
    print('No matching configurations found.')
    sys.exit(1)

print(f'Running PPBIL on {len(all_configs)} config(s): {all_configs}')


# ------------------------------------------------------------------
# Run one (dim, type_instance) configuration
# ------------------------------------------------------------------
def run_config(dim, type_instance):
    D = 2
    categories = np.full((dim,), D)

    # Load instances
    problems = []
    if type_problem == 'QUBO':
        path = os.path.join('instances', 'QUBO') + os.sep
        for num in range(1, nb_instances + 1):
            fname = (path + 'puboi_evo_n_' + str(dim) + '_t_' + str(type_instance)
                     + '_i_' + str(num) + '.json')
            f = WalshExpansion()
            f.load(fname)
            problems.append(f)
    else:  # NK
        path = os.path.join('instances', 'nk', str(dim), str(type_instance)) + os.sep
        for num in range(nb_instances):
            fname = path + 'nk_' + str(dim) + '_' + str(type_instance) + '_' + str(num) + '.txt'
            problems.append(problem_NKlandscape(fname))

    out_dir = os.path.join(
        REPO_ROOT, 'results', 'nevergrad', 'PPBIL',
        type_problem, str(dim), str(type_instance)
    )
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for i in range(nb_instances):
        problem = problems[i]

        for r in range(nb_restarts):
            optimizer = PPBIL(
                categories=categories,
                lr=0.1,
                lam=32,
                mut_prob=0.02,
                mut_shift=0.05,
            )

            best_raw  = float('-inf')
            num_evals = 0
            rows      = []

            while num_evals < budget:
                pop_size   = optimizer.lam
                population = np.zeros((pop_size, dim, D))
                fitnesses  = np.zeros(pop_size)
                collected  = 0

                for j in range(pop_size):
                    if num_evals >= budget:
                        break

                    indiv = optimizer.sampling()
                    x     = np.argmax(indiv, axis=1)

                    if type_problem == 'NK':
                        eval_val = problem.eval(x)
                        score    = -eval_val   # positive NK fitness
                        cost     = eval_val    # already negative → minimising = maximising fitness
                    else:  # QUBO
                        eval_val = problem.eval(2 * x - 1)
                        score    = eval_val    # positive QUBO value
                        cost     = -eval_val   # negate → minimising = maximising QUBO

                    if score > best_raw:
                        best_raw = score

                    population[j] = indiv
                    fitnesses[j]   = cost
                    collected    += 1
                    num_evals    += 1

                    if num_evals % step_record == 0:
                        rows.append((num_evals, best_raw))

                if collected > 0:
                    optimizer.update(population[:collected], fitnesses[:collected])

            filename = (
                'results_nevergrad_PPBIL_' + type_problem + '_' + str(dim)
                + '_' + str(type_instance) + '_budget_' + str(budget)
                + '_' + timestamp + '_i_' + str(i) + '_r_' + str(r) + '.txt'
            )
            with open(os.path.join(out_dir, filename), 'w') as f:
                f.write('runtime, score\n')
                for runtime, score in rows:
                    f.write(str(runtime) + ',' + str(score) + '\n')

            print(f'  [{type_problem} dim={dim} k={type_instance}] i={i} r={r}  best={best_raw:.6f}')


# ------------------------------------------------------------------
# Main loop over all configs
# ------------------------------------------------------------------
for dim, type_instance in all_configs:
    print(f'\n=== {type_problem}  dim={dim}  k={type_instance} ===')
    run_config(dim, type_instance)
