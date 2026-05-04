import numpy as np
import os
import sys
import datetime
import argparse
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from environment.nk import problem_NKlandscape
from utils.walsh_expansion import WalshExpansion
from eda.optimizer.peda import PEDA


parser = argparse.ArgumentParser(
    description='PEDA experiments — nevergrad output format.\n'
                'Omit dim / type_instance to run all available configurations.'
)
parser.add_argument('type_problem',  type=str, choices=['NK', 'NK3', 'QUBO'])
parser.add_argument('dim',           type=int, nargs='?', default=None,
                    help='Problem dimension (omit to run all available)')
parser.add_argument('type_instance', type=int, nargs='?', default=None,
                    help='K for NK / distribution type for QUBO (omit to run all)')
parser.add_argument('--seed',         type=int, default=0)
parser.add_argument('--budget',       type=int, default=50000)
parser.add_argument('--nb-instances', type=int, default=10, dest='nb_instances')
parser.add_argument('--nb-restarts',  type=int, default=10, dest='nb_restarts')
parser.add_argument('--step-record',  type=int, default=100, dest='step_record')
# PEDA hyper-parameters
parser.add_argument('--lam',       type=int,   default=1280)
parser.add_argument('--sub-num',   type=int,   default=8,   dest='sub_num')
parser.add_argument('--p-select',  type=float, default=0.7, dest='p_select')
parser.add_argument('--epo',       type=int,   default=4)

args = parser.parse_args()

type_problem  = args.type_problem
budget        = args.budget
nb_instances  = args.nb_instances
nb_restarts   = args.nb_restarts
step_record   = args.step_record

np.random.seed(args.seed)
random.seed(args.seed)


# ------------------------------------------------------------------
# Discover available (dim, type_instance) configurations
# ------------------------------------------------------------------
def discover_nk_configs():
    base = os.path.join('instances', 'nk')
    configs = []
    for dim_name in sorted(os.listdir(base)):
        dim_path = os.path.join(base, dim_name)
        if not os.path.isdir(dim_path) or not dim_name.isdigit():
            continue
        for k_name in sorted(os.listdir(dim_path)):
            if os.path.isdir(os.path.join(dim_path, k_name)) and k_name.isdigit():
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


def discover_nk3_configs():
    base = os.path.join('instances', 'nk3')
    configs = []
    if not os.path.isdir(base):
        return configs
    for dim_name in sorted(os.listdir(base)):
        dim_path = os.path.join(base, dim_name)
        if not os.path.isdir(dim_path) or not dim_name.isdigit():
            continue
        for k_name in sorted(os.listdir(dim_path)):
            if os.path.isdir(os.path.join(dim_path, k_name)) and k_name.isdigit():
                configs.append((int(dim_name), int(k_name)))
    return configs


if type_problem == 'NK':
    all_configs = discover_nk_configs()
elif type_problem == 'NK3':
    all_configs = discover_nk3_configs()
else:
    all_configs = discover_qubo_configs()

if args.dim is not None:
    all_configs = [(d, k) for d, k in all_configs if d == args.dim]
if args.type_instance is not None:
    all_configs = [(d, k) for d, k in all_configs if k == args.type_instance]

if not all_configs:
    print('No matching configurations found.')
    sys.exit(1)

print(f'Running PEDA on {len(all_configs)} config(s): {all_configs}')


# ------------------------------------------------------------------
# Run one (dim, type_instance) configuration
# ------------------------------------------------------------------
def run_config(dim, type_instance):
    if type_problem == 'NK3':
        D = 3
    else:
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
    elif type_problem == 'NK3':
        path = os.path.join('instances', 'nk3', str(dim), str(type_instance)) + os.sep
        for num in range(nb_instances):
            fname = path + 'nk_' + str(dim) + '_' + str(type_instance) + '_3_' + str(num) + '.txt'
            problems.append(problem_NKlandscape(fname))
    else:  # NK
        path = os.path.join('instances', 'nk', str(dim), str(type_instance)) + os.sep
        for num in range(nb_instances):
            fname = path + 'nk_' + str(dim) + '_' + str(type_instance) + '_' + str(num) + '.txt'
            problems.append(problem_NKlandscape(fname))

    out_dir = os.path.join(
        REPO_ROOT, 'results', 'nevergrad', 'PEDA',
        type_problem, str(dim), str(type_instance)
    )
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for i in range(nb_instances):
        problem = problems[i]

        for r in range(nb_restarts):
            optimizer = PEDA(
                categories=categories,
                lam=args.lam,
                sub_num=args.sub_num,
                p_select=args.p_select,
                epo=args.epo,
            )

            best_raw  = float('-inf')
            num_evals = 0
            rows      = []

            while num_evals < budget:
                # Check if a full round fits in the remaining budget
                remaining = budget - num_evals
                if remaining < optimizer.lam:
                    # Partial last round: sample individually from island 0
                    # just to reach the next record boundary if needed
                    for _ in range(remaining):
                        indiv = optimizer.sample_island(0)
                        x     = np.argmax(indiv, axis=1)
                        score, _ = _score_and_cost(problem, x)
                        if score > best_raw:
                            best_raw = score
                        num_evals += 1
                        if num_evals % step_record == 0:
                            rows.append((num_evals, best_raw))
                    break  # no update on partial round

                # Full round: sample sub_pop_size individuals per island
                populations = []
                evals_list  = []

                for island_idx in range(optimizer.sub_num):
                    pop_i  = np.zeros((optimizer.sub_pop_size, dim, D), dtype=bool)
                    cost_i = np.zeros(optimizer.sub_pop_size)

                    for j in range(optimizer.sub_pop_size):
                        indiv = optimizer.sample_island(island_idx)
                        x     = np.argmax(indiv, axis=1)
                        score, cost = _score_and_cost(problem, x)

                        if score > best_raw:
                            best_raw = score

                        pop_i[j]  = indiv
                        cost_i[j] = cost
                        num_evals += 1

                        if num_evals % step_record == 0:
                            rows.append((num_evals, best_raw))

                    populations.append(pop_i)
                    evals_list.append(cost_i)

                optimizer.update(populations, evals_list)

            filename = (
                'results_nevergrad_PEDA_' + type_problem + '_' + str(dim)
                + '_' + str(type_instance) + '_budget_' + str(budget)
                + '_' + timestamp + '_i_' + str(i) + '_r_' + str(r) + '.txt'
            )
            with open(os.path.join(out_dir, filename), 'w') as f:
                f.write('runtime, score\n')
                for runtime, score in rows:
                    f.write(str(runtime) + ',' + str(score) + '\n')

            print(f'  [PEDA {type_problem} dim={dim} k={type_instance}]'
                  f' i={i} r={r}  best={best_raw:.6f}')


def _score_and_cost(problem, x):
    """Returns (score_to_maximise, cost_to_minimise) for the given integer vector."""
    if type_problem in ('NK', 'NK3'):
        eval_val = problem.eval(x)
        return -eval_val, eval_val       # eval returns -fitness → score = -eval > 0
    else:  # QUBO
        eval_val = problem.eval(2 * x - 1)
        return eval_val, -eval_val       # eval returns positive QUBO value


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
for dim, type_instance in all_configs:
    print(f'\n=== {type_problem}  dim={dim}  k={type_instance} ===')
    run_config(dim, type_instance)
