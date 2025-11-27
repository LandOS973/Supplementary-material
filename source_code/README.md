Replication code for the paper "Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning"

The algorithm is in Python 3.11.5. 
All the library required to launch the multivariate RL EDAs are in the file requirement.txt.

Other libraries are required such as Nevergrad (see https://facebookresearch.github.io/nevergrad/) to run the competing algorithms.


An example of python command to run the (sigma,simga')-RL-EDA version (reference version of the paper) with default hyperparameters for 10 QUBO instances  with n=128 and K=0 and 10 restarts for each instance (100 runs) on GPU device is :

python main_ppo_eda.py QUBO 128 0 --verbose 


## Nevergrad competing algorithms

To run a nervergrad algorithm such as DiscreteDE the command line is :

python main_nevergrad.py QUBO DiscreteDE 128 0


## Other EDAs and Tabu algorithms

To run the Tabu algorithm on the same instance the command line is

python main_baseline_edas_and_tabu.py QUBO Tabu 128 0

To run the PBIL algorithm on the same instance the command line is

python main_baseline_edas_and_tabu.py QUBO PBIL 128 0


## Note on problem instances

In this supplementary material, the instances of the NK3 problem with N > 64 and K > 4 have been remove because their were to big. They will be added after the submission process in a github repository.

### Test grid overrides

When running from the **repo root**, call:

```
python source_code/main_univ.py --grid-settings grid/my_grid.json --visualization=false
```

If you are already inside `source_code/`, drop the extra prefix:

```
python main_univ.py --grid-settings ../grid/my_grid.json --visualization=false
```
