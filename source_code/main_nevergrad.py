import nevergrad as ng
from utils.walsh_expansion import WalshExpansion
import numpy as np
import os
import datetime
import argparse
from environment.nk import problem_NKlandscape
import random
import warnings
warnings.filterwarnings("ignore")
import sys

parser = argparse.ArgumentParser(description='Nevergrad')

parser.add_argument('type_problem', type=str, help='type_problem : QUBO, NK or NK3')
parser.add_argument('name_algo', type=str, help='name Nevergrad algo')
parser.add_argument('dim', type=int, help='Taille de l\'instance')
parser.add_argument('type_instance', type=int, help='Type instance. Corresponding to the K for NK landscape or to the type of distribution for PUBOi instances')
parser.add_argument('--nb_instances', type=int, default= 10, help="nb instances")

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--budget', type=int, default=10000, help='num function')

parser.add_argument('--step_record', type=int, default=100, help="nb instances")

args = parser.parse_args()

type_problem = args.type_problem
name_algo = args.name_algo
dim = args.dim
type_instance = args.type_instance
seed = args.seed

nb_instances = args.nb_instances
budget = args.budget
step_record = args.step_record


np.random.seed(seed)
random.seed(seed)



list_problem = []


if(type_problem == "QUBO"):

    path = "instances/QUBO/"
    for num_instance in range(1, nb_instances + 1):
        filename = path + "puboi_evo_n_" + str(dim) + "_t_" + str(type_instance) + "_i_" + str(num_instance) + ".json"
        f = WalshExpansion()
        f.load(filename)
        list_problem.append(f)

    param = ng.p.TransitionChoice(range(2), repetitions=dim, ordered=False)


elif(type_problem == "NK"):

    path = "instances/nk/" + str(dim) + "/" + str(type_instance) + "/"
    for num_instance in range(nb_instances):
        name_instance = path + "nk_" + str(dim) + "_" + str(type_instance) + "_" + str(num_instance) + ".txt"
        list_problem.append(problem_NKlandscape(name_instance))

    param = ng.p.TransitionChoice(range(2), repetitions=dim, ordered=False)
    
    
elif(type_problem == "NK3"):

    D = 3

    path = "instances/nk3/" + str(dim) + "/" + str(type_instance) + "/"
    for num_instance in range(nb_instances):
        name_instance = path + "nk_" + str(dim) + "_" + str(type_instance) + "_" + str(D) + "_" + str(num_instance) + ".txt"
        list_problem.append(problem_NKlandscape(name_instance))

    param = ng.p.TransitionChoice(range(D), repetitions=dim, ordered=False)


    
if(type_problem == "QUBO"):
    type_problem = "UBQP"

if not os.path.exists("results/results_nevergrad_final/" + name_algo ):
    os.mkdir("results/results_nevergrad_final/" + name_algo)

if not os.path.exists("results/results_nevergrad_final/" + name_algo + "/" + type_problem ):
    os.mkdir("results/results_nevergrad_final/" + name_algo + "/" + str(type_problem))
    
if not os.path.exists("results/results_nevergrad_final/" + name_algo + "/" + type_problem + "/" + str(dim) ):
    os.mkdir("results/results_nevergrad_final/" + name_algo + "/" + type_problem + "/" + str(dim))

if not os.path.exists("results/results_nevergrad_final/" + name_algo + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance) ):
    os.mkdir("results/results_nevergrad_final/" + name_algo + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance))
    

path_result = "results/results_nevergrad_final/" + name_algo + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance) + "/"
path_logs = "logs/"
name_file_result = "results_nevergrad_" + name_algo + "_" + type_problem + "_" + str(dim) + "_" + str(type_instance) + "_" + str(nb_instances) + "_budget_" + str(budget) + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(seed) + ".txt"


f_results = open(path_result + name_file_result, "w")
f_results.write("runtime, mean, median, std, 2%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 98%" + "\n")
f_results.close()


table_scores = np.zeros((budget//step_record, nb_instances ))


def get_Score_Problem(optimizer, problem, type_problem, id):

    cpt = 1
    index = 0

    best_loss = float("inf")

    for _ in range(optimizer.budget):

        x = optimizer.ask()

        if (type_problem == "QUBO" or type_problem == "UBQP"):
            solution = 2 * np.array(x.value) - 1
            loss = problem.eval(solution)
        else:
            loss = problem.eval(np.array(x.value))

        if(loss < best_loss ):
            best_loss = loss

        optimizer.tell(x, loss)

        if(cpt%step_record == 0):
            table_scores[index, id] = best_loss
            
            #print("index : " + str(index) + " best loss " + str(best_loss))

    

            index += 1

        cpt += 1
        
        
    print("table score")
    print(table_scores)
    
    print(best_loss)
    


    return best_loss



print(name_algo)

list_algos = [ng.optimizers.registry.get(name_algo)(parametrization=param, budget=budget) for i in range(nb_instances)]



list_all_scores = []
list_best_scores = []


for idx_run in range(nb_instances):

    print("instance : " + str(idx_run))

    if(type_problem == "nasbench"):
        score = get_Score_Problem(list_algos[idx_run], objective, type_problem, idx_run)
    else:
        score = get_Score_Problem(list_algos[idx_run], list_problem[idx_run], type_problem, idx_run)

    list_best_scores.append(score)




for index in range(table_scores.shape[0]):

    array_score = table_scores[index]

    #print(array_score)
    
    mean = np.mean(array_score)
    
    #print(mean)
    
    median = np.percentile(array_score, 50)
    std = np.std(array_score)
    _2per = np.percentile(array_score, 2)
    _5per = np.percentile(array_score, 5)
    _10per = np.percentile(array_score, 10)
    _25per = np.percentile(array_score, 25)
    _75per = np.percentile(array_score, 75)
    _90per = np.percentile(array_score, 90)
    _95per = np.percentile(array_score, 95)
    _98per = np.percentile(array_score, 98)



    f_results = open(path_result + name_file_result, "a")
    f_results.write(str((index + 1)*step_record) + "," +  str(mean) + "," +  str(median) + "," +  str(std) + "," +  str(_2per) + "," +  str(_5per) + "," +  str(_10per) + "," +  str(_25per) + "," +  str(median) + "," +  str(_75per) + "," +  str(_90per) + "," +  str(_95per) + "," +  str(_98per) + "\n")
    f_results.close()

os._exit(1)
