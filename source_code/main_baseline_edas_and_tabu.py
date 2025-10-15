
from utils.walsh_expansion import WalshExpansion
import numpy as np
import os
import datetime
import argparse
from environment.nk import problem_NKlandscape
import torch
import random
from eda import BOA, MIMIC, PBIL
from eda.optimizer.replacement import RestrictedTournament, Truncation
from eda.optimizer.selection import Top


parser = argparse.ArgumentParser(description='Nevergrad')

parser.add_argument('type_problem', type=str, help='type_problem : QUBO, NK, NK3')
parser.add_argument('name_algo', type=str, help='name Edas algo')
parser.add_argument('dim', type=int, help='Taille de l\'instance')
parser.add_argument('type_instance', type=int, help='Type instance. Corresponding to the K for NK landscape or to the type of distribution for PUBOi instances')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--budget', type=int, default=10000, help='num function')
parser.add_argument('--nb_instances', type=int, default=10, help="nb instances")
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


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def pbil_builder(dim, categories):
            
    return PBIL(
        categories=categories,
        lr=0.1,
        lam=32,
        negative_lr=0.075,
        mut_prob=0.02,
        mut_shift=0.05,
    )


def boa_builder(dim,categories):
    return BOA(
        categories=categories,
        lam=100,
        selection=Top(selection_rate=0.5),
        replacement=Truncation(replace_rate=0.5),
        criterion="bic",
    )


def mimic_builder(dim,categories):


    return MIMIC(
        categories=categories,
        lam=100,
        replacement=RestrictedTournament(dim, replace_rate=0.5, window_size=2),

    )


class EDASearchWrapper:
    def __init__(self, budget: int, optimizer_builder, name_algo, categories, id):
        self.budget = budget
        self.optimizer_builder = optimizer_builder
        self.name_algo = name_algo

        self.best_fitness = float("-inf")

        self.categories = categories
        
        self.id = id


    def __call__(self, problem, dim, D, type_pb, table_scores) -> None:

        optimizer = self.optimizer_builder(dim, self.categories)

        num_evals = 0
        
        index = 0
        
        while num_evals < self.budget:
            population_size = (
                optimizer.lam
            )  # this can change during the optimization (in update)

            population = np.zeros((population_size, dim, D))
            fitnesses = np.zeros((population_size,))

            for j in range(population_size):
                if (num_evals >= self.budget):
                    return


                indiv = optimizer.sampling()

                x = np.argmax(indiv, axis=1)

                if (type_pb == "NK" or type_pb == "NK3" or type_pb == "Bonnans"):
                    fitness = -problem.eval(x)
                elif(type_pb == "QUBO"):
                    solution = 2*x - 1
                    fitness = -problem.eval(solution)

                if(fitness > self.best_fitness):
                    self.best_fitness = fitness

                num_evals += 1

                population[j] = indiv
                fitnesses[j] = fitness


                if(num_evals%step_record == 0 and num_evals >  0):



                    table_scores[index, self.id] = -self.best_fitness

                    index += 1
                    
            optimizer.update(population, -fitnesses)
            
            

            
class TabuSearchAlgorithm:
    def __init__(self, budget: int, categories, id : int):
        self.budget = budget

        self.best_fitness = float("-inf")

        self.categories = categories

        self.max_dim = np.max(categories)

        self.id = id


    def __call__(self,  problem, dim,  K, type_pb, table_scores) -> None:


        self.params1 = int(dim//10)
        self.params2 = 10

        current_indiv = np.zeros((dim), dtype=int)

        for x in range(dim):
            current_indiv[x] = random.randint(0,int(self.categories[x])-1)

        if(type_pb == "NK" or type_pb == "NK3" or type_pb == "Bonnans"):
            current_fitness = -problem.eval(current_indiv)
        elif (type_pb == "QUBO"):
            solution = 2 * current_indiv - 1
            current_fitness = -problem.eval(solution)
        

        self.best_fitness = current_fitness

        num_evals = 1

        tabuTenure = np.zeros((dim))

        iter_ = 0
        
        index = 0

        while num_evals < self.budget:


            best_delta = -float("inf")
            best_x = -1
            best_v = -1
            trouve = 1

            for x in range(dim):


                for v in range(self.categories[x]):

                    if(current_indiv[x] != v):

                        new_indiv = np.copy(current_indiv)
                        new_indiv[x] = v

                        if (type_pb == "NK" or type_pb == "NK3" or type_pb == "Bonnans"):
                            new_fitness = -problem.eval(new_indiv)
                        elif(type_pb == "QUBO"):
                            solution = 2*new_indiv - 1
                            new_fitness = -problem.eval(solution)

                        num_evals += 1

                        delta = new_fitness - current_fitness

                        if ((tabuTenure[x] <= iter_) or (new_fitness > self.best_fitness)):

                            if (delta > best_delta):
                                best_x = x
                                best_v = v
                                best_delta = delta
                                trouve = 1

                            elif (delta == best_delta):

                                trouve += 1

                                if (random.randint(1, trouve) == 1):
                                    best_x = x
                                    best_v = v

                        if (new_fitness > self.best_fitness):
                            self.best_fitness = new_fitness

                        if(num_evals%100 == 0 and num_evals > 0):



                            table_scores[index, self.id] = self.best_fitness

                            index += 1

                        if (num_evals >= self.budget):
                            break
                if (num_evals >= self.budget):
                    break

            current_fitness += best_delta

            current_indiv[best_x] = best_v

            tabuTenure[best_x] =  int(self.params1) + random.randint(0, int(self.params2))  + iter_

            iter_ += 1


  
list_problem = []
if(type_problem == "QUBO"):

    path = "instances/QUBO/"
    for num_instance in range(1, nb_instances + 1):
        filename = path + "puboi_evo_n_" + str(dim) + "_t_" + str(type_instance) + "_i_" + str(num_instance) + ".json"
        f = WalshExpansion()
        f.load(filename)

        list_problem.append(f)

    D = 2
    categories = np.full((dim,), D)

elif(type_problem == "NK"):

    path = "instances/nk/" + str(dim) + "/" + str(type_instance) + "/"
    for num_instance in range(nb_instances):
        name_instance = path + "nk_" + str(dim) + "_" + str(type_instance)  + "_" + str(num_instance) + ".txt"
        
        list_problem.append(problem_NKlandscape(name_instance))

    D = 2
    categories = np.full((dim,), D)

elif(type_problem == "NK3"):

    D = 3
    categories = np.full((dim,), D)

    path = "instances/nk3/" + str(dim) + "/" + str(type_instance) + "/"
    for num_instance in range(nb_instances):
        name_instance = path + "nk_" + str(dim) + "_" + str(type_instance) + "_" + str(D) + "_" + str(num_instance) + ".txt"
        list_problem.append(problem_NKlandscape(name_instance))


    print("categories")
    print(categories)


elif (type_problem == "Bonnans"):

    list_problem = getListInstance_Bonnans(nb_instances, dim)
    
    D = 2
    categories = np.full((dim,), D)



if not os.path.exists("results/results_EDAs_final/" + name_algo ):
    os.mkdir("results/results_EDAs_final/" + name_algo)

if not os.path.exists("results/results_EDAs_final/" + name_algo + "/" + type_problem ):
    os.mkdir("results/results_EDAs_final/" + name_algo + "/" + str(type_problem))
    
if not os.path.exists("results/results_EDAs_final/" + name_algo + "/" + type_problem + "/" + str(dim) ):
    os.mkdir("results/results_EDAs_final/" + name_algo + "/" + type_problem + "/" + str(dim))

if not os.path.exists("results/results_EDAs_final/" + name_algo + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance) ):
    os.mkdir("results/results_EDAs_final/" + name_algo + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance))
    

path_result = "results/results_EDAs_final/" + name_algo + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance) + "/"
#path_logs = "logs/"
name_file_result = "results_EDAs_final_" + name_algo + "_" + type_problem + "_" + str(dim) + "_" + str(type_instance) + "_" + str(nb_instances) + "_budget_" + str(budget) + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(seed) + ".txt"


f_results = open(path_result + name_file_result, "w")
f_results.write("runtime, mean, median, std, 2%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 98%" + "\n")
f_results.close()




if (name_algo == "Tabu"):
    list_algos = [TabuSearchAlgorithm(budget, categories, i) for i in range(nb_instances)]

if (name_algo == "PBIL"):
    list_algos = [EDASearchWrapper(budget, pbil_builder, name_algo, categories, i) for i in range(nb_instances)]

if (name_algo == "MIMIC"):
    list_algos = [EDASearchWrapper(budget, mimic_builder, name_algo, categories, i) for i in range(nb_instances)]

if (name_algo == "BOA"):
    list_algos = [EDASearchWrapper(budget, boa_builder, name_algo, categories, i) for i in range(nb_instances)]
            
            
table_scores = np.zeros((budget//step_record,nb_instances ))



for idx_run in range(nb_instances):

    print("idx_run : " + str(idx_run))

    list_algos[idx_run].__call__(list_problem[idx_run], dim, D, type_problem, table_scores)
    




for index in range(table_scores.shape[0]):

    array_score = table_scores[index]

    mean = np.mean(array_score)
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

