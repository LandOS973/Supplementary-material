import torch
import numpy as np
from tqdm import tqdm



def getTensorInstances_NK(path, nb_instances, nb_restarts, size_pop,  N, D,  K, device):


    list_matrix_locus = []
    list_matrix_contrib = []
    list_matrix_Q = []

    if(path != ""):

        for num_instance in range(nb_instances):

            if(D > 2):
                name_instance = path + "nk_" + str(N) + "_" + str(K) + "_" + str(D) + "_" + str(num_instance) + ".txt"
            else:
                name_instance = path + "nk_" + str(N) + "_" + str(K) + "_" + str(num_instance) + ".txt"

            f = open(name_instance, "r")
            lignes = f.readlines()

            Q = np.zeros((N,N))

            links = []
            for n in range(N):
                links.append([])
                for k in range(K + 1):
                    links[n].append(int(lignes[1 + n * (K + 1) + k]))

                    if(int(lignes[1 + n * (K + 1) + k]) != n):
                        Q[n,int(lignes[1 + n * (K + 1) + k])] = 1
                        Q[int(lignes[1 + n * (K + 1) + k]), n] = 1

                links[n].append([])
                for k in range(D ** (K + 1)):
                    links[n][-1].append(float(lignes[1 + N * (K + 1) + n * (D ** (K + 1)) + k]))



            matrix_Q = torch.tensor(Q, dtype=torch.int64)

            matrix_locus = np.zeros((N, K + 1))
            for i in range(N):
                matrix_locus[i, :] = links[i][:-1]

            matrix_locus = torch.tensor(matrix_locus, dtype=torch.int64)

            matrix_contrib = np.zeros((N, D ** (K + 1)))

            for i in range(N):
                matrix_contrib[i, :] = links[i][-1]

            matrix_contrib = torch.tensor(matrix_contrib, dtype=torch.float32).unsqueeze(0)

            matrix_contrib = (matrix_contrib).repeat([size_pop, 1, 1])

            for i in range(nb_restarts):
                list_matrix_locus.append(matrix_locus)
                list_matrix_contrib.append(matrix_contrib)
                list_matrix_Q.append(matrix_Q)

    else:

        for i in range(nb_instances):

            matrix_locus = np.zeros((N, K + 1))
            # Générer les voisins de chaque élément dans le paysage NK
            for x in range(N):
                neigh = []
                neigh.append(x)
                for y in range(K):
                    x1 = np.random.randint(0, N)
                    while x1 in neigh:
                        x1 = np.random.randint(0, N)
                    neigh.append(x1)

                neigh.sort()  # Trie les voisins pour assurer un ordre
                matrix_locus[x, :] = neigh

            matrix_locus = torch.tensor(matrix_locus, dtype=torch.int64)


            matrix_contrib = np.random.random((N, D ** (K + 1)))

            matrix_contrib = torch.tensor(matrix_contrib, dtype=torch.float32).unsqueeze(0)

            matrix_contrib = (matrix_contrib).repeat([size_pop,1,1])

            for i in range(nb_restarts):
                list_matrix_locus.append(matrix_locus)
                list_matrix_contrib.append(matrix_contrib)

    with torch.no_grad():

        tensor_matrix_locus = torch.stack(list_matrix_locus, dim=0)
        tensor_matrix_contrib = torch.stack(list_matrix_contrib, dim=0)
        tensor_matrix_locus = (tensor_matrix_locus.unsqueeze(1)).repeat([1, size_pop, 1, 1]).to(device)
        tensor_matrix_contrib = tensor_matrix_contrib.to(device)
        
        tensor_matrix_Q = torch.stack(list_matrix_Q, dim=0)
        #tensor_matrix_Q = (tensor_matrix_Q.unsqueeze(1)).repeat([1, size_pop, 1, 1]).to(device)

    return tensor_matrix_locus, tensor_matrix_contrib, tensor_matrix_Q





def get_Score_trajectoriesNK_cuda(strategy, N, K, D, nb_instances, nb_restarts, budget, size_pop, vectorIndex, tensor_matrix_locus, tensor_matrix_contrib, device, verbose, name_file):



    strategy.reset_learned_parameters(nb_instances*nb_restarts)

    bestScore = torch.ones(nb_instances*nb_restarts).to(device)*(-99999)


    size_pop = strategy.lambda_



    nb_iterations = budget//size_pop


    if(verbose):
        pbar = tqdm(range(nb_iterations))
    else:
        pbar = range(nb_iterations)




    if(name_file is not None):
        f_results = open(name_file, "w")
        f_results.write("runtime, mean, median, std, 2%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 98%" + "\n")
        f_results.close()
        
    list_tensor_solution = []

    for epoch in pbar:




        tensor_solution = strategy.sample_solutions()

        #if epoch == 0:
            
            
            #print("startSolution")
            #print(startSolution.size())
        
        
        # #

        #Compute score
        tensor_solution_rep = torch.transpose(tensor_solution, 2,3).repeat([1, 1, N, 1])
        tensor_solution_locus = torch.gather(input=tensor_solution_rep, dim=3, index=tensor_matrix_locus)
        tensor_solution_locus = tensor_solution_locus.float()
        index_th = torch.sum(tensor_solution_locus*vectorIndex, dim=3).type(torch.int64).unsqueeze(3)
        tensor_score = torch.sum(tensor_matrix_contrib.gather(3, index_th), dim = 2).squeeze(2)


        #print("tensor score")
        #print(tensor_score[0]/N)
        
        #tensor_score_np = np.zeros((nb_instances, size_pop))
        
        #list_problem = []
        
        #path = "instances/nk/" + str(N) + "/" + str(K) + "/"
        #for num_instance in range(nb_instances):
            #name_instance = path + "nk_" + str(N) + "_" + str(K) + "_" + str(
                #num_instance) + ".txt"
            #list_problem.append(problem_NKlandscape(name_instance))
        
        #tensor_solution_np = tensor_solution.cpu().numpy()
        
        #print("tensor_solution_np")
        #print(tensor_solution_np[0,0,:])
        
        #for i in range(nb_instances):
            #for j in range(size_pop):
                #tensor_score_np[i,j] = list_problem[i].eval(tensor_solution_np[i,j])
        
        #print("tensor_score_np")
        #print(tensor_score_np[0])
        
        #print(coucou)


        current_score = torch.max(tensor_score, dim=1).values

        list_tensor_solution.append(tensor_solution)
        index_solution = torch.argmax(tensor_score, dim=1)
        index_solution = index_solution.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1,N,1)
        best_current_solution = torch.gather(tensor_solution, 1 , index_solution).squeeze(3).squeeze(1)

        if(epoch == 0):
            bestGlobalSolution = best_current_solution
        else:
            tmp_current_score = current_score.unsqueeze(1).repeat(1,N)
            tmp_bestScore = bestScore.unsqueeze(1).repeat(1,N)
            bestGlobalSolution = torch.where(tmp_current_score > tmp_bestScore, best_current_solution,  bestGlobalSolution)
        
        
        bestScore = torch.where(current_score > bestScore, current_score,  bestScore)


        strategy.updateDistribution(tensor_solution, tensor_score)

        if(verbose):
            pbar.set_postfix(bestScore=torch.mean(bestScore).item()/N,
                            current_score = torch.mean(current_score).item()/N)

        if(name_file is not None):
            if(((epoch +1)*size_pop) % 100 == 0):
                
                bestScore_np = -bestScore.cpu().numpy()/N               
                mean = np.mean(bestScore_np)
                median = np.percentile(bestScore_np, 50)
                std = np.std(bestScore_np)
                _2per = np.percentile(bestScore_np, 2)
                _5per = np.percentile(bestScore_np, 5)
                _10per = np.percentile(bestScore_np, 10)
                _25per = np.percentile(bestScore_np, 25)
                _75per = np.percentile(bestScore_np, 75)
                _90per = np.percentile(bestScore_np, 90)
                _95per = np.percentile(bestScore_np, 95)
                _98per = np.percentile(bestScore_np, 98)
                
                f_results = open(name_file, "a")
                f_results.write(str((epoch + 1)*size_pop) + "," +  str(mean) + "," +  str(median) + "," +  str(std) + "," +  str(_2per) + "," +  str(_5per) + "," +  str(_10per) + "," +  str(_25per) + "," +  str(median) + "," +  str(_75per) + "," +  str(_90per) + "," +  str(_95per) + "," +  str(_98per) + "\n")
                f_results.close()


    if(name_file is not None):
        f_hamming = open(name_file + "_HD", "w")
        #f_hamming = open("hamming_distance.csv", "w")
        f_hamming.write("runtime, avg distance, avg std pop" + "\n")
        f_hamming.close()


        
    lastSolution = tensor_solution[:,0,:,:].squeeze(2)

    
    bestGlobalSolution = bestGlobalSolution.unsqueeze(1).repeat(1,10,1)
    #lastSolution = lastSolution.unsqueeze(1).repeat(1,10,1)
    
    for idx, tensor_solution in enumerate(list_tensor_solution):
        

        
        hamming_distance = torch.sum(torch.abs(tensor_solution.squeeze(3) - bestGlobalSolution), dim=2).cpu().numpy()
        
        #hamming_distance = torch.sum(torch.abs(tensor_solution.squeeze(3) - lastSolution), dim=2).cpu().numpy()
        
        
        avg_distance = np.mean(hamming_distance)
        avg_std_distance = np.mean(np.std(hamming_distance, axis = 1))

        f_hamming = open(name_file + "_HD_best", "a")
        f_hamming.write(str((idx + 1)*size_pop) + "," +  str(avg_distance) + "," +  str(avg_std_distance) + "\n")
        f_hamming.close()
            
    return -bestScore_np




class problem_NKlandscape:

    def __init__(self, file, max_nb_turn=-1):

        f = open(file, "r")
        lignes = f.readlines()

        head = lignes[0].split()


        self.N = int(head[0])
        self.K = int(head[1])

        if (len(head) > 2):
            self.D = int(head[2])
        else:
            self.D = 2

        # print("self.D : " + str(self.D))

        self.links = []
        for n in range(self.N):
            self.links.append([])
            for k in range(self.K + 1):
                self.links[n].append(int(lignes[1 + n * (self.K + 1) + k]))

            self.links[n].append([])
            for k in range(self.D ** (self.K + 1)):
                self.links[n][-1].append(float(lignes[1 + self.N * (self.K + 1) + n * (self.D ** (self.K + 1)) + k]))

        # print(self.links)

        f.close()

        self.currentScore = 0

        self.max_nb_turn = max_nb_turn

        self.game_state = np.random.randint(2, size=self.N)
        self.turn = 0


    def reset(self):
        self.game_state = np.random.randint(2, size=self.N)
        self.turn = 0
        return self.game_state

    def setState(self, state):

        self.game_state = state

    def perturbation(self, alpha):

        num_bits_to_perturb = int(alpha * self.N)  # Calcul du nombre de bits à perturber

        # Choisissez aléatoirement num_bits_to_perturb indices de bits à perturber
        perturb_indices = np.random.choice(self.N, num_bits_to_perturb, replace=False)

        # Inversez les valeurs des bits choisis aléatoirement
        for index in perturb_indices:
            self.game_state[index] = (self.game_state[index] + 1) % 2

    def getDeltaFitness(self, action):

        old_value = self.game_state[action]
        self.game_state[action] = (self.game_state[action] + 1) % 2
        deltaFitness = 0

        for link in self.links:
            if action in link:
                malus = []
                bonus = []
                for i in link[:-1]:
                    bonus.append(self.game_state[i])
                    if i == action:
                        malus.append(old_value)
                    else:
                        malus.append(self.game_state[i])
                malus_index = 0
                bonus_index = 0
                for i in range(self.K + 1):
                    malus_index += (2 ** (self.K - i)) * malus[i]
                    bonus_index += (2 ** (self.K - i)) * bonus[i]

                deltaFitness -= link[-1][int(malus_index)]
                deltaFitness += link[-1][int(bonus_index)]

        self.game_state[action] = (self.game_state[action] + 1) % 2

        return deltaFitness


    def getAllDeltaFitness(self):

        self.neighDeltaFitness = []

        for i in range(self.N):
            self.neighDeltaFitness.append(self.getDeltaFitness(i))

        return self.neighDeltaFitness


    def step(self, action):
        old_value = self.game_state[action]
        self.game_state[action] = (self.game_state[action] + 1) % 2
        deltaFitness = 0

        for link in self.links:
            if action in link:
                malus = []
                bonus = []
                for i in link[:-1]:
                    bonus.append(self.game_state[i])
                    if i == action:
                        malus.append(old_value)
                    else:
                        malus.append(self.game_state[i])
                malus_index = 0
                bonus_index = 0
                for i in range(self.K + 1):
                    malus_index += (2 ** (self.K - i)) * malus[i]
                    bonus_index += (2 ** (self.K - i)) * bonus[i]

                deltaFitness -= link[-1][int(malus_index)]
                deltaFitness += link[-1][int(bonus_index)]

        self.turn += 1

        if self.turn == self.max_nb_turn:
            terminated = True
        else:
            terminated = False


        return self.game_state, deltaFitness, terminated

    def eval(self,solution):

        sco = 0

        for link in self.links:
            bonus = []
            for i in link[:-1]:
                bonus.append(solution[i])
            bonus_index = 0
            for i in range(self.K + 1):
                bonus_index += (self.D ** (self.K - i)) * bonus[i]

            sco += link[-1][int(bonus_index)]

        return -sco/self.N


    def setScore(self,currentScore):

        self.currentScore = currentScore

    def getScore(self):

        return self.currentScore
