import torch
import numpy as np
from tqdm import tqdm

from environment.visualization import render_agent_dashboard, render_svgd_field_plot
from environment.metrics import MetricsCalculator



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





def get_Score_trajectoriesNK_cuda(
    strategy,
    N,
    K,
    D,
    nb_instances,
    nb_restarts,
    budget,
    size_pop,
    vectorIndex,
    tensor_matrix_locus,
    tensor_matrix_contrib,
    device,
    verbose,
    enable_visualization=True,
    return_history=False,
):



    strategy.reset_learned_parameters(nb_instances*nb_restarts)

    bestScore = torch.ones(nb_instances*nb_restarts).to(device)*(-99999)
    agent_lambdas = getattr(strategy, "agent_lambdas", None)
    track_leader = isinstance(agent_lambdas, (list, tuple)) and len(agent_lambdas) > 0
    collect_summary_metrics = track_leader
    collect_pairwise_metrics = track_leader and bool(enable_visualization)
    agent_best_overall = None
    if track_leader:
        agent_best_overall = [torch.ones(nb_instances*nb_restarts).to(device)*(-99999) for _ in agent_lambdas]


    size_pop = strategy.lambda_



    nb_iterations = budget//size_pop


    avg_hamming_history = []
    avg_js_history = []
    avg_l2_history = []
    avg_l1_history = []
    avg_entropy_history = []
    best_fitness_history = []
    runtime_steps = []
    score_mean_history = []
    score_median_history = []
    score_std_history = []
    score_p2_history = []
    score_p5_history = []
    score_p10_history = []
    score_p25_history = []
    score_p50_history = []
    score_p75_history = []
    score_p90_history = []
    score_p95_history = []
    score_p98_history = []
    agent_fitness_history = []
    hamming_pairwise_history = []
    js_pairwise_history = []
    l2_pairwise_history = []
    l1_pairwise_history = []
    entropy_agent_history = []
    kl_pairwise_history = []
    avg_kernel_value_history = []
    avg_kernel_grad_history = []
    metrics = MetricsCalculator(normalization_factor=N)

    use_tqdm = bool(verbose and enable_visualization)
    pbar = tqdm(range(nb_iterations)) if use_tqdm else range(nb_iterations)




        

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

        scores_np = bestScore.detach().cpu().numpy() / N
        score_mean_history.append(float(np.mean(scores_np)))
        score_median_history.append(float(np.percentile(scores_np, 50)))
        score_std_history.append(float(np.std(scores_np)))
        score_p2_history.append(float(np.percentile(scores_np, 2)))
        score_p5_history.append(float(np.percentile(scores_np, 5)))
        score_p10_history.append(float(np.percentile(scores_np, 10)))
        score_p25_history.append(float(np.percentile(scores_np, 25)))
        score_p50_history.append(float(np.percentile(scores_np, 50)))
        score_p75_history.append(float(np.percentile(scores_np, 75)))
        score_p90_history.append(float(np.percentile(scores_np, 90)))
        score_p95_history.append(float(np.percentile(scores_np, 95)))
        score_p98_history.append(float(np.percentile(scores_np, 98)))

        global_current = metrics.compute_fitness(current_score)
        global_best = metrics.compute_fitness(bestScore)

        leader_idx = None
        avg_js = None
        avg_hamming = None
        if track_leader:
            agent_best_scores = []
            start_idx = 0
            for idx, agent_lambda in enumerate(agent_lambdas):
                end_idx = start_idx + agent_lambda
                agent_scores = tensor_score[:, start_idx:end_idx]
                agent_best_values, _ = torch.max(agent_scores, dim=1)
                agent_best_scores.append(agent_best_values)
                agent_best_overall[idx] = torch.where(agent_best_values > agent_best_overall[idx],
                                                       agent_best_values,
                                                       agent_best_overall[idx])
                start_idx = end_idx

            agent_mean_scores = torch.stack([scores.mean() for scores in agent_best_scores])
            leader_idx = torch.argmax(agent_mean_scores).item()

            if collect_summary_metrics:
                avg_hamming, pairwise_matrix = metrics.compute_average_hamming(strategy.agents)
                avg_l1, pairwise_l1 = metrics.compute_l1_distance(strategy.agents)
                avg_entropy, per_agent_entropy = metrics.compute_entropy(strategy.agents)
                avg_hamming_history.append(avg_hamming if avg_hamming is not None else 0.0)
                avg_l1_history.append(avg_l1 if avg_l1 is not None else 0.0)
                avg_entropy_history.append(avg_entropy if avg_entropy is not None else 0.0)
            else:
                pairwise_matrix = None
                pairwise_l1 = None
                per_agent_entropy = None
                avg_hamming_history.append(0.0)
                avg_l1_history.append(0.0)
                avg_entropy_history.append(0.0)

            if collect_pairwise_metrics:
                avg_js, pairwise_js = metrics.compute_average_js(strategy.agents)
                avg_l2, pairwise_l2 = metrics.compute_l2_distance(strategy.agents)
                avg_js_history.append(avg_js if avg_js is not None else 0.0)
                avg_l2_history.append(avg_l2 if avg_l2 is not None else 0.0)
                hamming_pairwise_history.append(pairwise_matrix.tolist() if pairwise_matrix is not None else None)
                js_pairwise_history.append(pairwise_js.tolist() if pairwise_js is not None else None)
                l2_pairwise_history.append(pairwise_l2.tolist() if pairwise_l2 is not None else None)
                l1_pairwise_history.append(pairwise_l1.tolist() if pairwise_l1 is not None else None)
                entropy_agent_history.append(per_agent_entropy if per_agent_entropy is not None else None)
            else:
                avg_js_history.append(0.0)
                avg_l2_history.append(0.0)
                hamming_pairwise_history.append(None)
                js_pairwise_history.append(None)
                l2_pairwise_history.append(None)
                l1_pairwise_history.append(None)
                entropy_agent_history.append(None)
            agent_fitness_history.append([score.item() for score in agent_mean_scores])
            kernel_stats_fn = getattr(strategy, "get_latest_kernel_metrics", None)
            kernel_stats = kernel_stats_fn() if callable(kernel_stats_fn) else None
            if kernel_stats:
                avg_kernel_value_history.append(kernel_stats.get("avg_kernel_value", 0.0))
                avg_kernel_grad_history.append(kernel_stats.get("avg_kernel_grad", 0.0))
            else:
                avg_kernel_value_history.append(0.0)
                avg_kernel_grad_history.append(0.0)

        runtime_steps.append((epoch + 1) * size_pop)
        best_fitness_history.append(-global_best)

        if(use_tqdm):
            postfix = {"bestScore": -global_best, "current_score": -global_current}
            if track_leader and leader_idx is not None:
                postfix["leader"] = leader_idx
                postfix["avg_hamming"] = avg_hamming
                postfix["avg_js"] = avg_js
            pbar.set_postfix(**postfix)



    bestScore_np = -bestScore.detach().cpu().numpy()/N
    if track_leader and enable_visualization and agent_best_overall is not None and hasattr(strategy, "agents"):
        print("Per-agent summary:")
        for idx, agent in enumerate(strategy.agents):
            avg_best = -torch.mean(agent_best_overall[idx]).item()/N
            theta_mean = torch.mean(metrics.agent_theta_tensor(agent)).item()
            print(f"Agent {idx}: avg_best_score={avg_best:.4f}, theta_mean={theta_mean:.6f}")

    if enable_visualization:
        iterations = [(idx + 1) * size_pop for idx in range(len(avg_hamming_history))] if avg_hamming_history else []
        num_agents = len(strategy.agents) if hasattr(strategy, "agents") else 0
        theta_history = None
        theta_history_fn = getattr(strategy, "get_theta_history", None)
        if callable(theta_history_fn):
            theta_history = theta_history_fn()
        render_agent_dashboard(
            iterations,
            avg_hamming_history,
            avg_js_history,
            agent_fitness_history,
            num_agents,
            theta_history,
            hamming_pairwise_history,
            js_pairwise_history,
            avg_l2_history,
            l2_pairwise_history,
            avg_l1_history,
            l1_pairwise_history,
            avg_entropy_history,
            entropy_agent_history,
            avg_kernel_value_history,
            avg_kernel_grad_history,
        )

        svgd_snapshot_fn = getattr(strategy, "get_svgd_field_snapshot", None)
        if callable(svgd_snapshot_fn):
            snapshot = svgd_snapshot_fn()
            if snapshot:
                render_svgd_field_plot(snapshot)

    if return_history:
        history = dict(
            runtime=runtime_steps,
            best_fitness=best_fitness_history,
            avg_hamming=avg_hamming_history,
            avg_js=avg_js_history,
            avg_l2=avg_l2_history,
            avg_l1=avg_l1_history,
            avg_entropy=avg_entropy_history,
            score_mean=score_mean_history,
            score_median=score_median_history,
            score_std=score_std_history,
            score_p2=score_p2_history,
            score_p5=score_p5_history,
            score_p10=score_p10_history,
            score_p25=score_p25_history,
            score_p50=score_p50_history,
            score_p75=score_p75_history,
            score_p90=score_p90_history,
            score_p95=score_p95_history,
            score_p98=score_p98_history,
        )
        return -bestScore_np, history

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
