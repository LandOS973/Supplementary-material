from utils.walsh_expansion import WalshExpansion
from tqdm import tqdm
import os
import torch
from random import sample
import numpy as np

from environment.visualization import render_agent_dashboard, render_svgd_field_plot



def get_Score_trajectoriesQUBO_cuda(strategy, N, nb_instances, nb_restarts, budget, size_pop, tensor_Q, device, verbose , name_file, enable_visualization=True):

    size_pop = strategy.lambda_

    # tensor_Q is expected to have shape (total_cases, N, N) where total_cases = nb_instances_found * nb_restarts
    # repeat to match population size
    tensor_Q = (tensor_Q.unsqueeze(1)).repeat([1, size_pop, 1, 1]).to(device)

    total_cases = tensor_Q.size(0)

    # Now initialize strategy and tracking tensors based on actual available cases
    strategy.reset_learned_parameters(total_cases)
    bestScore = torch.ones(total_cases).to(device) * (-99999)

    agent_lambdas = getattr(strategy, "agent_lambdas", None)
    track_leader = isinstance(agent_lambdas, (list, tuple)) and len(agent_lambdas) > 0
    agent_best_overall = None
    if track_leader:
        agent_best_overall = [torch.ones(total_cases).to(device) * (-99999) for _ in agent_lambdas]
    nb_iterations = budget // size_pop

    avg_hamming_history = []
    avg_kl_history = []
    agent_fitness_history = []

    if(verbose):
        pbar = tqdm(range(nb_iterations))
    else:
        pbar = range(nb_iterations)
        
        
    list_tensor_solution = []


    if(name_file is not None):
        f_results = open(name_file, "w")
        f_results.write("runtime, mean, median, std, 2%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 98%" + "\n")
        f_results.close()
    

    
    
    
    


    
    for epoch in pbar:

        tensor_solution = strategy.sample_solutions()

        if epoch == 0:
            startSolution = tensor_solution[:,0,:,:].squeeze(2)
        


        tensor_QUBO = tensor_solution*2 - 1

        Qx = tensor_Q @ tensor_QUBO

        tensor_score = -(torch.transpose(Qx, 2, 3) @ tensor_QUBO).squeeze(2).squeeze(2)  
        

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
        strategy.updateDistribution( tensor_solution, tensor_score)

        global_current = torch.mean(current_score).item()
        global_best = torch.mean(bestScore).item()

        leader_idx = None
        # avg hamming = sum of pairwise hamming distances between agent best solutions
        avg_hamming = None
        avg_kl = None
        if track_leader:
            agent_best_scores = []
            agent_best_solutions = []
            start_idx = 0
            for idx, agent_lambda in enumerate(agent_lambdas):
                end_idx = start_idx + agent_lambda
                agent_scores = tensor_score[:, start_idx:end_idx]
                agent_solutions = tensor_solution[:, start_idx:end_idx, :, :]
                agent_best_values, agent_best_idx = torch.max(agent_scores, dim=1)
                gather_idx = agent_best_idx.view(-1, 1, 1, 1).repeat(1, 1, N, 1)
                best_sol = torch.gather(agent_solutions, 1, gather_idx).squeeze(1).squeeze(-1)
                agent_best_scores.append(agent_best_values)
                agent_best_solutions.append(best_sol)
                agent_best_overall[idx] = torch.where(agent_best_values > agent_best_overall[idx],
                                                      agent_best_values,
                                                      agent_best_overall[idx])
                start_idx = end_idx

            agent_mean_scores = torch.stack([scores.mean() for scores in agent_best_scores])
            leader_idx = torch.argmax(agent_mean_scores).item()

            pairwise_distances = []
            for i in range(len(agent_best_solutions)):
                for j in range(i + 1, len(agent_best_solutions)):
                    dist = torch.abs(agent_best_solutions[i] - agent_best_solutions[j]).sum(dim=1).float()
                    pairwise_distances.append(dist)
            if pairwise_distances:
                stacked = torch.stack(pairwise_distances, dim=0)
                avg_hamming = torch.mean(stacked).item()
            else:
                avg_hamming = 0.0

            avg_kl = _compute_average_kl(strategy.agents)
            avg_hamming_history.append(avg_hamming if avg_hamming is not None else 0.0)
            avg_kl_history.append(avg_kl if avg_kl is not None else 0.0)
            agent_fitness_history.append([score.item() for score in agent_mean_scores])

        if(verbose):
           postfix = {"bestScore": -global_best, "current_score": -global_current}
           if track_leader and leader_idx is not None:
               postfix["leader"] = leader_idx
               postfix["avg_hamming"] = avg_hamming
               postfix["avg_kl"] = avg_kl
           pbar.set_postfix(**postfix)


        if(name_file is not None):
            if(((epoch +1)*size_pop) % 100 == 0):
                
                bestScore_np = -bestScore.cpu().numpy()               
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



    if name_file is not None:
        f_hamming = open(name_file + "_HD", "w")
        f_hamming.write("runtime, avg distance, avg std pop" + "\n")
        f_hamming.close()

        bestGlobalSolution = bestGlobalSolution.unsqueeze(1).repeat(1, size_pop, 1)

        for idx, tensor_solution in enumerate(list_tensor_solution):
            hamming_distance = torch.sum(torch.abs(tensor_solution.squeeze(3) - bestGlobalSolution), dim=2).cpu().numpy()
            avg_distance = np.mean(hamming_distance)
            avg_std_distance = np.mean(np.std(hamming_distance, axis = 1))

            f_hamming = open(name_file + "_HD", "a")
            f_hamming.write(str((idx + 1)*size_pop) + "," +  str(avg_distance) + "," +  str(avg_std_distance) + "\n")
            f_hamming.close()

    bestScore_np = -bestScore.detach().cpu().numpy()

    if track_leader and agent_best_overall is not None and hasattr(strategy, "agents"):
        print("Per-agent summary:")
        for idx, agent in enumerate(strategy.agents):
            avg_best = -torch.mean(agent_best_overall[idx]).item()
            theta_mean = torch.mean(agent.theta).item()
            print(f"Agent {idx}: avg_best_score={avg_best:.4f}, theta_mean={theta_mean:.6f}")

    if enable_visualization:
        iterations = [(idx + 1) * size_pop for idx in range(len(avg_hamming_history))] if avg_hamming_history else []
        num_agents = len(strategy.agents) if hasattr(strategy, "agents") else 0
        theta_history = None
        theta_history_fn = getattr(strategy, "get_theta_history", None)
        if callable(theta_history_fn):
            theta_history = theta_history_fn()
        render_agent_dashboard(iterations, avg_hamming_history, avg_kl_history, agent_fitness_history, num_agents, theta_history)

        svgd_snapshot_fn = getattr(strategy, "get_svgd_field_snapshot", None)
        if callable(svgd_snapshot_fn):
            snapshot = svgd_snapshot_fn()
            if snapshot:
                render_svgd_field_plot(snapshot)

    return bestScore_np




def getTensorInstances_QUBO(path, nb_instances, nb_restarts,  N, t, device, phase):

    list_matrix_Q = []
    list_matrix_K = []
    # Ensure path exists and discover available instance files matching pattern
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instances path not found: {path}")

    prefix = f"puboi_evo_n_{N}_t_{t}_i_"
    files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith('.json')]

    if len(files) == 0:
        raise FileNotFoundError(f"No QUBO instance files found in {path} with prefix {prefix}")

    # extract instance numbers and sort
    def inst_index(fname):
        try:
            part = fname[len(prefix):-5]  # strip prefix and .json
            return int(part)
        except Exception:
            return 0

    files_sorted = sorted(files, key=inst_index)

    # select up to nb_instances available files
    selected_files = files_sorted[:nb_instances]

    if len(selected_files) < nb_instances:
        print(f"Warning: requested {nb_instances} instances but only found {len(selected_files)} in {path}. Using {len(selected_files)} instances.")

    for fname in selected_files:
        filename = os.path.join(path, fname)
        f = WalshExpansion()
        f.load(filename)
        Q = f.to_symmetric_Q()

        Q_th = torch.tensor(Q, dtype=torch.float32)

        for i in range(nb_restarts):
            list_matrix_Q.append(Q_th)


    with torch.no_grad():

        tensor_Q = torch.stack(list_matrix_Q, dim=0)


    return tensor_Q


def _compute_average_kl(agents):
    if agents is None or len(agents) < 2:
        return 0.0

    eps = 1e-8
    total_pairwise_kl = 0.0
    comparisons = 0
    with torch.no_grad():
        agent_probs = [torch.sigmoid(agent.theta).detach() for agent in agents]

    for i in range(len(agent_probs)):
        for j in range(i + 1, len(agent_probs)):
            p = torch.clamp(agent_probs[i], eps, 1 - eps)
            q = torch.clamp(agent_probs[j], eps, 1 - eps)
            kl_pq_inst = (
                p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))
            ).mean(dim=1)  # moyenne par instance
            kl_qp_inst = (
                q * (torch.log(q) - torch.log(p)) + (1 - q) * (torch.log(1 - q) - torch.log(1 - p))
            ).mean(dim=1)
            kl_pair_inst = 0.5 * (kl_pq_inst + kl_qp_inst)
            total_pairwise_kl += kl_pair_inst.mean().item()
            comparisons += 1

    return (total_pairwise_kl / comparisons) if comparisons > 0 else 0.0
