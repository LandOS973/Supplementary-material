from utils.walsh_expansion import WalshExpansion
from tqdm import tqdm
import os
import numpy as np
import torch

from environment.visualization import render_agent_dashboard, render_svgd_field_plot
from environment.metrics import MetricsCalculator



def get_Score_trajectoriesQUBO_cuda(
    strategy,
    N,
    nb_instances,
    nb_restarts,
    budget,
    size_pop,
    tensor_Q,
    device,
    verbose,
    enable_visualization=True,
    return_history=False,
):

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
    collect_summary_metrics = track_leader
    collect_pairwise_metrics = track_leader and bool(enable_visualization)
    agent_best_overall = None
    if track_leader:
        agent_best_overall = [torch.ones(total_cases).to(device) * (-99999) for _ in agent_lambdas]
    nb_iterations = budget // size_pop

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
    solutions_history = [] if enable_visualization else None
    metrics = MetricsCalculator()

    use_tqdm = bool(verbose)
    pbar = tqdm(range(nb_iterations)) if use_tqdm else range(nb_iterations)
        
        


    

    
    
    
    


    
    for epoch in pbar:

        tensor_solution = strategy.sample_solutions()
        if solutions_history is not None:
            try:
                sample_first = tensor_solution[0, :, :, 0].detach().cpu().numpy().astype(np.uint8)
            except Exception:
                sample_first = None
            solutions_history.append(sample_first)

        if epoch == 0:
            startSolution = tensor_solution[:,0,:,:].squeeze(2)
        


        tensor_QUBO = tensor_solution*2 - 1

        Qx = tensor_Q @ tensor_QUBO

        tensor_score = -(torch.transpose(Qx, 2, 3) @ tensor_QUBO).squeeze(2).squeeze(2)  
        

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
        if hasattr(strategy, "decay_svgd_gamma"):
            strategy.decay_svgd_gamma(epoch, nb_iterations)
        strategy.updateDistribution(tensor_solution, tensor_score)

        scores_np = -bestScore.detach().cpu().numpy()
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
        best_fitness_history.append(-global_best)
        runtime_steps.append((epoch + 1) * size_pop)

        leader_idx = None
        avg_hamming = None
        avg_js = None
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

        if use_tqdm:
           postfix = {"bestScore": -global_best, "current_score": -global_current}
           if track_leader and leader_idx is not None:
               postfix["leader"] = leader_idx
               postfix["avg_hamming"] = avg_hamming
               postfix["avg_js"] = avg_js
           pbar.set_postfix(**postfix)





    bestScore_np = -bestScore.detach().cpu().numpy()

    if track_leader and enable_visualization and agent_best_overall is not None and hasattr(strategy, "agents"):
        print("Per-agent summary:")
        for idx, agent in enumerate(strategy.agents):
            avg_best = -torch.mean(agent_best_overall[idx]).item()
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
            {"values": solutions_history, "lambda_per_agent": size_pop // max(num_agents, 1)}
            if solutions_history is not None and num_agents > 0
            else None,
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
        return bestScore_np, history

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
