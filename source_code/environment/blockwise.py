import torch
import numpy as np
from tqdm import tqdm

from environment.visualization import render_agent_dashboard, render_svgd_field_plot
from environment.metrics import MetricsCalculator


def get_Score_trajectoriesBLOCK_cuda(
    strategy,
    N,
    block_size,
    nb_instances,
    nb_restarts,
    budget,
    size_pop,
    device,
    verbose,
    enable_visualization=True,
    return_history=False,
    dummy_blocks=0,
):
    size_pop = strategy.lambda_
    total_cases = nb_instances * nb_restarts

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if N % block_size != 0:
        raise ValueError(f"N={N} must be divisible by block_size={block_size}")
    num_blocks = N // block_size
    if dummy_blocks < 0:
        raise ValueError(f"dummy_blocks must be >= 0, got {dummy_blocks}")
    if dummy_blocks >= num_blocks:
        raise ValueError(
            f"dummy_blocks={dummy_blocks} must be less than num_blocks={num_blocks}"
        )
    # Les derniers blocs peuvent etre "dummy" et donc exclus du calcul du score.
    scoring_blocks = num_blocks - dummy_blocks
    if dummy_blocks == 0:
        dummy_range = "none"
    else:
        dummy_var_count = dummy_blocks * block_size
        dummy_start = N - dummy_var_count + 1
        dummy_end = N
        dummy_range = f"{dummy_start}-{dummy_end}"
    print(
        f"BLOCK problem | variables={N}, blocks={num_blocks}, block_size={block_size}, "
        f"dummy_blocks={dummy_blocks}, dummy_variables={dummy_range}"
    )
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
    num_agents = len(agent_lambdas) if isinstance(agent_lambdas, (list, tuple)) else 0
    if num_agents == 0 and hasattr(strategy, "agents"):
        num_agents = len(strategy.agents)
    per_agent_lambdas = None
    if isinstance(agent_lambdas, (list, tuple)) and agent_lambdas:
        per_agent_lambdas = list(agent_lambdas)
    elif num_agents > 0 and size_pop % num_agents == 0:
        per_agent_lambdas = [size_pop // num_agents for _ in range(num_agents)]
    valid_agent_partition = (
        per_agent_lambdas
        and num_agents == len(per_agent_lambdas)
        and sum(per_agent_lambdas) == size_pop
    )
    sample_hamming_history = [] if (enable_visualization and valid_agent_partition and num_agents > 1) else None
    sample_hamming_pairwise_history = [] if (enable_visualization and valid_agent_partition and num_agents > 1) else None

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

        tensor_binary = tensor_solution.squeeze(3)
        tensor_blocks = tensor_binary.reshape(total_cases, size_pop, num_blocks, block_size)
        # Score par bloc: proportion de 0/1, puis "purite" via max(p, 1-p).
        block_counts = tensor_blocks.sum(dim=3)
        block_proportions = block_counts / float(block_size)
        block_scores = torch.maximum(block_proportions, 1.0 - block_proportions)
        # Fitness par solution: moyenne sur les blocs utiles (les dummy sont ignores).
        tensor_score = block_scores[:, :, :scoring_blocks].mean(dim=2)

        sample_hamming_current = None
        if sample_hamming_history is not None:
            try:
                with torch.no_grad():
                    samples = tensor_solution.squeeze(3)  # (B, total_lambda, N)
                    start_idx = 0
                    best_per_agent = []
                    for agent_lambda in per_agent_lambdas:
                        end_idx = start_idx + agent_lambda
                        sub_samples = samples[:, start_idx:end_idx, :]
                        sub_scores = tensor_score[:, start_idx:end_idx]
                        idx = torch.argmax(sub_scores, dim=1)
                        idx_expand = idx[:, None, None].expand(-1, 1, N)
                        best = torch.gather(sub_samples, 1, idx_expand).squeeze(1)  # (B, N)
                        best_per_agent.append(best)
                        start_idx = end_idx
                    pairwise = torch.zeros(num_agents, num_agents, device="cpu")
                    for i in range(num_agents):
                        for j in range(num_agents):
                            diff = (best_per_agent[i] != best_per_agent[j]).float()
                            dist = diff.sum(dim=1).mean().item()
                            pairwise[i, j] = dist
                    avg = (pairwise.sum() - pairwise.diag().sum()) / (num_agents * (num_agents - 1))
                    sample_hamming_current = float(avg)
                    sample_hamming_pairwise_history.append(pairwise.numpy().tolist())
            except Exception:
                sample_hamming_pairwise_history.append(None)
                sample_hamming_current = None
            sample_hamming_history.append(sample_hamming_current if sample_hamming_current is not None else 0.0)

        # Fitness par instance: meilleur individu du pool.
        current_score = torch.max(tensor_score, dim=1).values

        # Meilleure solution courante (individu ayant la meilleure fitness).
        index_solution = torch.argmax(tensor_score, dim=1)
        index_solution = index_solution.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, N, 1)
        best_current_solution = torch.gather(tensor_solution, 1, index_solution).squeeze(3).squeeze(1)

        if epoch == 0:
            bestGlobalSolution = best_current_solution
        else:
            tmp_current_score = current_score.unsqueeze(1).repeat(1, N)
            tmp_bestScore = bestScore.unsqueeze(1).repeat(1, N)
            bestGlobalSolution = torch.where(tmp_current_score > tmp_bestScore, best_current_solution, bestGlobalSolution)

        bestScore = torch.where(current_score > bestScore, current_score, bestScore)
        if hasattr(strategy, "decay_svgd_gamma"):
            strategy.decay_svgd_gamma(epoch, nb_iterations)
        strategy.updateDistribution(tensor_solution, tensor_score)

        scores_np = bestScore.detach().cpu().numpy()
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

        # Agregation en scalaire pour le suivi (ex: moyenne/median selon MetricsCalculator).
        global_current = metrics.compute_fitness(current_score)
        global_best = metrics.compute_fitness(bestScore)
        best_fitness_history.append(global_best)
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
                agent_best_overall[idx] = torch.where(
                    agent_best_values > agent_best_overall[idx],
                    agent_best_values,
                    agent_best_overall[idx],
                )
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
            postfix = {"bestScore": global_best, "current_score": global_current}
            if track_leader and leader_idx is not None:
                postfix["leader"] = leader_idx
                postfix["avg_hamming"] = avg_hamming
                postfix["avg_js"] = avg_js
                if sample_hamming_current is not None:
                    postfix["sample_hamming"] = sample_hamming_current
            pbar.set_postfix(**postfix)

    bestScore_np = bestScore.detach().cpu().numpy()
    if track_leader and enable_visualization and agent_best_overall is not None and hasattr(strategy, "agents"):
        print("Per-agent summary:")
        for idx, agent in enumerate(strategy.agents):
            avg_best = torch.mean(agent_best_overall[idx]).item()
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
            sample_hamming_history=sample_hamming_history,
            sample_hamming_pairwise_history=sample_hamming_pairwise_history,
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
