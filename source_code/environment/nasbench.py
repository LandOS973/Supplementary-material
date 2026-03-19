import numpy as np
import torch
from tqdm import tqdm


def get_Score_trajectories_nasbench_cuda(
    objective,
    strategy,
    nb_instances,
    budget,
    size_popEA,
    device,
    verbose,
    name_file=None,
    return_history=False,
):
    """
    NasBench evaluation loop (categorical, 26 variables, one-hot with 3 classes).

    objective: callable that takes a (N, 3) one-hot numpy array and returns (evals, info).
    """

    strategy.reset_learned_parameters(nb_instances)

    bestScore = torch.ones(nb_instances, device=device) * (-99999)
    size_pop = strategy.lambda_
    nb_iterations = budget // size_pop

    use_tqdm = bool(verbose)
    pbar = tqdm(range(nb_iterations)) if use_tqdm else range(nb_iterations)

    runtime_steps = []
    best_fitness_history = []
    avg_hamming_history = []
    avg_l1_history = []
    avg_entropy_history = []
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

    if name_file is not None:
        with open(name_file, "w", encoding="utf-8") as f_results:
            f_results.write(
                "runtime, mean, median, std, 2%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 98%\n"
            )

    for epoch in pbar:
        tensor_solution = strategy.sample_solutions()  # (B, lambda, N, 1)
        tensor_solution_oh = torch.nn.functional.one_hot(
            tensor_solution.squeeze(-1).long(), num_classes=3
        ).float()
        tensor_solution_cpu = tensor_solution_oh.cpu().numpy()

        tensor_score = torch.zeros((nb_instances, size_pop), device=device)
        for i in range(nb_instances):
            for j in range(size_pop):
                solution = tensor_solution_cpu[i, j].astype(np.float32)
                evals, _info = objective(solution)
                tensor_score[i, j] = -float(evals[0])

        current_score = torch.max(tensor_score, dim=1).values
        bestScore = torch.where(current_score > bestScore, current_score, bestScore)

        if hasattr(strategy, "decay_svgd_gamma"):
            strategy.decay_svgd_gamma(epoch, nb_iterations)
        strategy.updateDistribution(tensor_solution, tensor_score)

        bestScore_np = bestScore.detach().cpu().numpy()
        mean = float(np.mean(bestScore_np))
        median = float(np.percentile(bestScore_np, 50))
        std = float(np.std(bestScore_np))
        _2per = float(np.percentile(bestScore_np, 2))
        _5per = float(np.percentile(bestScore_np, 5))
        _10per = float(np.percentile(bestScore_np, 10))
        _25per = float(np.percentile(bestScore_np, 25))
        _75per = float(np.percentile(bestScore_np, 75))
        _90per = float(np.percentile(bestScore_np, 90))
        _95per = float(np.percentile(bestScore_np, 95))
        _98per = float(np.percentile(bestScore_np, 98))

        runtime_steps.append((epoch + 1) * size_pop)
        best_fitness_history.append(mean)
        avg_hamming_history.append(0.0)
        avg_l1_history.append(0.0)
        avg_entropy_history.append(0.0)
        score_mean_history.append(mean)
        score_median_history.append(median)
        score_std_history.append(std)
        score_p2_history.append(_2per)
        score_p5_history.append(_5per)
        score_p10_history.append(_10per)
        score_p25_history.append(_25per)
        score_p50_history.append(median)
        score_p75_history.append(_75per)
        score_p90_history.append(_90per)
        score_p95_history.append(_95per)
        score_p98_history.append(_98per)

        if verbose:
            pbar.set_postfix(
                bestScore=torch.mean(bestScore).item(),
                current_score=torch.mean(current_score).item(),
            )

        if name_file is not None and ((epoch + 1) * size_pop) % 100 == 0:
            with open(name_file, "a", encoding="utf-8") as f_results:
                f_results.write(
                    f"{(epoch + 1) * size_pop},{mean},{median},{std},"
                    f"{_2per},{_5per},{_10per},{_25per},{median},"
                    f"{_75per},{_90per},{_95per},{_98per}\n"
                )

    bestScore_np = bestScore.detach().cpu().numpy()
    if return_history:
        history = dict(
            runtime=runtime_steps,
            best_fitness=best_fitness_history,
            avg_hamming=avg_hamming_history,
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
