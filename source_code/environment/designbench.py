import numpy as np
import torch
from tqdm import tqdm
from environment.metrics import MetricsCalculator
from environment.visualization import render_agent_dashboard, render_svgd_field_plot


def get_Score_trajectories_designbench_cuda(
    task,
    strategy,
    nb_instances,
    budget,
    size_popEA,
    device,
    verbose,
    oracle_batch_size=2048,
    alphabet_size=20,
    enable_visualization=False,
    name_file=None,
    return_history=False,
    warm_start=False,
):
    """
    DesignBench online evaluation loop (GFP-v0).

    The strategy is updated online:
    sample -> evaluate with task.predict -> updateDistribution.
    """

    del size_popEA  # kept for compatibility with existing call signatures

    if nb_instances <= 0:
        raise ValueError("nb_instances must be >= 1.")

    strategy.reset_learned_parameters(nb_instances=nb_instances)
    if warm_start:
        init_fn = getattr(strategy, "initialize_from_dataset", None)
        x_attr = getattr(task, "x", None)
        if callable(init_fn) and x_attr is not None:
            ok = init_fn(x_attr)
            if verbose and ok:
                print("[WarmStart] initialized theta from task.x distribution")

    best_score = torch.full((nb_instances,), -float("inf"), device=device)
    size_pop = int(strategy.lambda_)
    if size_pop <= 0:
        raise ValueError("strategy.lambda_ must be > 0.")
    if oracle_batch_size is None or int(oracle_batch_size) <= 0:
        raise ValueError("oracle_batch_size must be >= 1.")
    oracle_batch_size = int(oracle_batch_size)

    nb_iterations = budget // size_pop
    stochastic_remainder = budget - (nb_iterations * size_pop)

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
    avg_js_history = []
    avg_l2_history = []
    agent_fitness_history = []
    hamming_pairwise_history = []
    js_pairwise_history = []
    l2_pairwise_history = []
    l1_pairwise_history = []
    entropy_agent_history = []
    avg_kernel_value_history = []
    avg_kernel_grad_history = []

    if name_file is not None:
        with open(name_file, "w", encoding="utf-8") as f_results:
            f_results.write(
                "runtime, mean, median, std, 2%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 98%\n"
            )

    oracle_fn = getattr(task, "predict", None)
    if not callable(oracle_fn):
        oracle_fn = getattr(task, "score", None)
    if not callable(oracle_fn):
        raise AttributeError(
            "DesignBench task has no callable oracle method. Expected task.predict(...) "
            "or task.score(...)."
        )
    oracle_state = {"input_mode": None}

    def _tokens_to_onehot(tokens: np.ndarray, num_classes: int) -> np.ndarray:
        eye = np.eye(num_classes, dtype=np.float32)
        return eye[tokens]

    def _call_oracle(flat_sequences: np.ndarray) -> np.ndarray:
        mode = oracle_state["input_mode"]
        if mode == "tokens":
            return np.asarray(oracle_fn(flat_sequences), dtype=np.float32)
        if mode == "onehot":
            return np.asarray(oracle_fn(_tokens_to_onehot(flat_sequences, num_classes=alphabet_size)), dtype=np.float32)

        # Auto-detect input format once, then reuse it.
        try:
            scores = np.asarray(oracle_fn(flat_sequences), dtype=np.float32)
            oracle_state["input_mode"] = "tokens"
            if verbose:
                print("[DesignBench] oracle_input_mode=tokens")
            return scores
        except Exception:
            scores = np.asarray(oracle_fn(_tokens_to_onehot(flat_sequences, num_classes=alphabet_size)), dtype=np.float32)
            oracle_state["input_mode"] = "onehot"
            if verbose:
                print("[DesignBench] oracle_input_mode=onehot")
            return scores

    def _detect_oracle_input_mode(flat_sequences: np.ndarray) -> None:
        if oracle_state["input_mode"] is not None:
            return

        x_attr = getattr(task, "x", None)
        if x_attr is not None:
            x_arr = np.asarray(x_attr)
            if x_arr.ndim == 3 and x_arr.shape[-1] == 20:
                oracle_state["input_mode"] = "onehot"
                if verbose:
                    print("[DesignBench] oracle_input_mode=onehot (inferred from task.x)")
                return
            if x_arr.ndim == 2:
                oracle_state["input_mode"] = "tokens"
                if verbose:
                    print("[DesignBench] oracle_input_mode=tokens (inferred from task.x)")
                return

        probe = flat_sequences[:1]
        try:
            np.asarray(oracle_fn(probe), dtype=np.float32)
            oracle_state["input_mode"] = "tokens"
            if verbose:
                print("[DesignBench] oracle_input_mode=tokens (detected)")
        except Exception:
            np.asarray(oracle_fn(_tokens_to_onehot(probe, num_classes=alphabet_size)), dtype=np.float32)
            oracle_state["input_mode"] = "onehot"
            if verbose:
                print("[DesignBench] oracle_input_mode=onehot (detected)")

    def _evaluate_population(tensor_solution):
        pop_size = tensor_solution.size(1)
        # (B, lambda, N, 1) -> (B*lambda, N)
        sequences = tensor_solution[..., 0].detach().cpu().numpy().astype(np.int64)
        sequences = np.clip(sequences, 0, max(0, int(alphabet_size) - 1))
        flat_sequences = sequences.reshape(-1, sequences.shape[-1])
        _detect_oracle_input_mode(flat_sequences)
        total_flat = flat_sequences.shape[0]
        if total_flat <= oracle_batch_size:
            scores = _call_oracle(flat_sequences)
        else:
            chunks = []
            starts = range(0, total_flat, oracle_batch_size)
            if verbose:
                starts = tqdm(
                    starts,
                    total=(total_flat + oracle_batch_size - 1) // oracle_batch_size,
                    leave=False,
                    desc="oracle",
                )
            for start in starts:
                end = min(start + oracle_batch_size, total_flat)
                chunks.append(_call_oracle(flat_sequences[start:end]))
            scores = np.concatenate(chunks, axis=0)
        scores = np.asarray(scores, dtype=np.float32).reshape(nb_instances, pop_size)
        return torch.from_numpy(scores).to(device=device, dtype=torch.float32)

    def _record_history(step, avg_hamming=0.0, avg_l1=0.0, avg_entropy=0.0):
        best_np = best_score.detach().cpu().numpy()
        mean = float(np.mean(best_np))
        median = float(np.percentile(best_np, 50))
        std = float(np.std(best_np))
        p2 = float(np.percentile(best_np, 2))
        p5 = float(np.percentile(best_np, 5))
        p10 = float(np.percentile(best_np, 10))
        p25 = float(np.percentile(best_np, 25))
        p75 = float(np.percentile(best_np, 75))
        p90 = float(np.percentile(best_np, 90))
        p95 = float(np.percentile(best_np, 95))
        p98 = float(np.percentile(best_np, 98))

        runtime_steps.append(int(step))
        best_fitness_history.append(mean)
        avg_hamming_history.append(float(avg_hamming))
        avg_l1_history.append(float(avg_l1))
        avg_entropy_history.append(float(avg_entropy))
        score_mean_history.append(mean)
        score_median_history.append(median)
        score_std_history.append(std)
        score_p2_history.append(p2)
        score_p5_history.append(p5)
        score_p10_history.append(p10)
        score_p25_history.append(p25)
        score_p50_history.append(median)
        score_p75_history.append(p75)
        score_p90_history.append(p90)
        score_p95_history.append(p95)
        score_p98_history.append(p98)

        if name_file is not None and step % 100 == 0:
            with open(name_file, "a", encoding="utf-8") as f_results:
                f_results.write(
                    f"{step},{mean},{median},{std},"
                    f"{p2},{p5},{p10},{p25},{median},"
                    f"{p75},{p90},{p95},{p98}\n"
                )

    use_tqdm = bool(verbose)
    pbar = tqdm(range(nb_iterations)) if use_tqdm else range(nb_iterations)
    collect_dashboard = bool(enable_visualization) and hasattr(strategy, "agents")
    metrics = MetricsCalculator() if collect_dashboard else None

    for epoch in pbar:
        tensor_solution = strategy.sample_solutions()
        tensor_score = _evaluate_population(tensor_solution)
        batch_score_std = float(torch.std(tensor_score).item())

        current_score = torch.max(tensor_score, dim=1).values
        best_score = torch.where(current_score > best_score, current_score, best_score)

        if hasattr(strategy, "decay_svgd_gamma"):
            strategy.decay_svgd_gamma(epoch, nb_iterations)
        strategy.updateDistribution(tensor_solution, tensor_score)

        step = (epoch + 1) * size_pop
        avg_hamming = 0.0
        avg_l1 = 0.0
        avg_entropy = 0.0
        if collect_dashboard and metrics is not None:
            avg_hamming, hamming_pairwise = metrics.compute_average_hamming(strategy.agents)
            avg_js, js_pairwise = metrics.compute_average_js(strategy.agents)
            avg_l1, l1_pairwise = metrics.compute_l1_distance(strategy.agents)
            avg_entropy, per_agent_entropy = metrics.compute_entropy(strategy.agents)
            _avg_l2, _pair_l2 = 0.0, None
            avg_js_history.append(float(avg_js))
            avg_l2_history.append(float(_avg_l2))
            hamming_pairwise_history.append(hamming_pairwise.tolist() if hamming_pairwise is not None else None)
            js_pairwise_history.append(js_pairwise.tolist() if js_pairwise is not None else None)
            l2_pairwise_history.append(_pair_l2.tolist() if _pair_l2 is not None else None)
            l1_pairwise_history.append(l1_pairwise.tolist() if l1_pairwise is not None else None)
            entropy_agent_history.append(per_agent_entropy if per_agent_entropy is not None else None)

            agent_lambdas = getattr(strategy, "agent_lambdas", None)
            per_agent_fitness = []
            if isinstance(agent_lambdas, (list, tuple)) and len(agent_lambdas) > 0:
                start_idx = 0
                for agent_lambda in agent_lambdas:
                    end_idx = min(start_idx + int(agent_lambda), tensor_score.size(1))
                    if end_idx <= start_idx:
                        break
                    agent_scores = tensor_score[:, start_idx:end_idx]
                    agent_best_values, _ = torch.max(agent_scores, dim=1)
                    per_agent_fitness.append(float(agent_best_values.mean().item()))
                    start_idx = end_idx
            agent_fitness_history.append(per_agent_fitness)

            kernel_stats_fn = getattr(strategy, "get_latest_kernel_metrics", None)
            kernel_stats = kernel_stats_fn() if callable(kernel_stats_fn) else None
            if kernel_stats:
                avg_kernel_value_history.append(float(kernel_stats.get("avg_kernel_value", 0.0)))
                avg_kernel_grad_history.append(float(kernel_stats.get("avg_kernel_grad", 0.0)))
            else:
                avg_kernel_value_history.append(0.0)
                avg_kernel_grad_history.append(0.0)

        _record_history(step, avg_hamming=avg_hamming, avg_l1=avg_l1, avg_entropy=avg_entropy)

        if use_tqdm:
            pbar.set_postfix(
                bestScore=float(torch.mean(best_score).item()),
                current_score=float(torch.mean(current_score).item()),
                score_std=batch_score_std,
            )

    # Strict budget handling when budget is not a multiple of lambda_.
    # We evaluate the remainder but do not call updateDistribution on a truncated population.
    if stochastic_remainder > 0:
        tensor_solution = strategy.sample_solutions()[:, :stochastic_remainder, :, :]
        tensor_score = _evaluate_population(tensor_solution)
        current_score = torch.max(tensor_score, dim=1).values
        best_score = torch.where(current_score > best_score, current_score, best_score)
        if collect_dashboard:
            avg_js_history.append(avg_js_history[-1] if avg_js_history else 0.0)
            avg_l2_history.append(avg_l2_history[-1] if avg_l2_history else 0.0)
            hamming_pairwise_history.append(hamming_pairwise_history[-1] if hamming_pairwise_history else None)
            js_pairwise_history.append(js_pairwise_history[-1] if js_pairwise_history else None)
            l2_pairwise_history.append(l2_pairwise_history[-1] if l2_pairwise_history else None)
            l1_pairwise_history.append(l1_pairwise_history[-1] if l1_pairwise_history else None)
            entropy_agent_history.append(entropy_agent_history[-1] if entropy_agent_history else None)
            agent_fitness_history.append(agent_fitness_history[-1] if agent_fitness_history else [])
            avg_kernel_value_history.append(avg_kernel_value_history[-1] if avg_kernel_value_history else 0.0)
            avg_kernel_grad_history.append(avg_kernel_grad_history[-1] if avg_kernel_grad_history else 0.0)
        _record_history(budget)

    if collect_dashboard and hasattr(strategy, "agents"):
        theta_history = None
        theta_history_fn = getattr(strategy, "get_theta_history", None)
        if callable(theta_history_fn):
            theta_history = theta_history_fn()
        num_agents = len(strategy.agents)
        render_agent_dashboard(
            runtime_steps,
            avg_hamming_history,
            avg_js_history,
            agent_fitness_history,
            num_agents,
            theta_history,
            None,
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

    best_np = best_score.detach().cpu().numpy()
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
        return best_np, history
    return best_np
