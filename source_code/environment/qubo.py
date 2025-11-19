from utils.walsh_expansion import WalshExpansion
from tqdm import tqdm
import os
import torch
from random import sample
import numpy as np

try:
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    tk = None
    FigureCanvasTkAgg = None
    plt = None



def get_Score_trajectoriesQUBO_cuda(strategy, N, nb_instances, nb_restarts, budget, size_pop, tensor_Q, device, verbose , name_file):

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

        bestGlobalSolution = bestGlobalSolution.unsqueeze(1).repeat(1,10,1)

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

    if track_leader and avg_hamming_history and avg_kl_history:
        iterations = [(idx + 1) * size_pop for idx in range(len(avg_hamming_history))]
        num_agents = len(strategy.agents) if hasattr(strategy, "agents") else 0
        _render_agent_plots(iterations, avg_hamming_history, avg_kl_history, agent_fitness_history, num_agents)

    svgd_snapshot_fn = getattr(strategy, "get_svgd_field_snapshot", None)
    if callable(svgd_snapshot_fn):
        snapshot = svgd_snapshot_fn()
        if snapshot:
            _render_svgd_field_plot(snapshot)

    theta_history_fn = getattr(strategy, "get_theta_history", None)
    if callable(theta_history_fn):
        theta_history = theta_history_fn()
        if theta_history and theta_history.get("values"):
            _render_theta_slider(theta_history)

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


def _render_agent_plots(iterations, hamming_history, kl_history, agent_fitness_history, num_agents):
    if tk is None or plt is None or FigureCanvasTkAgg is None:
        print("Tkinter/matplotlib not available, skipping diversity plot.")
        return

    try:
        rows = 3 if agent_fitness_history and num_agents > 0 else 2
        root = tk.Tk()
        root.title("Agent Metrics")
        fig, axes = plt.subplots(rows, 1, figsize=(8, 4 * rows), sharex=True)
        if rows == 2:
            axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes

        axes[0].plot(iterations, hamming_history, color="tab:blue")
        axes[0].set_title("Average Hamming Distance")
        axes[0].set_ylabel("Hamming")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].plot(iterations, kl_history, color="tab:orange")
        axes[1].set_title("Average KL Distance")
        axes[1].set_ylabel("KL")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        if rows == 3:
            axes[2].set_title("Agent Fitness Evolution")
            axes[2].set_ylabel("Fitness")
            for agent_idx in range(num_agents):
                series = [epoch[agent_idx] for epoch in agent_fitness_history]
                axes[2].plot(iterations, series, label=f"Agent {agent_idx}")
            axes[2].grid(True, linestyle="--", alpha=0.4)
            axes[2].legend()

        axes[-1].set_xlabel("Evaluations")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def _close():
            root.quit()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _close)
        root.mainloop()
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - GUI failure is non critical
        print(f"Failed to render Tkinter plots: {exc}")


def _render_svgd_field_plot(snapshot):
    if tk is None or plt is None or FigureCanvasTkAgg is None:
        print("Tkinter/matplotlib not available, skipping SVGD field plot.")
        return

    theta = snapshot.get("theta")
    phi = snapshot.get("phi")
    dims = snapshot.get("dims", (0, 1))
    if theta is None or phi is None:
        return

    theta = torch.tensor(theta) if isinstance(theta, np.ndarray) else theta
    phi = torch.tensor(phi) if isinstance(phi, np.ndarray) else phi
    num_instances = theta.shape[0]
    num_agents = theta.shape[1]

    try:
        root = tk.Tk()
        root.title("SVGD Field Snapshot")
        fig, axes = plt.subplots(1, num_instances, figsize=(5 * num_instances, 5), squeeze=False)
        axes = axes.flatten()
        colors = plt.cm.get_cmap("tab10", num_agents)

        for inst_idx in range(num_instances):
            ax = axes[inst_idx]
            ax.set_title(f"Instance {inst_idx}")
            ax.set_xlabel(f"theta[{dims[0]}]")
            ax.set_ylabel(f"theta[{dims[1]}]")
            ax.grid(True, linestyle="--", alpha=0.3)
            for agent_idx in range(num_agents):
                x, y = theta[inst_idx, agent_idx].tolist()
                dx, dy = phi[inst_idx, agent_idx].tolist()
                color = colors(agent_idx)
                ax.scatter(x, y, color=color, label=f"Agent {agent_idx}" if inst_idx == 0 else None)
                ax.arrow(
                    x,
                    y,
                    dx,
                    dy,
                    color=color,
                    head_width=0.02,
                    head_length=0.02,
                    length_includes_head=True,
                    alpha=0.8,
                )

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def _close():
            root.quit()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _close)
        root.mainloop()
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to render SVGD field plot: {exc}")


def _render_theta_slider(history):
    if tk is None or plt is None or FigureCanvasTkAgg is None:
        print("Tkinter/matplotlib not available, skipping theta slider plot.")
        return

    values = history.get("values") or []
    if not values:
        return

    num_agents = len(values[0])
    if num_agents == 0:
        return

    sample = values[0][0]
    num_instances = sample.shape[0]
    num_dims = sample.shape[1]

    def _sym_kl(p, q):
        eps = 1e-8
        p = torch.clamp(p, eps, 1 - eps)
        q = torch.clamp(q, eps, 1 - eps)
        kl_pq = p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))
        kl_qp = q * (torch.log(q) - torch.log(p)) + (1 - q) * (torch.log(1 - q) - torch.log(1 - p))
        return 0.5 * (kl_pq.mean() + kl_qp.mean())

    try:
        root = tk.Tk()
        root.title("Theta Evolution Explorer")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
        axes = axes.flatten()
        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.4)

        scatters = [axes[0].scatter([], [], color="tab:blue"), axes[1].scatter([], [], color="tab:orange")]

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=10, pady=6)

        epoch_var = tk.IntVar(value=0)
        agent_a_var = tk.IntVar(value=0)
        agent_b_var = tk.IntVar(value=min(1, num_agents - 1))
        instance_var = tk.IntVar(value=0)
        dim_x_var = tk.IntVar(value=0)
        dim_y_var = tk.IntVar(value=1 if num_dims > 1 else 0)

        def clamp(var, upper):
            val = max(0, min(upper, var.get()))
            var.set(val)
            return val

        status_var = tk.StringVar()
        status_label = tk.Label(root, textvariable=status_var)
        status_label.pack(pady=2)
        kl_var = tk.StringVar()
        kl_label = tk.Label(root, textvariable=kl_var)
        kl_label.pack(pady=2)

        def update_plot(*_):
            epoch_idx = clamp(epoch_var, len(values) - 1)
            inst_idx = clamp(instance_var, num_instances - 1)
            dx = clamp(dim_x_var, num_dims - 1)
            dy = clamp(dim_y_var, num_dims - 1)
            agent_indices = [
                clamp(agent_a_var, num_agents - 1),
                clamp(agent_b_var, num_agents - 1),
            ]

            for ax, scatter, agent_idx in zip(axes, scatters, agent_indices):
                probs = values[epoch_idx][agent_idx]
                x = float(probs[inst_idx, dx].item())
                y = float(probs[inst_idx, dy].item())
                scatter.set_offsets([[x, y]])
                ax.set_title(f"Agent {agent_idx} – Instance {inst_idx} – dims ({dx},{dy})")
            status_var.set(f"Epoch {epoch_idx + 1}/{len(values)}")
            if agent_indices[0] != agent_indices[1]:
                p = values[epoch_idx][agent_indices[0]][inst_idx]
                q = values[epoch_idx][agent_indices[1]][inst_idx]
                kl_val = _sym_kl(p, q).item()
                kl_var.set(f"Instance KL (Agent {agent_indices[0]} vs {agent_indices[1]}): {kl_val:.4f}")
            else:
                kl_var.set("Instance KL: n/a (same agent)")
            canvas.draw_idle()

        def labeled_spinbox(parent, text, var, upper, width=5):
            frame = tk.Frame(parent)
            frame.pack(side="left", padx=4)
            tk.Label(frame, text=text).pack()
            spin = tk.Spinbox(
                frame,
                from_=0,
                to=max(0, upper),
                textvariable=var,
                width=width,
                command=update_plot,
            )
            spin.pack()
            var.trace_add("write", lambda *args: update_plot())
            return spin

        tk.Label(controls, text="Agent A").pack(side="left", padx=4)
        agent_options = [str(i) for i in range(num_agents)]
        agent_menu_a = tk.OptionMenu(controls, agent_a_var, *agent_options, command=lambda *_: update_plot())
        agent_menu_a.pack(side="left", padx=4)

        tk.Label(controls, text="Agent B").pack(side="left", padx=4)
        agent_menu_b = tk.OptionMenu(controls, agent_b_var, *agent_options, command=lambda *_: update_plot())
        agent_menu_b.pack(side="left", padx=4)

        labeled_spinbox(controls, "Instance", instance_var, num_instances - 1)
        labeled_spinbox(controls, "Dim X", dim_x_var, num_dims - 1)
        labeled_spinbox(controls, "Dim Y", dim_y_var, num_dims - 1)

        slider = tk.Scale(
            root,
            from_=0,
            to=len(values) - 1,
            orient="horizontal",
            length=450,
            command=lambda val: (epoch_var.set(int(float(val))), update_plot()),
            label="Epoch",
        )
        slider.pack(fill="x", padx=12, pady=6)

        update_plot()

        def _close():
            root.quit()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _close)
        root.mainloop()
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to render theta slider: {exc}")
