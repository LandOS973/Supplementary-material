import os
import numpy as np
import torch
from collections import OrderedDict

_HEADLESS = not (
    os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
    or os.environ.get("MPLBACKEND")
)

try:
    if _HEADLESS:
        raise RuntimeError("Headless environment detected")
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
except Exception:                                        
    tk = None
    FigureCanvasTkAgg = None
    plt = None
    _HEADLESS = True


def render_agent_dashboard(
    iterations,
    hamming_history,
    js_history,
    agent_fitness_history,
    num_agents,
    theta_history,
    solutions_history=None,
    hamming_pairwise_history=None,
    js_pairwise_history=None,
    entropy_history=None,
    entropy_agent_history=None,
    kernel_value_history=None,
    kernel_grad_history=None,
    sample_entropy_history=None,
    sample_entropy_agent_history=None,
    sample_hamming_history=None,
    sample_hamming_pairwise_history=None,
    score_history=None,
    ranking_lines=None,
    l1_history=None,
    l1_pairwise_history=None,
    best_individual_history=None,
    attraction_agent_history=None,
    repulsion_agent_history=None,
):
    if tk is None or plt is None or FigureCanvasTkAgg is None:
        print("Tkinter/matplotlib not available, skipping dashboard.")
        return

    def _prepare_pairwise(history):
        if not history:
            return None
        cleaned = []
        for step in history:
            if step is None:
                return None
            cleaned.append(np.asarray(step, dtype=np.float32))
        arr = np.asarray(cleaned, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] != num_agents:
            return None
        return arr

    def _prepare_agent_series(history):
        if not history:
            return None
        cleaned = []
        for step in history:
            if step is None:
                return None
            cleaned.append(np.asarray(step, dtype=np.float32))
        arr = np.asarray(cleaned, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != num_agents:
            return None
        return arr

    pairwise_hamming = _prepare_pairwise(hamming_pairwise_history)
    pairwise_js = _prepare_pairwise(js_pairwise_history)
    pairwise_l1 = _prepare_pairwise(l1_pairwise_history)
    entropy_agent_series = _prepare_agent_series(entropy_agent_history)
    sample_entropy_agent_series = _prepare_agent_series(sample_entropy_agent_history)
    sample_hamming_pairwise = _prepare_pairwise(sample_hamming_pairwise_history)

    metrics_data = OrderedDict()
    if iterations and hamming_history:
        metrics_data["Hamming"] = dict(
            average=hamming_history,
            ylabel="Hamming",
            title="Average / Pairwise Hamming Distance",
            color="tab:blue",
            overlay_type="pairwise",
            overlay_data=pairwise_hamming,
        )
    if iterations and js_history:
        metrics_data["JS"] = dict(
            average=js_history,
            ylabel="JS",
            title="Average Jensen-Shannon Distance",
            color="tab:orange",
            overlay_type="pairwise",
            overlay_data=pairwise_js,
        )
    if iterations and score_history:
        metrics_data["Score"] = dict(
            average=score_history,
            ylabel="Score",
            title="Population Best Score",
            color="tab:red",
            overlay_type=None,
            overlay_data=None,
        )
    if iterations and l1_history:
        metrics_data["L1"] = dict(
            average=l1_history,
            ylabel="L1",
            title="Average L1 Distance",
            color="tab:pink",
            overlay_type="pairwise",
            overlay_data=pairwise_l1,
        )
    if entropy_history and entropy_agent_series is not None:
        metrics_data["Entropy"] = dict(
            average=entropy_history,
            ylabel="Entropy",
            title="Average Entropy",
            color="tab:green",
            overlay_type="per_agent",
            overlay_data=entropy_agent_series,
        )
    if sample_entropy_history and sample_entropy_agent_series is not None:
        metrics_data["Sample Entropy"] = dict(
            average=sample_entropy_history,
            ylabel="Sample Entropy",
            title="Sample Entropy (Samples)",
            color="tab:olive",
            overlay_type="per_agent",
            overlay_data=sample_entropy_agent_series,
        )
    if sample_hamming_history and sample_hamming_pairwise is not None:
        metrics_data["Sample Hamming"] = dict(
            average=sample_hamming_history,
            ylabel="Sample Hamming",
            title="Sample Hamming (Best Samples)",
            color="tab:cyan",
            overlay_type="pairwise",
            overlay_data=sample_hamming_pairwise,
        )
    metric_names = list(metrics_data.keys())

    try:
        root = tk.Tk()
        root.title("Agent Dashboard")
        try:
            root.state("zoomed")
        except Exception:
            root.attributes("-zoomed", True)

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        pane = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        pane.pack(fill="both", expand=True)

        metrics_frame = tk.Frame(pane)
        pane.add(metrics_frame, stretch="always")

        if ranking_lines:
            ranking_frame = tk.LabelFrame(metrics_frame, text="Classement global")
            ranking_frame.pack(fill="x", anchor="n", padx=4, pady=(4, 0))
            tk.Label(
                ranking_frame,
                text="\n".join(ranking_lines),
                justify="left",
                font=("Courier", 9),
                anchor="w",
            ).pack(fill="x", padx=6, pady=4)

        extra_series_config = OrderedDict()
        if agent_fitness_history and num_agents > 0:
            extra_series_config["Fitness"] = dict(
                history=agent_fitness_history, ylabel="Fitness", title="Agent Fitness Evolution"
            )
        extra_series_vars = {name: tk.IntVar(value=1) for name in extra_series_config}

        theta_available = bool(theta_history and theta_history.get("values"))
        hamming_evo_series, hamming_evo_num_instances = (
            _compute_agent_hamming_evolution(theta_history) if theta_available else (None, 0)
        )
        hamming_evo_available = bool(iterations and hamming_evo_series is not None and num_agents > 0)
        show_hamming_evo_var = tk.IntVar(value=1 if hamming_evo_available else 0)

        best_individual_hamming = best_individual_history.get("hamming") if best_individual_history else None
        best_individual_available = best_individual_hamming is not None and getattr(best_individual_hamming, "size", 0) > 0
        show_instance_column = theta_available or best_individual_available
        theta_var = tk.IntVar(value=1 if show_instance_column else 0)
        theta_panel = None
        theta_pack_info = None
        theta_container = None
        pane_theta_width = max(400, root.winfo_screenwidth() // 5)
        if show_instance_column:
            theta_container = tk.Frame(pane, width=pane_theta_width)
            pane.add(theta_container)
            pane.paneconfigure(theta_container, minsize=pane_theta_width // 2)

            shared_num_instances = 0
            if best_individual_available:
                shared_num_instances = int(best_individual_hamming.shape[0])
            elif theta_available:
                first_agent = theta_history["values"][0][0]
                first_arr = (
                    first_agent.detach().cpu().numpy() if hasattr(first_agent, "detach") else np.asarray(first_agent)
                )
                shared_num_instances = int(first_arr.shape[0]) if first_arr.ndim >= 1 else 0

            shared_instance_var = tk.IntVar(value=0)
            shared_average_var = tk.IntVar(value=0)
            if shared_num_instances > 0:
                instance_controls = tk.Frame(theta_container)
                instance_controls.pack(side="top", fill="x", padx=10, pady=(6, 2))
                tk.Label(instance_controls, text="Instance:").pack(side="left", padx=(0, 4))
                instance_menu = tk.OptionMenu(instance_controls, shared_instance_var, *range(shared_num_instances))
                instance_menu.pack(side="left", padx=(0, 12))
                tk.Checkbutton(instance_controls, text="Moyenne", variable=shared_average_var).pack(side="left")

                def _toggle_instance_menu(*_):
                    instance_menu.configure(state="disabled" if shared_average_var.get() else "normal")

                shared_average_var.trace_add("write", _toggle_instance_menu)

            if best_individual_available:
                _build_best_individual_table_panel(
                    theta_container, best_individual_history, shared_instance_var, shared_average_var
                )
            if theta_available:
                theta_panel = _build_probs_heatmap_panel(
                    theta_container, root, theta_history, num_agents, shared_instance_var, shared_average_var
                )
            if theta_panel:
                theta_pack_info = theta_panel.pack_info()
                if theta_var.get() == 0:
                    pane.forget(theta_container)

        button_container = tk.Frame(metrics_frame)
        button_container.pack(fill="x", anchor="n")
        graph_container = tk.Frame(metrics_frame)
        graph_container.pack(fill="both", expand=True)

        fig = plt.Figure(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=graph_container)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, graph_container)
        toolbar.update()
        toolbar.pack(side="bottom", anchor="se")

        metric_vars = {}
        selected_metrics = []
        metrics_order = list(metrics_data.keys())

        agent_labels = [f"Agent {idx}" for idx in range(num_agents)]
        agent_options = ["Moyenne"] + agent_labels if agent_labels else ["Moyenne"]
        selected_agent = tk.StringVar(value=agent_options[0])

        plot_handles = {}
        overlay_lines = {}
        color_map = plt.cm.get_cmap("tab10", max(num_agents, 1))

        def enabled_extra_series():
            return [name for name in extra_series_config if extra_series_vars[name].get() == 1]

        extra_axes = {}
        extra_lines = {}

        def draw_extra_axis(name, total_rows, row):
            cfg = extra_series_config[name]
            ax = fig.add_subplot(total_rows, 1, row)
            ax.set_title(cfg["title"])
            ax.set_ylabel(cfg["ylabel"])
            lines = []
            if iterations:
                for agent_idx in range(num_agents):
                    series = [epoch[agent_idx] for epoch in cfg["history"]]
                    (line,) = ax.plot(iterations, series, label=f"Agent {agent_idx}")
                    lines.append((agent_idx, line))
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            return ax, lines

        def is_hamming_evo_enabled():
            return hamming_evo_available and show_hamming_evo_var.get() == 1

        hamming_evo_ax = None

        def draw_hamming_evo_axis(total_rows, row):
            if not is_hamming_evo_enabled():
                return None
            ax = fig.add_subplot(total_rows, 1, row)
            if shared_average_var.get():
                series = hamming_evo_series.mean(axis=-1)
                label_suffix = f"moyenne sur {hamming_evo_num_instances} instances"
            else:
                idx = max(0, min(hamming_evo_num_instances - 1, shared_instance_var.get()))
                series = hamming_evo_series[:, :, idx]
                label_suffix = f"instance {idx}"
            steps = min(len(iterations), series.shape[0])
            x_axis = iterations[:steps]
            for agent_idx in range(num_agents):
                ax.plot(x_axis, series[:steps, agent_idx], label=f"Agent {agent_idx}")
            ax.set_title(f"Hamming Evolution par agent ({label_suffix})")
            ax.set_ylabel("Hamming vs autres agents")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize="x-small", ncol=min(num_agents, 4))
            return ax

        def draw_metrics():
            nonlocal hamming_evo_ax
            fig.clear()
            plot_handles.clear()
            overlay_lines.clear()
            extra_axes.clear()
            extra_lines.clear()
            active_extra = enabled_extra_series()
            total_metric_rows = len(selected_metrics)
            total_rows = (
                total_metric_rows
                + len(active_extra)
                + (1 if is_hamming_evo_enabled() else 0)
            )
            if total_rows == 0:
                fig.text(0.5, 0.5, "No metrics to display", ha="center", va="center")
                canvas.draw_idle()
                return
            row = 1
            for metric_name in selected_metrics:
                data = metrics_data[metric_name]
                ax = fig.add_subplot(total_rows, 1, row)
                row += 1
                avg_line = None
                avg_values = data.get("average")
                if iterations and avg_values:
                    (avg_line,) = ax.plot(iterations, avg_values, color=data["color"], label="Average")
                ax.set_title(data["title"])
                ax.set_ylabel(data["ylabel"])
                ax.grid(True, linestyle="--", alpha=0.4)
                plot_handles[metric_name] = dict(axis=ax, avg_line=avg_line)
                overlay_lines[metric_name] = []

            last_axis = None
            for name in active_extra:
                ax, lines = draw_extra_axis(name, total_rows, row)
                row += 1
                extra_axes[name] = ax
                extra_lines[name] = lines
                last_axis = ax
            hamming_evo_ax = draw_hamming_evo_axis(total_rows, row)
            if hamming_evo_ax is not None:
                row += 1
                last_axis = hamming_evo_ax
            if last_axis is None and selected_metrics:
                last_axis = plot_handles[selected_metrics[-1]]["axis"]
            if last_axis is not None:
                last_axis.set_xlabel("Evaluations")
            fig.tight_layout()
            canvas.draw_idle()
            update_overlays()

        def _toggle_metric(metric_name):
            nonlocal selected_metrics
            selected = [name for name in metrics_order if metric_vars.get(name, tk.IntVar()).get()]
            if not selected:
                if metric_name in metrics_order:
                    metric_vars[metric_name].set(1)
                    selected = [metric_name]
                elif metrics_order:
                    first = metrics_order[0]
                    metric_vars[first].set(1)
                    selected = [first]
            selected_metrics = selected
            draw_metrics()

        def update_overlays(*_):
            agent_choice = selected_agent.get()
            show_average = agent_choice == "Moyenne"
            anchor_idx = None
            if not show_average:
                try:
                    anchor_idx = agent_options.index(agent_choice) - 1
                except ValueError:
                    anchor_idx = -1
                if anchor_idx < 0:
                    anchor_idx = 0
            for metric_name in selected_metrics:
                handles = plot_handles.get(metric_name)
                if handles is None:
                    continue
                axis = handles["axis"]
                avg_line = handles.get("avg_line")
                for line in overlay_lines.get(metric_name, []):
                    try:
                        line.remove()
                    except ValueError:
                        pass
                overlay_lines[metric_name] = []
                data = metrics_data[metric_name]
                overlay_type = data.get("overlay_type")
                overlay_data = data.get("overlay_data")
                overlay_enabled = (
                    overlay_data is not None and overlay_type is not None and anchor_idx is not None and not show_average
                )
                if avg_line:
                    avg_line.set_visible(not overlay_enabled or show_average)
                if overlay_enabled and anchor_idx is not None:
                    if overlay_type == "pairwise":
                        steps = min(len(iterations), overlay_data.shape[0])
                        if steps > 0:
                            x_axis = iterations[:steps]
                            for other_idx in range(num_agents):
                                if other_idx == anchor_idx:
                                    continue
                                series = overlay_data[:steps, anchor_idx, other_idx]
                                (line,) = axis.plot(
                                    x_axis,
                                    series,
                                    linestyle="--",
                                    color=color_map(other_idx % color_map.N),
                                    label=f"{metric_name}: Agent {anchor_idx} ↔ {other_idx}",
                                )
                                overlay_lines[metric_name].append(line)
                    elif overlay_type == "per_agent":
                        steps = min(len(iterations), overlay_data.shape[0])
                        if steps > 0:
                            x_axis = iterations[:steps]
                            series = overlay_data[:steps, anchor_idx]
                            (line,) = axis.plot(
                                x_axis,
                                series,
                                linestyle="--",
                                color=color_map(anchor_idx % color_map.N),
                                label=f"{metric_name}: Agent {anchor_idx}",
                            )
                            overlay_lines[metric_name].append(line)

                legend_handles = []
                legend_labels = []
                if avg_line and avg_line.get_visible():
                    legend_handles.append(avg_line)
                    legend_labels.append("Average")
                for line in overlay_lines.get(metric_name, []):
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())
                if legend_handles:
                    axis.legend(legend_handles, legend_labels, loc="upper right")
                else:
                    leg = axis.get_legend()
                    if leg:
                        leg.remove()
            for name, lines in extra_lines.items():
                ax = extra_axes.get(name)
                if not lines or ax is None:
                    continue
                if show_average or anchor_idx is None:
                    for _, line in lines:
                        line.set_visible(True)
                else:
                    for idx, line in lines:
                        line.set_visible(idx == anchor_idx)
                visible = [(line, line.get_label()) for _, line in lines if line.get_visible()]
                if visible:
                    handles_vis, labels_vis = zip(*visible)
                    ax.legend(handles_vis, labels_vis, loc="upper right")
                else:
                    leg = ax.get_legend()
                    if leg:
                        leg.remove()

            canvas.draw_idle()

        def _toggle_theta_panel():
            if not theta_container:
                return
            if theta_var.get():
                pane_children = pane.panes()
                if str(theta_container) not in pane_children:
                    pane.add(theta_container)
                    pane.paneconfigure(theta_container, minsize=pane_theta_width // 2)
            else:
                try:
                    pane.forget(theta_container)
                except tk.TclError:
                    pass

        agent_labels = [f"Agent {idx}" for idx in range(num_agents)]
        agent_options = ["Moyenne"] + agent_labels if agent_labels else ["Moyenne"]
        selected_agent = tk.StringVar(value=agent_options[0])
        agent_frame = tk.Frame(button_container)
        agent_frame.pack(side="left", padx=4, pady=4)
        tk.Label(agent_frame, text="Agent:").pack(side="left", padx=(0, 2))
        agent_menu = tk.OptionMenu(agent_frame, selected_agent, *agent_options, command=lambda *_: update_overlays())
        agent_menu.pack(side="left")

        hidden_defaults = {"Entropy", "JS"}
        if metrics_order:
            default_selection = [name for name in metrics_order if name not in hidden_defaults]
            if not default_selection:
                default_selection = metrics_order[:]
            metric_frame = tk.Frame(button_container)
            metric_frame.pack(side="left", padx=4, pady=4)
            tk.Label(metric_frame, text="Metrics:").pack(side="left")
            for name in metrics_order:
                var = tk.IntVar(value=1 if name in default_selection else 0)
                metric_vars[name] = var
                chk = tk.Checkbutton(
                    metric_frame,
                    text=name,
                    variable=var,
                    command=lambda metric=name: _toggle_metric(metric),
                )
                chk.pack(side="left", padx=(2, 2))
        else:
            selected_metrics = []

        options_frame = tk.Frame(button_container)
        options_frame.pack(side="left", padx=4, pady=4)
        for name in extra_series_config:
            tk.Checkbutton(
                options_frame,
                text=f"Show {name}",
                variable=extra_series_vars[name],
                command=draw_metrics,
            ).pack(side="left", padx=4)
        if hamming_evo_available:
            tk.Checkbutton(
                options_frame,
                text="Show Hamming Evolution",
                variable=show_hamming_evo_var,
                command=draw_metrics,
            ).pack(side="left", padx=4)
            shared_instance_var.trace_add("write", lambda *_: draw_metrics())
            shared_average_var.trace_add("write", lambda *_: draw_metrics())
        if theta_container is not None:
            tk.Checkbutton(
                options_frame,
                text="Instance Explorer",
                variable=theta_var,
                command=_toggle_theta_panel,
            ).pack(side="left", padx=4)

        selected_metrics = [name for name in metrics_order if metric_vars.get(name, tk.IntVar()).get()]
        if not selected_metrics and metrics_order:
            selected_metrics = [name for name in metrics_order if name not in hidden_defaults] or metrics_order[:]
            for name in metrics_order:
                metric_vars[name].set(1 if name in selected_metrics else 0)

        root.update_idletasks()
        draw_metrics()
        _toggle_theta_panel()

        def _close():
            root.quit()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _close)
        root.mainloop()
        plt.close(fig)
    except Exception as exc:                    
        print(f"Failed to render Tkinter plots: {exc}")


def render_svgd_field_plot(snapshot):
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
        colors = plt.cm.get_cmap("tab11", num_agents)

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
    except Exception as exc:                    
        print(f"Failed to render SVGD field plot: {exc}")


def _build_best_individual_table_panel(container, history, instance_var, average_var):
    """
    Table M x M de la distance de Hamming réelle (pas d'expectation) entre les
    meilleurs individus trouvés par chaque agent, par instance (axe B préservé,
    aucune moyenne inter-instances par défaut). `instance_var`/`average_var` sont
    partagés avec les autres panels d'instance (ex: heatmap probs) pour rester
    synchronisés. Quand `average_var` est actif, affiche la moyenne sur toutes
    les instances plutôt que l'instance sélectionnée.
    """
    hamming = history.get("hamming")
    best_scores = history.get("best_scores")
    best_epochs = history.get("best_epochs")
    if hamming is None or hamming.ndim != 3 or hamming.shape[0] == 0:
        return None

    num_instances, num_agents, _ = hamming.shape

    panel = tk.LabelFrame(container, text="Best Individuals Hamming (bits)")
    panel.pack(side="top", fill="x", padx=10, pady=6)

    status_var = tk.StringVar()
    tk.Label(panel, textvariable=status_var).pack(pady=(0, 2))

    tables_row = tk.Frame(panel)
    tables_row.pack(fill="x", padx=10, pady=6)

    table_frame = tk.Frame(tables_row)
    table_frame.pack(side="left", anchor="n")

    groups_frame = tk.Frame(tables_row)
    groups_frame.pack(side="left", anchor="n", padx=(20, 0))

    def _shade(ratio):
        ratio = max(0.0, min(1.0, float(ratio)))
        r0, g0, b0 = 255, 255, 255
        r1, g1, b1 = 217, 83, 79
        r = int(r0 + (r1 - r0) * ratio)
        g = int(g0 + (g1 - g0) * ratio)
        b = int(b0 + (b1 - b0) * ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _group_agents_by_optimum(mat, scores_row):
        parent = list(range(num_agents))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if float(mat[i, j]) == 0.0:
                    union(i, j)

        groups = {}
        for idx in range(num_agents):
            groups.setdefault(find(idx), []).append(idx)
        ordered = sorted(groups.values(), key=lambda members: (-len(members), members[0]))
        lines = []
        for rank, members in enumerate(ordered, start=1):
            agents_str = ", ".join(f"Agent {m}" for m in members)
            score_str = f"{scores_row[members[0]]:.3f}" if scores_row is not None else "n/a"
            lines.append(f"Optimum {rank} ({agents_str}) score: {score_str}")
        return lines

    def draw_table(*_):
        for widget in table_frame.winfo_children():
            widget.destroy()
        for widget in groups_frame.winfo_children():
            widget.destroy()
        if average_var.get():
            mat = hamming.mean(axis=0)
            scores_row = best_scores.mean(axis=0) if best_scores is not None else None
            epochs_row = best_epochs.mean(axis=0) if best_epochs is not None else None
            status_var.set(f"Moyenne sur {num_instances} instances")
        else:
            instance_idx = max(0, min(num_instances - 1, instance_var.get()))
            mat = hamming[instance_idx]
            scores_row = best_scores[instance_idx] if best_scores is not None else None
            epochs_row = best_epochs[instance_idx] if best_epochs is not None else None
            status_var.set(f"Instance {instance_idx}")

        if average_var.get():
            tk.Label(
                groups_frame,
                text="Optima (masqué en vue Moyenne, sélectionnez une instance)",
                anchor="w",
                justify="left",
            ).pack(anchor="w")
        else:
            tk.Label(groups_frame, text="Optima trouvés :", anchor="w", justify="left", font=("", 9, "bold")).pack(
                anchor="w"
            )
            for line in _group_agents_by_optimum(mat, scores_row):
                tk.Label(groups_frame, text=line, anchor="w", justify="left").pack(anchor="w")

        mat_max = float(mat.max()) if mat.size else 0.0

        tk.Label(table_frame, text="", width=9).grid(row=0, column=0)
        for j in range(num_agents):
            header = f"Agent {j}"
            if scores_row is not None:
                header += f"\n{scores_row[j]:.3f}"
            if epochs_row is not None:
                header += f"\nep {epochs_row[j]:.0f}"
            tk.Label(table_frame, text=header, borderwidth=1, relief="solid", width=9).grid(
                row=0, column=j + 1, sticky="nsew"
            )
        for i in range(num_agents):
            tk.Label(table_frame, text=f"Agent {i}", borderwidth=1, relief="solid", width=9).grid(
                row=i + 1, column=0, sticky="nsew"
            )
            for j in range(num_agents):
                value = float(mat[i, j])
                ratio = value / mat_max if mat_max > 0 else 0.0
                tk.Label(
                    table_frame,
                    text=f"{value:.1f}",
                    borderwidth=1,
                    relief="solid",
                    width=9,
                    bg=_shade(ratio),
                ).grid(row=i + 1, column=j + 1, sticky="nsew")

    instance_var.trace_add("write", draw_table)
    average_var.trace_add("write", draw_table)
    draw_table()
    return panel


def _compute_agent_hamming_evolution(history):
    """
    Calcul pur (sans Tkinter) : pour chaque step enregistre, la distance de
    Hamming attendue moyenne de chaque agent vis-a-vis des M-1 autres, a partir
    des probs (meme formule que HK.py). Axe B preserve (pas via metrics.py).

    Retourne (per_agent_all, num_instances) avec per_agent_all de forme
    (T, M, B), ou (None, 0) si les donnees ne s'y pretent pas.
    """
    values = history.get("values") or []
    if not values or len(values[0]) < 2:
        return None, 0

    def _to_numpy(tensor):
        return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)

    try:
        stacked = np.stack(
            [np.stack([_to_numpy(agent_tensor) for agent_tensor in step], axis=0) for step in values],
            axis=0,
        )
    except ValueError:
        return None, 0
    # stacked: (T, M, B, N) binaire, ou (T, M, B, N, D) categoriel
    if stacked.ndim not in (4, 5):
        return None, 0
    M = stacked.shape[1]
    num_instances = stacked.shape[2]
    if num_instances <= 0 or M < 2:
        return None, 0
    is_categorical = stacked.ndim == 5

    pi = stacked[:, :, None, ...]
    pj = stacked[:, None, :, ...]
    if is_categorical:
        match = (pi * pj).sum(axis=-1)
        dist = (1.0 - match).mean(axis=-1)
    else:
        dist = (pi + pj - 2 * pi * pj).mean(axis=-1)
    # dist: (T, M, M, B) ; dist[t, i, i, b] == 0, donc sommer sur j inclut le "soi" sans biaiser
    per_agent_all = dist.sum(axis=2) / max(M - 1, 1)  # (T, M, B)
    return per_agent_all, num_instances


def _build_probs_heatmap_panel(container, root_window, history, num_agents, instance_var, average_var):
    """
    Heatmap M agents x N dimensions des probabilités (sigmoid/softmax de theta),
    avec un slider sur les steps enregistrés. `instance_var`/`average_var` sont
    partagés avec les autres panels d'instance (ex: table Hamming) pour rester
    synchronisés. Quand `average_var` est actif, affiche la moyenne des probas
    sur toutes les instances plutôt que l'instance sélectionnée.
    """
    values = history.get("values") or []
    if not values or num_agents == 0:
        return

    def _agent_matrix(agent_tensor, instance_idx):
        arr = agent_tensor.detach().cpu().numpy() if hasattr(agent_tensor, "detach") else np.asarray(agent_tensor)
        row = arr.mean(axis=0) if instance_idx is None else arr[instance_idx]
        if row.ndim == 2:
            return row.argmax(axis=-1), row.max(axis=-1), row.shape[-1]
        return row, None, None

    def _to_matrix(step_entry, instance_idx):
        rows = [_agent_matrix(agent_tensor, instance_idx) for agent_tensor in step_entry]
        if rows[0][1] is not None:
            cat_matrix = np.stack([r[0] for r in rows], axis=0)
            conf_matrix = np.stack([r[1] for r in rows], axis=0)
            return True, cat_matrix, conf_matrix, rows[0][2]
        matrix = np.stack([r[0] for r in rows], axis=0)
        return False, matrix, None, None

    first_agent = values[0][0]
    first_arr = first_agent.detach().cpu().numpy() if hasattr(first_agent, "detach") else np.asarray(first_agent)
    num_instances = int(first_arr.shape[0]) if first_arr.ndim >= 1 else 0
    if num_instances <= 0:
        return

    panel = tk.LabelFrame(container, text="Probs Heatmap Explorer")
    panel.pack(side="bottom", fill="both", expand=True, padx=10, pady=6)
    panel.pack_propagate(False)

    fig = plt.figure(figsize=(6, 4))
    canvas = FigureCanvasTkAgg(fig, master=panel)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    epoch_var = tk.IntVar(value=0)
    status_var = tk.StringVar()
    tk.Label(panel, textvariable=status_var).pack(pady=2)

    def clamp(var, upper):
        try:
            val = int(var.get())
        except (tk.TclError, ValueError):
            val = 0
        val = max(0, min(upper, val))
        if val != var.get():
            var.set(val)
        return val

    def update_plot(*_):
        epoch_idx = clamp(epoch_var, len(values) - 1)
        if average_var.get():
            instance_idx = None
            instance_label = f"Moyenne sur {num_instances} instances"
        else:
            instance_idx = clamp(instance_var, num_instances - 1)
            instance_label = f"Instance {instance_idx}"
        is_categorical, mat_a, mat_b, num_categories = _to_matrix(values[epoch_idx], instance_idx)

        fig.clear()
        ax = fig.add_subplot(111)
        num_rows = mat_a.shape[0]

        if is_categorical:
            from matplotlib.colors import hsv_to_rgb
            from matplotlib.patches import Patch
            num_categories = max(int(num_categories), 1)
            hue = mat_a.astype(np.float32) / num_categories
            sat = np.ones_like(hue)
            val = np.clip(mat_b, 0.0, 1.0)
            rgb = hsv_to_rgb(np.stack([hue, sat, val], axis=-1))
            ax.imshow(rgb, aspect="auto", interpolation="nearest")
            legend_handles = [
                Patch(facecolor=hsv_to_rgb([c / num_categories, 1.0, 1.0]), label=f"Cat. {c}")
                for c in range(num_categories)
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize="x-small",
                title="Teinte=catégorie, luminosité=confiance",
                framealpha=0.85,
            )
        else:
            im = ax.imshow(mat_a, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_yticks(range(num_rows))
        ax.set_yticklabels([f"Agent {m}" for m in range(num_rows)])
        for row in range(1, num_rows):
            ax.axhline(row - 0.5, color="white", linewidth=1.2)
        ax.set_xlabel("Dimension")
        ax.set_title(f"{instance_label} – Epoch {epoch_idx + 1}/{len(values)}")
        fig.tight_layout()
        status_var.set(f"Epoch {epoch_idx + 1}/{len(values)} – {instance_label}")
        canvas.draw_idle()

    slider = tk.Scale(
        panel,
        from_=0,
        to=len(values) - 1,
        orient="horizontal",
        length=450,
        command=lambda val: (epoch_var.set(int(float(val))), update_plot()),
        label="Epoch",
    )
    slider.pack(fill="x", padx=12, pady=6)

    def step_epoch(delta):
        new_idx = max(0, min(len(values) - 1, epoch_var.get() + delta))
        slider.set(new_idx)

    root_window.bind("<Left>", lambda event: step_epoch(-1))
    root_window.bind("<Right>", lambda event: step_epoch(1))

    instance_var.trace_add("write", update_plot)
    average_var.trace_add("write", update_plot)
    update_plot()

    return panel
