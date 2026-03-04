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
except Exception:  # pragma: no cover - plotting optional
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
    l2_history=None,
    l2_pairwise_history=None,
    l1_history=None,
    l1_pairwise_history=None,
    entropy_history=None,
    entropy_agent_history=None,
    kernel_value_history=None,
    kernel_grad_history=None,
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
    pairwise_l2 = _prepare_pairwise(l2_pairwise_history)
    pairwise_l1 = _prepare_pairwise(l1_pairwise_history)
    entropy_agent_series = _prepare_agent_series(entropy_agent_history)

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
    if iterations and l2_history:
        metrics_data["L2"] = dict(
            average=l2_history,
            ylabel="L2",
            title="Average L2 Distance",
            color="tab:red",
            overlay_type="pairwise",
            overlay_data=pairwise_l2,
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
    if iterations and kernel_value_history:
        metrics_data["Kernel Value"] = dict(
            average=kernel_value_history,
            ylabel="k(i,j)",
            title="Average Kernel Similarity",
            color="tab:purple",
            overlay_type=None,
            overlay_data=None,
        )
    if iterations and kernel_grad_history:
        metrics_data["Kernel Gradient"] = dict(
            average=kernel_grad_history,
            ylabel="‖∇k‖",
            title="Average Kernel Gradient ",
            color="tab:brown",
            overlay_type=None,
            overlay_data=None,
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

        fitness_available = bool(agent_fitness_history and num_agents > 0)
        show_fitness_var = tk.IntVar(value=1 if fitness_available else 0)

        theta_available = bool(solutions_history and solutions_history.get("values"))
        theta_var = tk.IntVar(value=1 if theta_available else 0)
        theta_panel = None
        theta_pack_info = None
        theta_container = None
        pane_theta_width = max(400, root.winfo_screenwidth() // 5)
        if theta_available:
            theta_container = tk.Frame(pane, width=pane_theta_width)
            pane.add(theta_container)
            pane.paneconfigure(theta_container, minsize=pane_theta_width // 2)
            theta_panel = _build_solution_tsne_panel(theta_container, root, solutions_history, num_agents)
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

        def is_fitness_enabled():
            return fitness_available and show_fitness_var.get() == 1

        fitness_lines = []

        fitness_ax = None
        fitness_lines = []

        def draw_fitness_axis(total_rows):
            if not is_fitness_enabled():
                return None, []
            ax = fig.add_subplot(total_rows, 1, total_rows)
            ax.set_title("Agent Fitness Evolution")
            ax.set_ylabel("Fitness")
            if iterations:
                lines = []
                for agent_idx in range(num_agents):
                    series = [epoch[agent_idx] for epoch in agent_fitness_history]
                    (line,) = ax.plot(iterations, series, label=f"Agent {agent_idx}")
                    lines.append((agent_idx, line))
            else:
                lines = []
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            return ax, lines

        def draw_metrics():
            nonlocal fitness_ax, fitness_lines
            fig.clear()
            plot_handles.clear()
            overlay_lines.clear()
            total_metric_rows = len(selected_metrics)
            total_rows = total_metric_rows + (1 if is_fitness_enabled() else 0)
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

            nonlocal fitness_ax, fitness_lines
            fitness_ax, fitness_lines = draw_fitness_axis(total_rows)
            last_axis = fitness_ax if fitness_ax is not None else (
                plot_handles[selected_metrics[-1]]["axis"] if selected_metrics else None
            )
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
            if fitness_lines and fitness_ax:
                if show_average or anchor_idx is None or not is_fitness_enabled():
                    for _, line in fitness_lines:
                        line.set_visible(True)
                else:
                    for idx, line in fitness_lines:
                        line.set_visible(idx == anchor_idx)
                visible = [(line, line.get_label()) for _, line in fitness_lines if line.get_visible()]
                if visible:
                    handles_vis, labels_vis = zip(*visible)
                    fitness_ax.legend(handles_vis, labels_vis, loc="upper right")
                else:
                    leg = fitness_ax.get_legend()
                    if leg:
                        leg.remove()

            canvas.draw_idle()

        def _toggle_theta_panel():
            if not theta_container or not theta_panel:
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

        hidden_defaults = {"Entropy", "L2", "JS", "L1"}
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
        if fitness_available:
            tk.Checkbutton(
                options_frame,
                text="Show Fitness",
                variable=show_fitness_var,
                command=draw_metrics,
            ).pack(side="left", padx=4)
        if theta_panel is not None:
            tk.Checkbutton(
                options_frame,
                text="Solutions t-SNE",
                variable=theta_var,
                command=_toggle_theta_panel,
            ).pack(side="left", padx=4)

        selected_metrics = [name for name in metrics_order if metric_vars.get(name, tk.IntVar()).get()]
        if not selected_metrics and metrics_order:
            selected_metrics = [name for name in metrics_order if name not in hidden_defaults] or metrics_order[:]
            for name in metrics_order:
                metric_vars[name].set(1 if name in selected_metrics else 0)

        # ensure Tk has finalized widget sizes before drawing to avoid large initial padding
        root.update_idletasks()
        draw_metrics()
        _toggle_theta_panel()

        def _close():
            root.quit()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _close)
        root.mainloop()
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
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
    except Exception as exc:  # pragma: no cover
        print(f"Failed to render SVGD field plot: {exc}")


def _build_solution_tsne_panel(container, root_window, history, num_agents):
    try:
        from sklearn.manifold import TSNE
    except Exception:
        TSNE = None

    values = history.get("values") or []
    if not values:
        return

    if num_agents == 0:
        return
    lambda_per_agent = int(history.get("lambda_per_agent") or 0)
    if lambda_per_agent <= 0:
        return

    panel = tk.LabelFrame(container, text="Solutions t-SNE Explorer")
    panel.pack(side="right", fill="both", expand=True, padx=10, pady=6)
    panel.pack_propagate(False)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=panel)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    controls = tk.Frame(panel)
    controls.pack(fill="x", padx=10, pady=6)

    epoch_var = tk.IntVar(value=0)
    total_lambda = lambda_per_agent * num_agents
    fixed_perplexity = min(10, max(1, total_lambda - 1))

    def clamp(var, upper):
        try:
            val = int(var.get())
        except (tk.TclError, ValueError):
            val = 0
        val = max(0, min(upper, val))
        if isinstance(var, tk.StringVar):
            var.set(str(val))
        else:
            var.set(val)
        return val

    status_var = tk.StringVar()
    tk.Label(panel, textvariable=status_var).pack(pady=2)

    if num_agents < 2:
        tk.Label(panel, text="t-SNE requires at least 2 agents.").pack(pady=8)
        return panel
    if TSNE is None:
        tk.Label(panel, text="scikit-learn not available (install scikit-learn).").pack(pady=8)
        return panel

    labels = np.repeat(np.arange(num_agents), lambda_per_agent)
    embedding_cache = {}

    def update_plot(*_):
        epoch_idx = clamp(epoch_var, len(values) - 1)
        perp = fixed_perplexity

        entry = values[epoch_idx]
        if entry is None:
            status_var.set("t-SNE skipped: no sampled solutions.")
            return
        X = np.asarray(entry)
        if X.ndim == 3 and X.shape[-1] == 1:
            X = X[:, :, 0]
        if X.ndim != 2:
            status_var.set("t-SNE skipped: expected (total_lambda, N) samples.")
            return
        if X.shape[0] != total_lambda:
            status_var.set("t-SNE skipped: unexpected sample count.")
            return

        init_value = "pca"
        prev_embedding = embedding_cache.get(epoch_idx - 1)
        if prev_embedding is not None and prev_embedding.shape[0] == X.shape[0]:
            init_value = prev_embedding

        try:
            tsne = TSNE(
                n_components=2,
                metric="hamming",
                init=init_value,
                perplexity=perp,
                random_state=0,
            )
            embedding = tsne.fit_transform(X)
        except Exception as exc:
            status_var.set(f"t-SNE failed: {exc}")
            return

        embedding_cache[epoch_idx] = embedding

        ax.clear()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"Epoch {epoch_idx + 1}/{len(values)} – samples {total_lambda}")
        from matplotlib.lines import Line2D
        from matplotlib.colors import hsv_to_rgb
        hues = np.linspace(0, 1, max(num_agents, 1), endpoint=False)
        colors = hsv_to_rgb(np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1))
        legend_handles = []
        marker = "o"
        scale = float(np.std(embedding, axis=0).mean()) if embedding.size else 0.0
        jitter_scale = max(0.001, 0.02 * scale)
        for agent_idx in range(num_agents):
            mask = labels == agent_idx
            if not np.any(mask):
                continue
            points = embedding[mask]
            rounded = np.round(points, 3)
            uniq, counts = np.unique(rounded, axis=0, return_counts=True)
            sizes = 45 * (1 + 0.6 * (counts - 1))
            angle = 2 * np.pi * (agent_idx / max(num_agents, 1))
            offset = np.array([np.cos(angle), np.sin(angle)]) * jitter_scale
            ax.scatter(
                uniq[:, 0] + offset[0],
                uniq[:, 1] + offset[1],
                s=sizes,
                color=colors[agent_idx],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
                alpha=0.9,
                marker=marker,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor=colors[agent_idx],
                    markeredgecolor="white",
                    markersize=8,
                    label=f"Agent {agent_idx}",
                )
            )
        if legend_handles:
            ax.legend(handles=legend_handles, loc="best", fontsize="small")
        status_var.set(f"Epoch {epoch_idx + 1}/{len(values)} – perplexity {perp}")
        canvas.draw_idle()

    def recompute_now():
        update_plot()

    slider = tk.Scale(
        panel,
        from_=0,
        to=len(values) - 1,
        orient="horizontal",
        length=450,
        command=lambda val: (epoch_var.set(int(float(val))), status_var.set("Press Recompute t-SNE")),
        label="Epoch",
    )
    slider.pack(fill="x", padx=12, pady=6)
    tk.Button(controls, text="Recompute t-SNE", command=recompute_now).pack(side="left", padx=4)

    def step_epoch(delta):
        new_idx = max(0, min(len(values) - 1, epoch_var.get() + delta))
        slider.set(new_idx)

    root_window.bind("<Left>", lambda event: step_epoch(-1))
    root_window.bind("<Right>", lambda event: step_epoch(1))
    root_window.bind("<Return>", lambda event: recompute_now())
    root_window.bind("<KP_Enter>", lambda event: recompute_now())

    update_plot()

    return panel
