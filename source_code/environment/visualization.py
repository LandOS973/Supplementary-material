import os
import numpy as np
import torch

_HEADLESS = not (
    os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
    or os.environ.get("MPLBACKEND")
)

try:
    if _HEADLESS:
        raise RuntimeError("Headless environment detected")
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting optional
    tk = None
    FigureCanvasTkAgg = None
    plt = None
    _HEADLESS = True


def render_agent_dashboard(iterations, hamming_history, kl_history, agent_fitness_history, num_agents, theta_history):
    if tk is None or plt is None or FigureCanvasTkAgg is None:
        print("Tkinter/matplotlib not available, skipping dashboard.")
        return

    try:
        root = tk.Tk()
        root.title("Agent Dashboard")
        try:
            root.state("zoomed")
        except Exception:
            root.attributes("-zoomed", True)

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        metrics_frame = tk.Frame(main_frame)
        metrics_frame.pack(side="left", fill="both", expand=True)

        rows = 3 if agent_fitness_history and num_agents > 0 else 2
        fig, axes = plt.subplots(rows, 1, figsize=(10, 6), sharex=True)
        axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes

        if iterations and hamming_history:
            axes[0].plot(iterations, hamming_history, color="tab:blue")
        axes[0].set_title("Average Hamming Distance")
        axes[0].set_ylabel("Hamming")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        if iterations and kl_history:
            axes[1].plot(iterations, kl_history, color="tab:orange")
        axes[1].set_title("Average KL Distance")
        axes[1].set_ylabel("KL")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        if len(axes) == 3:
            axes[2].set_title("Agent Fitness Evolution")
            axes[2].set_ylabel("Fitness")
            if iterations:
                for agent_idx in range(num_agents):
                    series = [epoch[agent_idx] for epoch in agent_fitness_history]
                    axes[2].plot(iterations, series, label=f"Agent {agent_idx}")
            axes[2].grid(True, linestyle="--", alpha=0.4)
            axes[2].legend()

        axes[-1].set_xlabel("Evaluations")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=metrics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        if theta_history and theta_history.get("values"):
            _build_theta_panel(main_frame, root, theta_history)

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


def _build_theta_panel(container, root_window, history):
    values = history.get("values") or []
    if not values:
        return

    first_entry = values[0]
    num_agents = len(first_entry)
    if num_agents == 0:
        return

    sample = first_entry[0]
    num_instances = sample.shape[0]
    num_dims = sample.shape[1]

    panel = tk.LabelFrame(container, text="Theta Evolution Explorer")
    panel.pack(side="right", fill="both", expand=True, padx=10, pady=6)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("sigmoid(theta[dim X])")
    ax.set_ylabel("sigmoid(theta[dim Y])")

    cmap = plt.cm.get_cmap("tab20", max(num_agents, 1))
    scatters = [
        ax.scatter([], [], color=cmap(agent_idx), label=f"Agent {agent_idx}")
        for agent_idx in range(num_agents)
    ]

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=panel)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    controls = tk.Frame(panel)
    controls.pack(fill="x", padx=10, pady=6)

    epoch_var = tk.IntVar(value=0)
    instance_var = tk.IntVar(value=0)
    dim_x_var = tk.IntVar(value=0)
    dim_y_var = tk.IntVar(value=1 if num_dims > 1 else 0)

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

    def update_plot(*_):
        epoch_idx = clamp(epoch_var, len(values) - 1)
        inst_idx = clamp(instance_var, num_instances - 1)
        dx = clamp(dim_x_var, num_dims - 1)
        dy = clamp(dim_y_var, num_dims - 1)

        ax.set_title(f"Instance {inst_idx} – dims ({dx},{dy})")

        entry = values[epoch_idx]
        for agent_idx, scatter in enumerate(scatters):
            final_probs = entry[agent_idx]
            x = float(final_probs[inst_idx, dx].item())
            y = float(final_probs[inst_idx, dy].item())
            scatter.set_offsets([[x, y]])
        status_var.set(f"Epoch {epoch_idx + 1}/{len(values)} – Instance {inst_idx}")
        ax.legend(loc="upper right", ncol=2 if num_agents > 6 else 1, fontsize="small")
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

    labeled_spinbox(controls, "Instance", instance_var, num_instances - 1)
    labeled_spinbox(controls, "Dim X", dim_x_var, num_dims - 1)
    labeled_spinbox(controls, "Dim Y", dim_y_var, num_dims - 1)

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

    update_plot()
