from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .core import run_sensitivity_from_config
else:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from source_code.sensitivity.core import run_sensitivity_from_config

DEFAULT_CONFIG_NAME = "krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01"


def main() -> None:
    config_input = (
        input(
            f"Enter config name (ex: {DEFAULT_CONFIG_NAME}) [default: {DEFAULT_CONFIG_NAME}]: "
        ).strip()
        or DEFAULT_CONFIG_NAME
    )
    try:
        summary_csv, plot_dir = run_sensitivity_from_config(
            config_input,
            problem_label="QUBO",
            folder_pattern=r"^QUBO_dim(?P<n>\d+)_t(?P<t>\d+)$",
            summary_filename="sensitivity_qubo_from_config.csv",
            key_labels=("n", "type_instance"),
            title_fn=lambda n, t: f"QUBO (N={n}, Type={t})",
            filename_fn=lambda n, t: f"boxplot_QUBO_{n}_type{t}.png",
            abs_values=True,
            key_groups=("n", "t"),
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved boxplots to: {plot_dir}")


if __name__ == "__main__":
    main()
