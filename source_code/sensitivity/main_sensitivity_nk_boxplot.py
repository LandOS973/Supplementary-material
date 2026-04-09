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
            problem_label="NK",
            folder_pattern=r"^NK_dim(?P<n>\d+)_t(?P<t>\d+)$",
            summary_filename="sensitivity_nk_from_config.csv",
            key_labels=("N", "K"),
            title_fn=lambda n, k: f"NK (N={n}, K={k})",
            filename_fn=lambda n, k: f"boxplot_NK_{n}_{k}.png",
            abs_values=False,
            key_groups=("n", "t"),
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved boxplots to: {plot_dir}")


if __name__ == "__main__":
    main()
