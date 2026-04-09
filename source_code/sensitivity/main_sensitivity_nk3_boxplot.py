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


def main() -> None:
    config_input = input("Enter config name (with __M<value>__): ").strip()
    try:
        summary_csv, plot_dir = run_sensitivity_from_config(
            config_input,
            problem_label="NK3",
            folder_pattern=r"^NK3_dim(?P<n>\d+)_t(?P<t>\d+)$",
            summary_filename="sensitivity_nk3_from_config.csv",
            key_labels=("N", "K"),
            title_fn=lambda n, k: f"NK3 (N={n}, K={k})",
            filename_fn=lambda n, k: f"boxplot_NK3_{n}_{k}.png",
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
