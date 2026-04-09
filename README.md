# Supplementary Material

## Experiment Reference

This repository contains the experiments for:

**Stein Variational Black-Box Combinatorial Optimization**

Thomas Landais, Olivier Goudet, Adrien Goëffon, Frédéric Saubion, and Sylvain Lamprier.

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r source_code/requirement.txt
```

## Main Entry Point

The main script is `source_code/main.py`.

It relies on Hydra configuration files in `config/`:

- `config/config.yaml`: global settings (problem, budget, seed, etc.)
- `config/problem/*.yaml`: problem settings
- `config/agent/*.yaml`: strategy settings
- `config/kernel/*.yaml`: kernel settings

Standard run:

```bash
python source_code/main.py
```

## Repository Structure

- `source_code/`: training code, experiment scripts, and sensitivity scripts.
- `curves/`: curve/table generation and LaTeX exports.
- `additional_results/`: aggregation scripts (final scores, global ranking).
- `results/`: experiment outputs.
- `results/nevergrad/`: baseline results by algorithm/problem.
- `results/config/`: SVGD-EDA results by configuration.

## Review Pipeline

### 1) Generate tables

```bash
python curves/main_table.py --format all
python curves/main_table_interact_vs_no_interact.py \
  --config krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01
```

### 2) Generate curves

```bash
python curves/main_courbes_overall.py
```

## Default Configuration

When a script asks for a config name and the input is empty, the default value is:

`krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01`
