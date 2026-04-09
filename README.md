# Supplementary Material

## Point d'entrée principal

Le script de base est `source_code/main.py`.

Il s'appuie sur la configuration Hydra dans `config/` :

- `config/config.yaml` : configuration globale (problème, budget, seed, etc.)
- `config/problem/*.yaml` : paramètres des problèmes
- `config/agent/*.yaml` : paramètres des stratégies
- `config/kernel/*.yaml` : paramètres des kernels

Exécution standard :

```bash
python source_code/main.py
```

## Structure du dépôt

- `source_code/` : code d'entraînement, scripts d'expériences et sensibilité.
- `courbes/` : génération des courbes, tableaux et exports LaTeX.
- `additional_results/` : scripts d'agrégation (scores finaux, ranking global).
- `results/` : sorties expérimentales.
  - `results/nevergrad/` : résultats baselines par algo/problème.
  - `results/config/` : résultats SVGD-EDA par configuration.

## Pipeline review

### 1) Génération des tables

```bash
python courbes/main_table.py --format all
python courbes/main_table_interact_vs_no_interact.py \
  --config krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01
```

### 2) Génération des courbes

```bash
python courbes/main_courbes_overall.py
```

## Configuration par défaut

Quand un script demande un nom de config et que l'entrée est vide, la valeur par défaut est :

`krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01`

