# Suite EvoCOP du papier PPSN (SVGD-EDA) — état du projet

Document pont entre la session Claude Code (exécution, code, cluster) et le Project
claude.ai dédié à la rédaction. À tenir à jour au fil des résultats.

## Papier de référence

"Stein Variational Black-Box Combinatorial Optimization" (PPSN, Landais et al.),
arXiv:2604.15837v1. Code : github.com/LandOS973/Supplementary-material.

## Contributions proposées pour EvoCOP

1. **Avantage par rang par agent** (`advperagentrankweighted` dans le code) au lieu
   du rang global du papier PPSN (§3.2). Chaque agent calcule Ŝ dans sa propre
   population plutôt que dans la population complète.
2. **Mise à jour proximale multi-époques** (`ppo_active` / mode KL dans
   `SVGD_EDA.py`) : K époques internes par génération sur le même batch, ratio
   d'importance par variable, pénalité KL(π_old‖π_new) à β fixe, SVGD appliqué à
   chaque époque interne. Motivation : plus de pas de gradient par évaluation,
   sans coût d'évaluation supplémentaire — vise la *dilution du budget* identifiée
   au §4.3 du PPSN (dégradation au-delà de m≈14).
3. **Extension du benchmark** : QUBO/PUBOi (déjà présent dans le repo pour
   n=64/128/256) étendu à n=512, plus NK/NK3 à n=512 (instances générées cette
   session via `source_code/utils/generate_nk_instances.py` et
   `source_code/environment/qubo.py` + générateur PUBOi officiel cloné et
   paramétré par reverse-engineering des instances existantes).

## Ossature (6 sections)

1. Intro — reprendre la motivation du PPSN, pointer explicitement les deux
   faiblesses (dilution du budget §4.3, cas K=8/NK3 où SVGD-EDA perdait :
   rangs 14/44/46 sur 83).
2. Background — condensé, renvoi au PPSN pour SVGD-EDA de base.
3. Méthode — 3.1 rang par agent (justification IGO par agent), 3.2 mise à jour
   proximale (dérivation, cadrage théorique — voir section dédiée ci-dessous).
4. Protocole — même méthodologie que PPSN : grid search unique sur un ensemble
   représentatif d'instances, config figée appliquée partout (pas de tuning par
   instance). Nevergrad 1.0.12 (81 algos "Discrete*"), + PBIL/MIMIC/BOA.
   100 runs (10 instances × 10 restarts), Wilcoxon apparié.
5. Résultats — tableau global étendu (NK/NK3/QUBO, n=64..512), courbe gain vs n
   et vs rugosité, ablation en cascade (PPSN → +rang par agent → +proximal),
   sensibilité à K/β/eps et à m (le proximal repousse-t-il la dilution ?),
   analyse de diversité (Hamming/entropie par agent).
6. Conclusion et limites.

## Cadrage théorique retenu : option A

Le mécanisme SVGD-EDA de base (PPSN) approxime une distribution de Boltzmann
unique p(θ) ∝ exp(J(θ)/γ). Avec le rang par agent et le mode proximal, cette
garantie ne tient plus telle quelle :
- rang par agent → objectif par particule, plus une cible commune ;
- mode proximal → ancrage par particule (KL vs θ_j_old propre à chaque agent),
  ratio calculé par variable (pas le ratio joint), répulsion appliquée K fois
  par génération sur échantillons figés.

Décision : ne pas prétendre à une nouvelle preuve de convergence. Présenter les
deux mécanismes comme des heuristiques motivées (cible inclinée + pas proximal,
cf. cadrage KL-regularized reward maximization façon Korbak et al. 2022,
rapproché du papier "Variational Proximal Policy Optimization" arXiv:2606.08032
— à citer comme travaux connexes, pas comme justification, leur propre dérivation
a des trous, cf. discussion session). Légitimer empiriquement via l'ablation
`no_repulsion` (déjà dans le code) appliquée aux nouvelles variantes.

Piste non prioritaire mais à garder en perspective : noyau SVGD avec métrique de
Fisher (naturel pour Bernoulli/catégoriel, Fisher diagonale = p(1-p) sur les
logits) — recoupe le "future work" du PPSN sur les noyaux Fisher-Rao/Jensen-Shannon.

## Défis identifiés et plan de résolution

| # | Défi | Statut | Plan |
|---|---|---|---|
| 1 | Ablation propre (rang par agent et proximal mélangés avec M et γ différents du PPSN) | À faire | Cascade PPSN → +rang par agent → +proximal, chaque étage avec sa propre config figée trouvée par grid search de même taille |
| 2 | Biais de sélection (config choisie et rapportée sur les mêmes instances) | À faire | Même méthodologie que PPSN (déjà acceptée par l'utilisateur) : un seul grid search par variante, config figée partout. Documenter explicitement l'ensemble d'instances utilisé pour le grid search |
| 3 | Théorie (plus de cible Boltzmann unique) | Résolu (option A ci-dessus) | Ablation no_repulsion sur les nouvelles variantes à faire |
| 4 | Mécanisme peu clair (diversité Hamming mitigée : plus basse que PPSN à faible K sur NK, plus basse aussi à K=8 sur NK3 malgré le meilleur score) | À investiguer | Persister les stats de debug (`_print_debug` est un stub aujourd'hui) en gardant l'axe instance B ; comparer à un PPSN dont γ est réglé pour atteindre la même diversité (test décisif : si PPSN score moins bien à diversité égale, le gain ne vient pas de la quantité de diversité) |
| 5 | Contribution perçue comme mince | Risque réel | Étoffer via le fil "dilution du budget" (balayage en m complet sur les 3 variantes) + passage à l'échelle n=512 + analyse de mécanisme |
| 6 | Données n=512 | En cours | Campagne nevergrad (81 algos × QUBO n=512) en cours sur Jean Zay (job JZ, ~4860 tâches réparties, voir `jeanzay/expe_nevergrad_512.slurm` et `jeanzay/compute_missing_tasks.py`). Reste : NK/NK3 à n=512 côté nevergrad, et les runs SVGD-EDA propres (PPSN/+rang/+proximal) à toutes tailles |
| 7 | Nommage "PPO" trompeur (pas de clipping, c'est un mode KL) | Mineur | Renommer en "proximal"/"KL-regularized multi-epoch" dans le papier |

## Exclusions documentées du pool de concurrents

- **`DiscreteNoisyInfSplits`, exclu à n=512 uniquement.** Mesuré sur JZ (campagne QUBO
  n=512, job array) : ~5h par instance (10 instances requises par run), contre
  ~15-20 min/instance pour les 80 autres algos du pool — un facteur ~15-20x.
  Progression réelle confirmée (scores qui évoluent normalement), donc pas un bug/
  boucle infinie mais un cas pathologiquement coûteux à cette échelle (le nom même,
  "InfSplits", suggère un paramètre de split interne démesuré). Coût total estimé
  intraitable (~50h+ par combo (t, seed), ×60 combos). À documenter dans le papier
  avec la même justification structurelle que l'exclusion de PBIL sur NK3 dans le
  PPSN (Table 1) — reste dans le pool pour n=64/128/256 où il tournait normalement.

- **NK3 K=8 exclu du dépôt git à n=512 (raison différente : taille de fichier, pas
  performance).** La table de contribution NKD grandit en N×D^(K+1) ; pour NK3
  (D=3), K=8 donne 3^9=19683 valeurs/variable × 512 variables ≈ 10M floats en
  texte = 186 Mo/fichier × 10 instances, au-dessus de la limite GitHub (100 Mo).
  Les instances existent quand même (générées localement et sur JZ via
  `source_code/utils/generate_nk_instances.py --dim 512 --k 8 --problems nk3`,
  transférées par `rsync` direct plutôt que git) mais ne sont pas versionnées.
  NK K=8 n'a pas ce problème (D=2 -> 512 valeurs/variable seulement, ~5 Mo/fichier).
  À mentionner dans le protocole expérimental si NK3 K=8 à n=512 est rapporté.

## Expériences en cours / à lancer (côté calcul)

- [x] Génération instances NK/NK3 n=512 (`source_code/utils/generate_nk_instances.py`)
- [x] Génération instances QUBO n=512 (générateur PUBOi officiel, paramètres
      reverse-engineerés depuis les instances 64/128/256 existantes)
- [~] Campagne nevergrad 81 algos × QUBO n=512 (Jean Zay, job array, proche de
      complet — voir `sacct` / `compute_missing_tasks.py` pour l'état courant)
- [ ] Campagne nevergrad 81 algos × NK/NK3 n=512
- [ ] Grid search par variante (PPSN / +rang par agent / +proximal), même
      protocole que PPSN §4.1
- [ ] Ablation en cascade sur toutes tailles/problèmes
- [ ] Balayage en m (1..~24) pour les 3 variantes, NK/NK3/QUBO, plusieurs n
- [ ] Ablation `no_repulsion` sur les nouvelles variantes
- [ ] Persistance des stats de debug (agent-level) pour l'analyse de diversité

## Pièges techniques rencontrés (pour ne pas les refaire)

- `main_nevergrad.py` utilisait des chemins relatifs au répertoire courant —
  corrigé en chemins absolus basés sur l'emplacement du script.
- nevergrad doit être pinné à 1.0.12 (version du papier) — sinon des noms
  d'algo diffèrent silencieusement (crash instantané mais fichier résultat
  quand même créé avec juste l'en-tête, car l'écriture du fichier précède la
  validation du nom d'algo dans le script).
- Granularité des jobs Slurm : un algo peut être jusqu'à ~3x plus lent qu'un
  autre pour le même budget — découper au niveau (algo, t, seed), pas
  (algo, t), pour éviter les timeouts en cascade.
