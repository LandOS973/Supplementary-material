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

## Ossature (5 sections, LNCS ~15 pages)

### Config retenue (grid search terminée, M et L identiques à PPSN)

`krbf__advperagentrankweighted__M7__L13__eps0p06__g0p01__ds1__dm0p01__ppokl__pe6__b0p1`
- M=7, L=13 (mêmes valeurs que PPSN → comparaison à population égale)
- eps=0.06 (contre 0.08 PPSN : rééchelonné car appliqué K=6 fois par génération)
- γ=0.01, K=6 epochs, β=0.1, ds=1 (pas de décroissance de γ)
- Contre PPSN (`krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01`) :
  43 victoires/56, rang moyen 1.9 contre 9.5, top-1 36 contre 23.
  NK3 256 K8 : rang 1 (PPSN 46) ; NK3 512 K8 : rang 23 (PPSN 47).

### Plan

1. **Intro** — motivation du PPSN, deux faiblesses identifiées : dilution du
   budget (§4.3 PPSN) et cas K=8 où SVGD-EDA décrochait (rangs 14/44/46 sur 83).
2. **Background** — condensé, renvoi au PPSN pour SVGD-EDA de base.
3. **Méthode** — 3.1 avantage par rang par agent, 3.2 mise à jour proximale
   multi-epoch (KL, K pas par génération sur le même batch). Cadrage : option A
   ci-dessous. Nommer « proximal / KL-regularized multi-epoch », pas « PPO ».
4. **Expériences**
   - **4.1 Protocole et réglages** (~¾ p.)
     - NK, NK3 (K ∈ {1,2,4,8}), QUBO/PUBOi (t0–t5), n ∈ {64,128,256,512} :
       56 distributions × 10 instances. 512 et QUBO nouveaux vs PPSN.
     - Budget 50000 évaluations, 100 runs (10 instances × 10 restarts),
       Wilcoxon apparié sur moyennes par instance.
     - Pool : ~80 nevergrad 1.0.12 + PBIL, PPBIL, PEDA + PPSN (PPSN compte
       dans le classement). Exclusions (voir section dédiée) : MIMIC/BOA à 512,
       PBIL/PPBIL sur NK3, DiscreteNoisyInfSplits à 512.
     - Réglages : une grid search par variante, config figée partout. Donner
       toutes les valeurs (M, L, eps, γ, K, β) ; une phrase sur eps rééchelonné
       par K. Chiffrer la robustesse (« X % des configs de la grid battent PPSN
       en rang moyen »). M et L identiques à PPSN pour isoler l'effet des
       nouveautés. **Ne pas parler du decay** (absent du PPSN, inutile ici :
       hyperparamètres complets des deux configs dans le README du dépôt).
     - Argument contre le sur-réglage : une config figée comparée au meilleur
       des ~80 concurrents choisi a posteriori par instance (oracle), dont les
       nevergrad incluent déjà des variantes de réglage du même algo.
   - **4.2 Comparaison globale** (~1,5 p.)
     - Tableau 1, colonnes : PPSN | nouvelle méthode | PPBIL (« — » sur NK3) |
       meilleur concurrent (autres, hors variantes SVGD-EDA) ; rang + score.
       NK et NK3 complets dans le corps, QUBO agrégé par n, table complète en
       annexe / dépôt.
     - Marqueurs : `*` significatif vs 2e du classement, `†` significatif vs PPSN.
     - Messages : rang moyen 1.9 vs 9.5 ; gain croissant avec n et K (+3 à 5 %
       à 512 et K=8) ; cas K=8 perdus par PPSN repassés rang 1–2 jusqu'à 256 ;
       défaites marginales non significatives sur instances faciles.
     - Limite affichée : NK3 512 K8 rang 23 → dilution du budget (renvoi 4.4).
   - **4.3 Ablation** (~1 p.) — cascade PPSN → + avantage par agent →
     + proximal (+ global avec KL pour croiser), mêmes M=7, L=13. Petit
     tableau de rangs moyens par famille ou barres de gain par n.
   - **4.4 Efficacité et diversité** (~1 p., figure 2 panneaux)
     - Gauche : convergence PPSN / nouvelle méthode / 2–3 meilleurs nevergrad
       sur QUBO 512 et NK3 256 K8 (argument sample efficiency).
     - Droite : Hamming moyen inter-agents (`avg_hamming`, distance moyenne
       entre distributions d'agents), nouvelle méthode vs PPSN, selon n.
     - Si la place le permet : M-sweep à population fixe par n → démontre la
       dilution du budget, prépare l'ouverture.
5. **Conclusion** — récapitulatif ; limite NK3 512 K8 (dilution du budget à M
   fixe malgré le gain d'efficacité) ; perspective : M adaptatif au cours du
   run (beaucoup d'agents pour explorer, fusion/élagage pour concentrer le budget).

### Données à produire (par priorité)

1. Wilcoxon (Tableau 1) : vs PPSN et vs meilleur concurrent par instance
   (`curves/main_table.py`, relancer le meilleur concurrent à 512 avec `-f`).
2. Ablation en cascade (4.3).
3. Courbes de convergence + diversité (4.4), depuis les historiques existants.
4. Chiffre de robustesse de la grid (dashboard).
5. M-sweep, si temps et place.

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

- **`MIMIC` et `BOA`, exclus à n=512 (QUBO, NK et NK3) — coût algorithmique
  O(n²), pas un bug de configuration.** Les deux implémentations
  (`source_code/eda/optimizer/mimic.py`, `boa.py`) construisent une structure
  de dépendance entre chaque paire de variables à chaque génération :
  `MIMIC.calc_bi_frequency` fait une triple boucle Python pure
  `for m in range(dim): for n in range(m,dim): for j in range(lam)` (≈16.7M
  itérations à n=512, contre ≈4.2M à n=256) ; `BOA.k2_algorithm` évalue le
  gain d'ajout d'arête pour chaque paire (i,j) via l'algorithme K2, même
  ordre de grandeur. Confirmé sur JZ : un job MIMIC/BOA avec budget=500 (donc
  quelques générations seulement) reste `RUNNING` plusieurs minutes sans
  produire de sortie, et les runs de la campagne QUBO n=512 (budget=50000, 6h
  de limite) n'ont jamais dépassé l'en-tête du fichier résultat (0/60 pour
  chacun des deux algos). PBIL, modèle univarié (O(n), pas de structure par
  paire), n'est pas concerné et reste dans le pool à n=512 (QUBO et NK — pas
  NK3, voir point suivant). `jeanzay/expe_edas_512.slurm` ne lance plus que
  PBIL (`ALGOS=(PBIL)`).

- **`PBIL` confirmé binaire seulement, ne supporte pas NK3 — correction d'une
  vérification erronée faite plus tôt dans le projet.** `eda/optimizer/pbil.py:26`
  fait `assert self.Cmax == 2` ; NK3 a D=3 catégories/variable donc
  `AssertionError` immédiate, quelle que soit la taille n. Les fichiers
  `results/nevergrad/PBIL/NK3/{64,128,256}` qui semblaient prouver le contraire
  ne sont en fait que des stubs d'un en-tête (`runtime, score`, 1 ligne, datés
  de mars) — un crash jamais remarqué, pas un résultat réel. C'est en fait
  cohérent avec le PPSN d'origine (Table 1), qui exclut déjà PBIL de NK3 pour
  la même raison. `expe_edas_512.slurm` exit proprement sans rien lancer si
  `-p NK3` (plus aucun algo de ce script n'est éligible une fois PBIL/MIMIC/BOA
  tous exclus de NK3 à n=512).

- **Bug de direction QUBO dans PEDA/PPBIL, corrigé — n'affecte pas le papier
  publié.** `x^T Q x` (= `WalshExpansion.eval`) est une MINIMISATION dans ce
  dépôt (convention partagée par `main_nevergrad.py`,
  `main_baseline_edas_and_tabu.py` et le SVGD-EDA du papier, `environment/qubo.py`).
  `main_expe_peda.py`/`main_expe_ppbil.py` (ajoutés le 2026-05-04, commit
  `4b3e2fc98`, donc après soumission PPSN — pas dans le pool du papier)
  **maximisaient** `x^T Q x` brut au lieu de le minimiser, sur QUBO uniquement
  (NK/NK3 étaient déjà corrects, convention de maximisation naturelle
  là-bas). Corrigé dans les 4 emplacements (CPU + GPU × PEDA + PPBIL) en
  négant le score interne. Toutes les données PEDA/PPBIL QUBO existantes
  (n=64/128/256/512) ont été supprimées (invalides, résolvaient le mauvais
  problème) et relancées.

- **Classement (`additional_results/main_global_ranking.py`) — bug de signe
  distinct, root-cause finalement isolée après deux corrections ratées.**
  N'affecte pas le papier publié (`curves/main_table.py` exclut explicitement
  QUBO du tableau final via `EXCLUDED_TABLE_PROBLEMS = {"QUBO", "UBQP"}`).
  Deux tentatives de fix ont échoué avant la bonne :
  1. D'abord négation conditionnée au type de problème (`if problem == "QUBO"`)
     — cassait PEDA/PPBIL une fois leur propre bug corrigé (leur score déjà
     bien orienté se retrouvait négativé deux fois).
  2. Puis négation conditionnée à l'algo (`if algo not in (PEDA, PPBIL)`,
     appliquée à QUBO/NK/NK3) — cassait VIENNARNA (négativé à tort) et restait
     fausse sur NK à n≤256 (vérifié : `DiscreteDE`/`DiscreteLengler3OnePlusOne`
     sur NK ont un score brut déjà positif et croissant dans leurs fichiers
     `.txt`, donc déjà "plus haut = meilleur", à ne pas négativer).
  **Cause réelle** : deux formats de fichiers résultat incompatibles
  coexistent sur disque pour le **même algo** et le **même type de
  problème**, selon la *génération/version du script* qui a produit le
  fichier — pas selon l'algo ni le type de problème :
  - Format ancien (n≤256, données de mars 2026), en-tête `runtime, score`
    (2 colonnes) : valeur déjà orientée "plus haut = meilleur". PEDA/PPBIL
    écrivent aussi ce format, systématiquement, sur toutes les tailles y
    compris n=512.
  - Format actuel (données n=512 générées cette session), en-tête
    `runtime, mean, median, std, 2%, 5%, ...` (13 colonnes) : la valeur
    "mean" est la quantité brute minimisée en interne par
    nevergrad/PBIL/MIMIC/BOA (`x^T Q x` pour QUBO, `-fitness` pour NK/NK3),
    non retournée — "plus bas = meilleur".
  Vérifié directement sur disque : même algo (`DiscreteLengler3OnePlusOne`),
  même problème (QUBO), instance/génération différente → en-tête différent,
  signe différent, échelle de grandeur totalement différente.
  **Fix retenu** : `read_last_score()` inspecte l'en-tête de chaque fichier
  et négative uniquement pour le format 13-colonnes ; plus aucune
  distinction par algo ou type de problème dans `build_rankings()`. Gère
  aussi VIENNARNA sans cas particulier (même format ancien, en-tête
  2-colonnes, confirmé identique octet-pour-octet après le fix — aucune
  régression). CSVs de `additional_results/global_ranking/*` régénérés et
  vérifiés sur QUBO/NK/NK3, ancienne et nouvelle génération : PEDA/PPBIL
  s'intègrent naturellement dans le peloton des deux côtés (plus jamais
  gonflés ni écrasés artificiellement), et les algos nevergrad réputés
  forts (famille DiscreteLengler) remontent bien en tête sur les données
  n=512.

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
