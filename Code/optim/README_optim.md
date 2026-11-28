# Le dossier `optim/` expliqué

> **Dernière mise à jour : 28/11/2026 à 16h55**, d'après le code du commit `30d54af` (branche
> `branch_meddi`). Si un fichier de `optim/` a été modifié après cette date (`git log -- Code/optim`), ce
> README n'est peut-être plus à jour.

Ce document explique **ce que fait chaque fichier du dossier `optim/`**, **comment il relie l'aéro, la
SER et le quantique**, et **juste assez de théorie d'optimisation** pour comprendre le code et répondre au
jury. Les chiffres cités sortent des fichiers de `resultats/` (on dit lequel à chaque fois).

> **En une phrase.** `optim/` est le **chef d'orchestre** : il définit le problème bi-objectif
> (minimiser la traînée $C_D$ **et** la SER sous contraintes), il contient les méthodes qui cherchent le
> **front de Pareto** (NSGA-II, la **boucle QUBO**, l'optimisation bayésienne) et les scripts qui les
> **comparent** en comptant le nombre de simulations dont chacune a besoin.

---

## Sommaire

1. [Où se place `optim/` dans la chaîne](#1-où-se-place-optim-dans-la-chaîne)
2. [La théorie minimale](#2-la-théorie-minimale)
3. [Les fichiers un par un](#3-les-fichiers-un-par-un)
4. [Qui utilise quoi : interactions avec le reste du code](#4-qui-utilise-quoi--interactions-avec-le-reste-du-code)
5. [Les résultats à connaître](#5-les-résultats-à-connaître)
6. [Questions probables du jury](#6-questions-probables-du-jury)
7. [Commandes](#7-commandes)

---

## 1. Où se place `optim/` dans la chaîne

```
        aero/ (géométrie, C_D, C_L,max, α)        ser/ (SER du profil à l'incidence α)
                       │                                   │
                       └──────────────┬────────────────────┘
                                      ▼
     problem.py (V1) / problem_v2.py (V2) : une « simulation » = aéro + SER + contraintes
         └── table de TOUS les profils, calculée une fois (base_complete_*.csv, base_V2.csv)
                                      │
                                      ▼
     CountedProblem : accès à la table qui COMPTE les profils distincts demandés
                                      │
        ┌──────────────┬──────────────┼──────────────────┬──────────────┐
        ▼              ▼              ▼                  ▼              ▼
     hasard        nsga2.py      qubo_loop.py         bayes.py      énumération
   (référence)   (génétique)   (modèle → QUBO →     (krigeage +    (front EXACT,
                                 exact / recuit /     amélioration     la vérité)
                                 QAOA : quantique/)   espérée, V3)
        └──────────────┴──────────────┴──────────────────┴──────────────┘
                                      │
                                      ▼
     experiences_pipeline.py (V1), experiences_v2.py (V2), experiences_l1.py (V3)
         └── combien de simulations pour retrouver le front exact ? → resultats/pipeline*.json
```

**L'idée centrale du projet** : une simulation est **chère** (XFOIL + SER ici ; un vrai calcul CFD ou
électromagnétique prendrait des heures). La bonne méthode est celle qui trouve le front de Pareto
**avec le moins de simulations**. Comme la grille est petite (512 puis 1 024 profils), on peut tout
calculer une fois et connaître le **front exact** : on mesure alors objectivement chaque méthode.

---

## 2. La théorie minimale

### 2.1 Optimisation bi-objectif et front de Pareto

On veut minimiser **deux** objectifs en conflit : $f_1 = C_D$ (en counts) et $f_2 = \text{SER}$ (en dBm). Il
n'y a pas « un » meilleur profil, mais un ensemble de **compromis**.

- Un profil $a$ **domine** $b$ si $a$ est au moins aussi bon sur les deux objectifs et strictement
  meilleur sur au moins un.
- Le **front de Pareto** = l'ensemble des profils **non dominés** : on ne peut améliorer l'un des
  objectifs sans dégrader l'autre. C'est ce qu'on donne à l'ingénieur, qui choisit ensuite.

**Contraintes** (un profil qui les viole est **infaisable**, quelle que soit sa performance) :
- V1 : épaisseur $t \ge 8\,\%$, $C_L \ge 0{,}3$ à $\alpha = 2°$, calcul aéro valide ;
- V2 : $t \ge 8\,\%$, $C_{L,\max} \ge 1{,}2$ (manœuvre), croisière XFOIL convergée.

### 2.2 Mesurer la qualité d'un front trouvé

- **Fraction du front exact retrouvée** : combien des profils du vrai front ont été simulés.
- **Hypervolume (HV)** : l'**aire** de la zone dominée par les profils trouvés, jusqu'à un point de
  référence (objectifs normalisés dans $[0, 1]$, référence $(1{,}1 ; 1{,}1)$). Plus c'est grand, mieux
  c'est ; on le rapporte à l'HV du front exact (1 = parfait). C'est la mesure standard, qui récompense à
  la fois la **qualité** et l'**étalement** du front.
- **IGD** : distance moyenne de chaque point du vrai front au point trouvé le plus proche.

**Les indicateurs des tableaux de résultats :**
- `full_front` : nombre de simulations pour retrouver **tout** le front exact ;
- `hv99` : nombre de simulations pour atteindre **99 %** de l'HV exact ;
- `success` : part des graines qui retrouvent le front complet dans le budget ;
- **arrêt réaliste** (`stop_*`, V2) : en vrai on ne connaît pas le front exact. On s'arrête quand l'HV ne
  progresse plus (3 itérations pour la boucle, 5 générations pour NSGA-II), et on regarde ce qu'on a.

Chaque méthode est lancée avec **plusieurs graines aléatoires** (5 à 20) ; on donne les quartiles
(25 %, médiane, 75 %), car une seule exécution peut être chanceuse.

### 2.3 NSGA-II (algorithme génétique, la référence classique)

NSGA-II (Deb et al., 2002) fait évoluer une **population** de $N$ profils :
1. **Sélection** par tournoi : on préfère le meilleur **rang** (numéro du front non dominé), puis la plus
   grande **distance d'encombrement** (pour garder un front bien étalé) ;
2. **Croisement SBX** et **mutation polynomiale** sur les indices de grille vus comme des réels, puis
   arrondi ; les doublons sont éliminés ;
3. **Survie élitiste** : parents + enfants ($2N$) sont triés, on garde les $N$ meilleurs.

**Contraintes (règle de Deb)** : faisable > infaisable ; entre deux infaisables, la plus petite violation
gagne ; entre deux faisables, la dominance de Pareto.

C'est une méthode **sans modèle** : elle n'apprend rien de la physique, elle a juste besoin de beaucoup
d'évaluations.

### 2.4 La boucle QUBO (notre méthode, celle qui utilise le quantique)

C'est une méthode **à modèle de substitution** : au lieu d'explorer à l'aveugle, on **apprend** une
approximation bon marché du problème et on l'utilise pour choisir les prochains profils à simuler.

À chaque itération :
1. **Ajuster des modèles** sur les profils déjà simulés : quadratique pour $C_D$ et pour la SER
   (10 coefficients en V1, 15 en V2, sur les variables normalisées $z \in [-1, 1]$), **affine** pour la
   contrainte $C_L$ / $C_{L,\max}$ ;
2. **Pour chaque poids** $w \in \{0 ; 0{,}25 ; 0{,}5 ; 0{,}75 ; 1\}$, construire un **QUBO** :
   $$w\,\frac{\hat C_D}{\Delta C_D} + (1-w)\,\frac{\widehat{\text{SER}}}{\Delta\text{SER}} + \text{pénalités de contraintes}$$
   Les poids **balaient le front** : $w = 1$ cherche la traînée minimale, $w = 0$ la SER minimale ;
3. **Résoudre chaque QUBO** avec l'un des trois solveurs (exact, recuit simulé, QAOA ; voir
   `quantique/README_quantique.md`) et garder le meilleur profil **pas encore simulé** ;
4. **Simuler** ces profils (5 par itération), les ajouter aux données, recommencer.

**Les pénalités** (fonctions de `qubo_loop.py`) :
- $t \ge 8\,\%$ = interdire l'indice 0 de $t$. La pénalité naturelle est **cubique** (3 bits à 0) ; on la
  rend quadratique avec **1 qubit auxiliaire** (**réduction de Rosenberg**) ;
- $C_L \ge C_{L,\min}$ : on écrit $\hat C_L$ comme une fonction affine des bits, on ajoute une **variable
  d'écart** $s \ge 0$ sur **3 qubits**, et on pénalise $P\,(\hat C_L - C_{L,\min} - s)^2$ (nul si la
  contrainte est respectée avec le bon $s$).

D'où **13 qubits en V1** (9 + 1 + 3) et **14 en V2** (10 + 1 + 3).

**Le front rendu ne contient que des profils réellement simulés** : le modèle sert à **proposer**, jamais
à **conclure**.

### 2.5 L'optimisation bayésienne (V3, le vrai concurrent)

Même structure de boucle, mais :
- modèle = **krigeage** (processus gaussien), qui donne une prédiction **et une incertitude** ;
- sélection par **amélioration espérée** (EI) : on choisit le profil qui a la meilleure espérance de
  battre le meilleur actuel, en tenant compte de l'incertitude (équilibre exploration / exploitation) ;
- contrainte gérée par la **probabilité de faisabilité** (PoF).

C'est l'état de l'art pour les problèmes à simulations chères. `bayes.py` permet de faire une
**ablation** : changer le modèle (quadratique ou krigeage) et la sélection (minimum de la prédiction,
comme un QUBO résolu exactement, ou EI) séparément, pour savoir **d'où vient** la performance.

---

## 3. Les fichiers un par un

### Le cœur (méthodes et problème)

#### `pareto.py` — outils multi-objectifs *(utilisé partout)*
| Fonction | Rôle |
|---|---|
| `dominates(a, b)` | $a$ domine-t-il $b$ ? (tout est à **minimiser**) |
| `non_dominated_sort(F)` | tri non dominé rapide de Deb : liste des fronts successifs |
| `crowding_distance(F)` | distance d'encombrement (les extrémités valent $+\infty$) |
| `pareto_front(F)` | indices du premier front |
| `hypervolume_2d(F, ref)` | aire dominée, calcul exact en 2D (tri sur $f_1$, somme de rectangles) |
| `igd(F_found, F_true)` | distance générationnelle inversée |
| `Normalizer` | ramène les objectifs à $[0, 1]$ sur les profils faisables |

#### `nsga2.py` — NSGA-II écrit à la main
- `constrained_ranks(F, CV)` : rangs et encombrement selon la dominance contrainte de Deb ;
- `sbx`, `poly_mutation` : opérateurs génétiques ;
- `nsga2(problem, pop_size, n_gen, seed, ..., n_levels, max_sims, callback)` : l'algorithme. Les indices
  sont relâchés en réels dans $[-0{,}5 ; L - 0{,}5]$ puis arrondis (`to_idx`).

**Pourquoi le réécrire alors que pymoo existe** : pour le comprendre ligne à ligne et **contrôler
exactement ce qui est compté**. `check_pymoo.py` vérifie qu'il se comporte comme celui de pymoo.

#### `problem.py` — le problème V1 et le compteur
- `Scenario`, `scenario("S1_note" | "S2_sol")` : cas de vol + scénario radar (S1 : élévations
  $[-20°, 20°]$ ; S2 : $[-60°, 0°]$, radar au sol).
- `simulate(idx, space, sc)` : **une simulation V1** = modèle aéro rapide, puis SER à l'incidence trouvée,
  puis contraintes. Renvoie une ligne (dict) avec `CD_counts`, `SER_db`, `cv` (violation totale),
  `feasible`, `bits`...
- `build_table(sc, space, path)` : les 512 profils, ou relit le CSV s'il existe.
- **`CountedProblem(table, keys)`** : la pièce centrale de la comparaison. On l'appelle avec un indice de
  profil ; il renvoie la ligne de la table et **ajoute le profil à `seen` s'il est nouveau**. `n_sims` =
  nombre de profils **distincts** demandés. `F(idx)` renvoie `([C_D, SER], cv)`. Lire la table est
  **strictement équivalent** à appeler le simulateur, puisqu'il est déterministe, mais des milliers de fois
  plus rapide : on peut répéter les expériences sur des dizaines de graines.

#### `problem_v2.py` — le problème V2 (XFOIL)
- `SpaceV2` : grille $m$ (8 niveaux, 0–7 %) × $p$ (4 : 0,25–0,55) × $t$ (8 : 6–20 %) × $I$ (4 : 2, 4, 6, 8)
  = **1 024 profils, 10 bits**.
- `CaseV2` : $Re = 3\cdot10^6$, croisière $C_L = 0{,}4$, $C_{L,\max} \ge 1{,}2$ sur la rampe 4°–14°,
  $t \ge 8\,\%$, $T = 0{,}3$, secteurs S1 et S2.
- `simulate_v2(idx)` : NACA modifié → XFOIL croisière ($C_D$, $\alpha$) → XFOIL rampe ($C_{L,\max}$) → SER
  S1 et S2 (optique physique) → contraintes.
- `build_table_v2()` : les 1 024 profils en parallèle (10 processus) → `resultats/base_V2.csv`, relu
  ensuite comme **cache**.
- `with_objective(table, sector, source)` : choisit **quelle SER** sert d'objectif : secteur S1 ou S2,
  source `"PO"` (optique physique) ou `"MoM"` (méthode des moments, relue dans `ser_mom_V2.csv`). **Toutes
  les expériences V2 utilisent `"MoM"`.**

#### `qubo_loop.py` — la boucle QUBO
| Partie | Fonctions | Rôle |
|---|---|---|
| Modèles | `z_of`, `fit_quadratic`, `fit_linear`, `predict` | indices → $z \in [-1,1]$ ; moindres carrés (légère régularisation ridge $10^{-6}$) sur les monômes de degré ≤ 2 (ou ≤ 1 pour $C_L$) |
| Pénalités | `grow`, `add_term`, `penalty_index_nonzero`, `penalty_linear_geq`, `lin_bits` | agrandir $Q$ pour les qubits auxiliaires ; Rosenberg ; variable d'écart ; modèle affine → coefficients par bit |
| QUBO | `build_qubo(...)` | somme pondérée normalisée + pénalités → `(Q, c, n_qubits)`. Poids de pénalité $P = 3$, multiplié par `cl_weight = 25` pour $C_L$ |
| Solveurs | `decode`, `ranked_candidates(Q, c, solver)` | chaînes de bits → indices de profil (on jette les qubits auxiliaires) ; liste des profils classés par le solveur `"exact"` (énumération, 400 meilleurs états), `"sa"` (recuit, 40 lectures) ou `"qaoa"` ($p = 2$, 512 tirs) |
| Boucle | `qubo_loop(problem, n_init=10, weights, k_per_w=1, solver, budget, seed, cons, stop_when, ...)` | 10 profils initiaux au hasard, puis les itérations de §2.4. S'il y a moins de 4 profils faisables, exploration aléatoire |

#### `bayes.py` — optimisation bayésienne (V3, lot L1)
- Modèles : `GPModel` (krigeage SMT, noyau Matérn 5/2, bruit estimé), `QuadModel` (régression
  quadratique **bayésienne** : mêmes monômes que la boucle QUBO, mais avec une variance), `_Constant`
  (repli s'il y a trop peu de points).
- Acquisition : `expected_improvement`, `tchebychev` (scalarisation de ParEGO, qui atteint aussi les
  parties **non convexes** du front, contrairement à la somme pondérée).
- `bayes_loop(problem, model, selection, scalar, constraint, ...)` : **même structure** que `qubo_loop`
  (mêmes 10 points initiaux pour une graine donnée, mêmes 5 poids, un profil par poids et par itération,
  même compteur) : la comparaison est **appariée**. L'acquisition est évaluée sur **tous** les profils non
  simulés (la grille est énumérable).

### Les expériences (scripts qui produisent `resultats/`)

| Script | Itération | Ce qu'il compare | Sortie |
|---|---|---|---|
| `experiences_pipeline.py` | V1 | hasard, NSGA-II ($N = 16$), boucle QUBO exact / recuit / QAOA, sur les 512 profils du modèle rapide | `pipeline.json` |
| `check_pymoo.py` | V1 | notre NSGA-II contre celui de pymoo, mêmes réglages, 20 graines | `check_pymoo.json` |
| `experiences_v2.py` | V2 | hasard, NSGA-II ($N = 8$ et $16$), boucle QUBO exact / recuit / QAOA, sur la table XFOIL + SER MoM ; ajoute l'**arrêt réaliste** | `pipeline_V2.json` |
| `experiences_l1.py` | V3 (L1) | boucle QUBO exacte contre l'ablation bayésienne de `bayes.py` (8 variantes), mêmes graines | `pipeline_L1_V3.json` |
| `diagnostic_l1.py` | V3 (L1) | **pourquoi** ces résultats : effet du poids de pénalité, part des profils proposés qui violent la contrainte, erreur des modèles selon le nombre de points | `diagnostic_L1_V3.json` |

Fonctions communes à ces scripts : `setup` (table, front exact, normalisation, HV exact), `metrics` /
`state` (fraction du front et HV relatif après chaque itération), `run_*` (une méthode sur plusieurs
graines), `summary` (quartiles de `full_front`, `hv99`, `success`, arrêt réaliste), `stop_point`.

`__init__.py` est vide.

---

## 4. Qui utilise quoi : interactions avec le reste du code

```
aero/design_space.py, aero/model.py ──► problem.py ──────────────┐
ser/model.py ─────────────────────────► problem.py, problem_v2.py │
aero/naca4.py, aero/xfoil_runner.py ──► problem_v2.py ────────────┤
resultats/ser_mom_V2.csv (ser/compare_po_mom.py) ► problem_v2.py   │
                                                                   ▼
quantique/qubo.py, recuit.py, qaoa.py ──► qubo_loop.py ──► experiences_pipeline.py, experiences_v2.py,
                                              ▲              experiences_l1.py, diagnostic_l1.py
quantique/qubo.py (quadratic_features) ──► bayes.py ─────┘
pareto.py, nsga2.py ──────────────────────► toutes les expériences

Utilisateurs extérieurs de optim/ :
  quantique/montee_en_taille.py   (fit_quadratic)
  quantique/anatomie_qubo.py      (problem_v2, qubo_loop, experiences_v2.CONS)
  ia_qubo/*                       (problem_v2, qubo_loop, problem.CountedProblem, experiences_v2)
  verifier_installation.py        (simulate_v2, experiences_v2.setup/run)
  Cours/*/tools, Fiche_Projet/tools (relisent les JSON de resultats/)
```

---

## 5. Les résultats à connaître

### 5.1 V1 : modèle aéro rapide, 512 profils
*(`resultats/pipeline.json` ; 392 profils faisables ; front exact : 7 profils en S1, 15 en S2 ; médiane du
nombre de simulations pour retrouver le front complet)*

| Méthode | S1 | S2 |
|---|---|---|
| Hasard | 454 | 494 |
| NSGA-II ($N = 16$) | 106,5 | 106 |
| Boucle QUBO + exact | **35** | **67,5** |
| Boucle QUBO + recuit simulé | 40 | 85 |
| Boucle QUBO + QAOA | **35** | 80 |

La boucle QUBO divise par **~3** le nombre de simulations par rapport à NSGA-II, et par **~13** par rapport
au hasard. Notre NSGA-II et pymoo : 106,5 contre 95 (S1), 106 contre 110,5 (S2) : même comportement
(`check_pymoo.json`).

### 5.2 V2 : XFOIL + SER par méthode des moments, 1 024 profils
*(`resultats/pipeline_V2.json` ; 702 faisables ; front exact : 4 profils en S1, 10 en S2 ; budget
250 simulations, 150 pour QAOA)*

| Méthode | Front complet (médiane) S1 / S2 | HV 99 % (médiane) S1 / S2 | Succès S1 / S2 |
|---|---|---|---|
| Hasard | 911 / 972 | 632 / 740 | — |
| NSGA-II ($N = 8$ et $16$) | jamais dans le budget | jamais dans le budget | 0–5 % |
| Boucle QUBO + exact | **75** / **142,5** | **62,5** / **75** | 100 % / 100 % |
| Boucle QUBO + recuit | 97,5 / 235 | 62,5 / 90 | 100 % / 55 % |
| Boucle QUBO + QAOA (5 graines) | 105 / — | 75 / 70 | 100 % / 60 % |

**Lecture** :
- Le problème V2 est **plus dur** que V1 : front au bord de la contrainte $C_{L,\max}$, objectifs moins
  réguliers (XFOIL). NSGA-II ne retrouve **jamais** le front complet en 250 simulations ; avec l'arrêt
  réaliste, il s'arrête vers 78–151 simulations avec 50 à 80 % du front.
- La boucle QUBO atteint 99 % de l'hypervolume en **60 à 90 simulations**, environ **10 fois moins** que le
  hasard.
- Les solveurs approchés (recuit, QAOA) sont un peu moins bons que l'exact pour retrouver **tout** le
  front, mais aussi bons pour l'hypervolume. Le solveur n'est pas le facteur limitant (voir
  `quantique/README_quantique.md`, §5.4).

### 5.3 V3 (L1) : la boucle QUBO face à l'optimisation bayésienne
*(`resultats/pipeline_L1_V3.json`, 20 graines, mêmes points initiaux ; médiane du front complet, S1 / S2)*

| Méthode | Modèle | Sélection | Front complet | HV 99 % |
|---|---|---|---|---|
| **Boucle QUBO exacte** | quadratique | minimum | **75** / 142,5 | **62,5** / 75 |
| `quad_mean` (boucle sans encodage binaire) | quadratique | minimum | 75 / **95** | 65 / 75 |
| `gp_mean` | krigeage | minimum | 87,5 / 110 | 72,5 / 85 |
| `quad_ei` | quadratique | EI | 85 / 117,5 | 72,5 / 85 |
| `gp_ei` (optimisation bayésienne) | krigeage | EI | 125 / 127,5 | 100 / 110 |
| `parego` | krigeage | EI, Tchebychev | 157,5 / 170 | 125 / 140 |
| `quad_mean_pof` | quadratique | minimum, PoF stricte | **jamais** (0 %) | — |
| `gp_ei_dur` | krigeage | EI, contrainte en dur | 5 % / 70 % de succès | — |

**Lecture** :
- Sur ce problème, la boucle QUBO fait **aussi bien ou mieux** que l'optimisation bayésienne classique
  (`gp_ei`, ParEGO). Le modèle quadratique simple, exploité directement, suffit : le krigeage et l'EI
  n'apportent rien ici.
- `quad_mean` ≈ boucle QUBO en S1 : c'est **le modèle et la sélection** qui font la performance, pas
  l'encodage en bits. (En S2, la version sans bits est meilleure, 95 contre 142,5 : l'encodage, avec ses
  variables d'écart, coûte un peu.)
- **Les contraintes strictes échouent** : le front est **au bord** de $C_{L,\max} \ge 1{,}2$. Une méthode
  qui écarte tout profil « prédit infaisable » écarte aussi les profils du front, car le modèle de
  $C_{L,\max}$ est imprécis. La **pénalité douce** de la boucle QUBO laisse passer ces profils limites.

### 5.4 Pourquoi : les diagnostics
*(`resultats/diagnostic_L1_V3.json`)*

- **Qualité des modèles** (erreur RMS rapportée à l'écart-type, sur les profils non vus, avec 160 points
  d'entraînement) : SER très bien prédite (0,07 en quadratique, 0,02 en krigeage) ; **$C_D$ mal prédit**
  (0,78 et 0,76) ; $C_{L,\max}$ moyen (0,54 et 0,68). Le modèle est donc **imprécis en valeur** mais assez
  bon pour **orienter** la recherche : c'est pour ça qu'on boucle.
- **Poids de la pénalité** (S1) : réglage V2 (×25) → front en 75 simulations ; ×2,5 → 85 ; ×250 → 107,5 ;
  sans contrainte → 112,5. Il y a un **optimum** : trop douce, on propose trop de profils infaisables ;
  trop dure, on écarte le bord de la contrainte, là où est le front.

---

## 6. Questions probables du jury

**« Pourquoi compter les simulations et pas le temps de calcul ? »**
Parce que, dans un vrai bureau d'études, une simulation CFD ou électromagnétique coûte des minutes à des
heures, alors que le reste (ajuster un modèle, résoudre un QUBO de 14 qubits : quelques millisecondes) est
négligeable. Le nombre de simulations est la métrique qui reste valable quand on change de simulateur.

**« Vous trichez en lisant une table pré-calculée ? »**
Non : le simulateur est **déterministe**, donc lire la table donne exactement le même résultat que
relancer XFOIL. `CountedProblem` compte chaque profil distinct demandé comme une simulation. La table
permet seulement de répéter les expériences sur 20 graines, et de connaître le **front exact** pour
évaluer objectivement les méthodes.

**« Pourquoi une somme pondérée ? Elle rate les parties non convexes du front. »**
C'est vrai, et c'est une limite connue. On l'a choisie parce qu'elle donne directement un QUBO
(quadratique). On a testé l'alternative (Tchebychev, ParEGO, dans `bayes.py`) : sur ce problème, elle fait
moins bien. Et les profils d'un front non convexe peuvent quand même être simulés au passage, puisque le
front rendu est calculé sur **tous** les profils simulés.

**« Pourquoi un modèle quadratique, alors qu'il prédit mal $C_D$ ? »**
Parce qu'un QUBO **impose** un modèle quadratique en bits. Le modèle n'a pas besoin d'être précis : il doit
**proposer** de bons candidats, qui sont ensuite **vraiment simulés**, puis il est réajusté avec les
nouveaux points. Le diagnostic montre que le krigeage, plus flexible, ne fait pas mieux dans la boucle.

**« Pourquoi NSGA-II est-il si mauvais ? »**
C'est un algorithme **sans modèle**, conçu pour des budgets de milliers d'évaluations. Sur un budget de
quelques dizaines à centaines de simulations, il n'a pas le temps de converger. C'est précisément le
régime où les méthodes à modèle de substitution sont utiles. Nous l'avons **réglé** ($N = 8$ en plus de
16) et **vérifié** contre pymoo pour que la comparaison soit honnête.

**« Quel est l'apport du quantique dans tout ça ? »**
La boucle QUBO est **compatible** avec un solveur quantique (QAOA, recuit quantique) : c'est le format
natif de ces machines. Nos résultats montrent que la boucle est efficace, **quel que soit le solveur** ; à
14 qubits, le solveur classique exact suffit. L'intérêt quantique viendrait sur des problèmes beaucoup plus
grands (voir `ia_qubo/`, problème à 30 bits, et `quantique/README_quantique.md`).

**« Comment gérez-vous les contraintes ? »**
Dans le QUBO, par des **pénalités** : exacte pour l'épaisseur (Rosenberg, 1 qubit auxiliaire), approchée
pour la portance max (modèle affine + variable d'écart, 3 qubits). Le poids de pénalité a été réglé
(×25) : le diagnostic montre qu'il existe un optimum, car le front est **au bord** de la contrainte.

---

## 7. Commandes

Depuis le dossier `Code/` (sous Windows, `python` au lieu de `python3`) :

```bash
python3 -m optim.problem_v2                 # table XFOIL des 1 024 profils (≈ 6 min sur 10 cœurs) -> resultats/base_V2.csv
python3 -m optim.experiences_pipeline       # comparaison V1 (quelques minutes)  -> resultats/pipeline.json
python3 -m optim.check_pymoo                # notre NSGA-II contre pymoo         -> resultats/check_pymoo.json
python3 -m optim.experiences_v2             # comparaison V2 (≈ 1 h)             -> resultats/pipeline_V2.json
python3 -m optim.experiences_v2 PO rapide   # version rapide, SER par optique physique
python3 -m optim.experiences_l1             # boucle QUBO contre bayésien (V3)   -> resultats/pipeline_L1_V3.json
python3 -m optim.experiences_l1 rapide      # 4 graines                          -> resultats/pipeline_L1_V3_rapide.json
python3 -m optim.diagnostic_l1              # diagnostics (quelques minutes)     -> resultats/diagnostic_L1_V3.json
```

Petit test interactif (sans XFOIL, sur la table V2 déjà calculée) :

```python
from optim.problem import CountedProblem
from optim.problem_v2 import build_table_v2, with_objective, SpaceV2, KEYS
from optim.qubo_loop import qubo_loop
from optim.experiences_v2 import CONS

SP = SpaceV2()
T = with_objective(build_table_v2(), "S1", "MoM")       # relit resultats/base_V2.csv
pb = CountedProblem(T, keys=KEYS)
out = qubo_loop(pb, solver="exact", budget=60, cons=CONS, bits=SP.bits, levels=SP.n_levels,
                keys=KEYS, lin_col="CLmax")
print(pb.n_sims, "simulations ; historique :", out["history"])
```
