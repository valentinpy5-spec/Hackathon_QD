# Le dossier `ia_qubo/` expliqué

> **Dernière mise à jour : 28/11/2026 à 17h00**, d'après le code du commit `30d54af` (branche
> `branch_meddi`). Si un fichier de `ia_qubo/` a été modifié après cette date (`git log -- Code/ia_qubo`),
> ce README n'est peut-être plus à jour. Le `README.md` du dossier (fiche courte : fichiers et commandes)
> reste valable ; ce document-ci explique **en détail**.

Ce document explique **ce que fait chaque fichier du dossier `ia_qubo/`**, **comment il réutilise
`optim/` et `quantique/`**, et **juste assez de théorie** pour comprendre le code et répondre au jury. Les
chiffres cités sortent des fichiers de `resultats/` (on dit lequel à chaque fois).

> **En une phrase.** Dans la boucle QUBO de la V2, le modèle de substitution est un simple polynôme de
> degré 2. Ici (V3, lot L6), on le remplace par des **modèles appris** issus de la littérature
> « IA + QUBO » (**FMQA**, **BOCS**, réseau de neurones), on teste une contrainte traitée en **boîte
> noire**, et on passe à un **problème beaucoup plus grand** (profils CST, **30 bits**, ~10⁹ profils) où
> l'on ne peut plus tout énumérer.

---

## Sommaire

1. [Où se place `ia_qubo/` dans le projet](#1-où-se-place-ia_qubo-dans-le-projet)
2. [La théorie minimale](#2-la-théorie-minimale)
3. [Les fichiers un par un](#3-les-fichiers-un-par-un)
4. [Qui utilise quoi : interactions avec le reste du code](#4-qui-utilise-quoi--interactions-avec-le-reste-du-code)
5. [Les résultats à connaître](#5-les-résultats-à-connaître)
6. [Questions probables du jury](#6-questions-probables-du-jury)
7. [Commandes](#7-commandes)

---

## 1. Où se place `ia_qubo/` dans le projet

La boucle QUBO (`optim/qubo_loop.py`) a trois réglages : le **modèle** de substitution, la **façon de
traiter la contrainte** $C_{L,\max} \ge 1{,}2$, et le **solveur** du QUBO. `ia_qubo/` réécrit la boucle de
façon **générique** pour faire varier chacun séparément :

```
                     boucle_ia(problem, spec, modele, contrainte, solveur)        (boucle.py)
                                            │
   modele :      "quadz"  polynôme de degré 2 en z          ← l'approche V2, à l'identique
                 "fm"     machine de factorisation (FMQA)   ┐
                 "bocs"   quadratique bayésien sur les bits ├ des QUBO appris   (modeles.py)
                 "mlp"    réseau de neurones                ┐ PAS des QUBO : recherche directe
                 "hybride" réseaux pour C_D/SER + affine pour C_L,max ┘
   contrainte :  "modele"       C_L,max modélisé + variables d'écart (comme en V2)
                 "boite_noire"  pas de modèle de contrainte : les infaisables reçoivent une mauvaise note
   solveur :     "exact" (énumération, ≤ 22 qubits)  ou  "recuit" (recuit simulé vectorisé)
                                            │
             ┌──────────────────────────────┴─────────────────────────────┐
             ▼                                                            ▼
   problème V2 (1 024 profils, front exact connu)             problème CST (30 bits, ~10⁹ profils)
   optim/problem_v2.py + CountedProblem                       probleme_cst.py : ProblemeSimule
   experiences_v2_ia.py, diagnostic_v2_ia.py                  XFOIL + SER à la demande, cache
                                                              experiences_echelle.py
```

**Garde-fou** : `verifier_boucle()` vérifie que `boucle_ia(modele="quadz", contrainte="modele",
solveur="exact")` simule **exactement les mêmes profils, dans le même ordre** que `qubo_loop` de la V2.
Les comparaisons sont donc honnêtes : seule la brique changée diffère.

`ia_qubo/` **ne modifie aucun fichier existant** : il importe les briques de `optim/` et `quantique/`.

---

## 2. La théorie minimale

### 2.1 Pourquoi apprendre le QUBO ?

Un QUBO, c'est $E(b) = \sum_i Q_{ii} b_i + \sum_{i<j} Q_{ij} b_i b_j + c$. En V2, on obtenait $Q$ en
ajustant un polynôme de degré 2 sur les variables **physiques** normalisées $z$, puis en substituant
l'encodage binaire $z = a_0 + A\,b$ (`quantique/qubo.py : surrogate_to_qubo`). Ce modèle a **peu de
paramètres** (15 pour 4 variables) mais il est **rigide**.

L'idée de la littérature « IA + QUBO » : **apprendre directement $Q$ sur les bits**, avec un modèle qui
**reste un QUBO par construction**. On peut alors toujours le donner à un solveur quantique.

### 2.2 FMQA : la machine de factorisation (Kitai et al., 2020)

$$\hat y(b) = w_0 + \sum_i w_i b_i + \sum_{i<j} \langle v_i, v_j\rangle\, b_i b_j, \qquad v_i \in \mathbb{R}^k$$

- C'est un QUBO : $Q_{ii} = w_i$, $Q_{ij} = \langle v_i, v_j\rangle$.
- Au lieu d'apprendre les $n(n-1)/2$ couplages un par un, on apprend un **vecteur** $v_i$ de dimension
  $k$ par bit (ici $k = 4$) : $1 + n + nk$ paramètres. C'est une **factorisation de rang faible** de la
  matrice des couplages, qui évite le sur-apprentissage avec peu de données.
- Ajustement par descente de gradient (Adam) avec pénalité L2 ; moyenne de 3 machines (une moyenne de
  QUBO est encore un QUBO).

### 2.3 BOCS : quadratique bayésien sur les bits (Baptista & Poloczek, 2018)

- Modèle **complet** sur les bits : $[1, b_i, b_i b_j]$, soit $1 + n + n(n-1)/2$ coefficients.
- **Régression bayésienne** : a priori gaussien sur les coefficients, hyperparamètres (précision de
  l'a priori et du bruit) réglés par maximum de l'évidence (MacKay). On obtient une **loi a posteriori**
  sur les coefficients, pas une seule valeur.
- **Échantillonnage de Thompson** : à chaque itération, on **tire un modèle au hasard** dans cette loi et on
  minimise **ce** QUBO. Là où le modèle est incertain, les tirages varient : ça crée de l'**exploration**
  naturellement.

### 2.4 Le réseau de neurones (témoin, pas un QUBO)

Ensemble de 5 perceptrons à 2 couches cachées (32 neurones, tanh), entrée $z$, écrits en NumPy. C'est le
modèle **le plus flexible**, mais ce **n'est pas un QUBO** : on ne peut pas le donner à un ordinateur
quantique. On le minimise par **énumération** (V2) ou par **recherche locale** (CST). Il sert de
**témoin** : « un modèle plus précis fait-il mieux ? ».

La variante **hybride** utilise des réseaux pour $C_D$ et la SER, mais garde le modèle **affine** de la V2
pour $C_{L,\max}$.

### 2.5 Contrainte modélisée ou en boîte noire

- **Modélisée** (V2) : on ajuste un modèle affine de $C_{L,\max}$ et on ajoute au QUBO une pénalité avec
  **3 qubits d'écart** (+1 qubit de Rosenberg pour l'épaisseur) : 14 qubits.
- **Boîte noire** (pratique de FMQA) : aucun modèle de contrainte. Les profils infaisables entrent dans
  les données avec une **note dégradée** (pire faisable + 20 % de l'étendue, `PENALITE_INFAISABLE`) et le
  modèle **apprend seul** à les éviter. Pas de variables d'écart : 11 qubits.

### 2.6 Le problème mis à l'échelle : profils CST

**CST** (*Class-Shape Transformation*, Kulfan 2008) : une forme de profil très utilisée en industrie.
$$y(x) = \underbrace{\sqrt{x}\,(1-x)}_{\text{classe : nez rond, BF pointu}} \;\times\; \underbrace{\sum_{i=0}^{4} A_i \binom{4}{i} x^i (1-x)^{4-i}}_{\text{forme : polynômes de Bernstein}}$$

- 5 coefficients pour l'extrados + 5 pour l'intrados = **10 variables**, 8 niveaux chacune (3 bits) :
  **30 bits, $8^{10} \approx 10^9$ profils**. Les bornes couvrent les NACA de 0 à 6 % de cambrure et 8 à
  18 % d'épaisseur (+10 % de marge).
- **On ne peut plus énumérer** ni la grille ni le QUBO ($2^{30}$ à $2^{33}$ états) : les solveurs sont
  heuristiques (recuit simulé, recherche locale), comme sur un vrai problème industriel.
- **Le front exact n'est plus connu** : on mesure l'hypervolume par rapport à un **front de référence**
  = les meilleurs profils trouvés par **toutes** les méthodes et graines réunies.
- Évaluation par **XFOIL** (croisière $C_L = 0{,}4$, rampe de manœuvre) et **SER par optique physique**
  (secteur S2 ; la méthode des moments serait trop lente pour des milliers de profils, et la PO garde le
  classement : corrélation de rang 0,999, voir `ser/README_SER.md`).
- **Filtre géométrique gratuit** (seconde campagne, `FILTRE = True`) : avant de simuler, on vérifie en
  quelques microsecondes que l'extrados reste au-dessus de l'intrados et que $t \ge 8\,\%$. Un profil rejeté
  n'est **pas compté** comme simulation, pour toutes les méthodes. Une traînée > 200 counts est comptée comme
  un échec d'XFOIL (`CD_ABERRANT`).

---

## 3. Les fichiers un par un

### `modeles.py` — les quatre familles de modèles

Même interface pour tous : `fit(B, y)`, `predict(B)`, et `to_qubo()` quand le modèle est un QUBO. Toutes les
sorties sont **centrées-réduites** avant l'ajustement (`_Scaler`).

| Élément | Rôle |
|---|---|
| `encode(I, bits)`, `decode_bits(B, bits)` | indices de grille ↔ bits (poids faible en premier, même convention que partout) |
| `bits_quad_features(B)`, `theta_to_qubo(theta, n)` | monômes $[1, b_i, b_ib_j]$ et coefficients → $(Q, c)$ |
| `QuadZ` | l'approche V2 : `fit_quadratic` en $z$, `to_qubo` = `surrogate_to_qubo` |
| `FactorizationMachine` | FMQA (§2.2) ; `to_qubo` reconstruit $Q_{ij} = \langle v_i, v_j\rangle$ et dé-normalise |
| `BayesQuadBits` | BOCS (§2.3) ; `sample_qubo()` = tirage de Thompson (Cholesky de la covariance a posteriori) |
| `MLPEnsemble` | réseau de neurones (§2.4), rétropropagation et Adam écrits à la main ; `predict_std` = écart entre membres |

### `boucle.py` — la boucle générique

| Fonction | Rôle |
|---|---|
| `recuit(Q, c, n_sweeps, n_reads, ...)` | recuit simulé **vectorisé** : les 64–128 lectures avancent **en parallèle** (matrices NumPy), bien plus rapide que `quantique/recuit.py` sur 33 qubits |
| `classer(Q, c, solveur, rng, n_design, T_obj)` | états classés par énergie (exact : 600 meilleurs ; recuit : lectures triées), réduits aux bits de conception. **Amélioration** : la température du recuit est réglée sur l'échelle de l'**objectif**, pas des pénalités (meilleure de 64 lectures = optimum dans 67 % des cas, contre 39 % avec le réglage V2) |
| `ajoute_contraintes(Q, c, coef_cl, spec, cl_min, ...)` | ajoute à un QUBO appris **les mêmes pénalités qu'en V2** (Rosenberg + variables d'écart), en réutilisant les fonctions de `optim/qubo_loop.py` |
| `fabrique(modele, spec, seed)` | crée le bon modèle |
| `recherche_directe(score, spec, rng, ...)` | pour les modèles qui ne sont pas des QUBO : énumération si la grille a ≤ 50 000 profils, sinon 20 000 tirages puis **recherche locale** depuis les 20 meilleurs (changer une variable à la fois) |
| `boucle_ia(problem, spec, modele, contrainte, solveur, ...)` | la boucle : 10 profils initiaux, 5 poids $w$, un profil nouveau par poids et par itération. `spec` décrit le problème (bits, niveaux, clés, variable d'épaisseur, initialisation et filtre éventuels) |
| `verifier_boucle()` | le garde-fou (§1) |

### `probleme_cst.py` — le problème à 30 bits

| Élément | Rôle |
|---|---|
| `bernstein_matrix`, `classe`, `cst_coords(Au, Al)` | géométrie CST, même convention de contour que `aero/naca4.py` |
| `fit_cst(X, Y)`, `bornes()` | coefficients CST qui approchent un NACA ; bornes = enveloppe de 60 NACA + 10 % |
| `VALEURS`, `KEYS`, `SPEC` | les 8 niveaux de chaque coefficient, les clés `a0..a9`, la description pour `boucle_ia` |
| `geometrie(idx)`, `admissible(idx)` | contrôle géométrique gratuit (faces sans croisement, $t \ge 8\,\%$) |
| `init_aleatoire(rng, n)` | tirages initiaux (admissibles si `FILTRE`) |
| `evaluer(idx)` | **une vraie simulation** : XFOIL croisière + rampe, puis SER PO (secteur S2), puis contraintes |
| `corrige(r)`, `_charge_cache()` | seuil de traînée aberrante ; relecture de `resultats/cache_cst.jsonl` |
| `ProblemeSimule` | **même interface que `CountedProblem`**, mais simule à la demande ; chaque profil est **mis en cache** (une seule simulation par profil, toutes méthodes et graines confondues). Le compteur, lui, compte les profils distincts demandés par **chaque** exécution |

### Les expériences

| Script | Ce qu'il fait | Sortie |
|---|---|---|
| `experiences_v2_ia.py` | 8 méthodes (`quadz`, `fm`, `bocs`, `mlp` × contrainte `modele` / `bn`) + `hybride_modele`, sur S1 et S2, 20 graines, budget 250, même protocole que `optim/experiences_v2.py` | `ia_qubo_V2.json` |
| `diagnostic_v2_ia.py` | pour chaque profil du front exact : est-il **supporté** (atteignable par une somme pondérée) ? Avec quelle fréquence chaque modèle le trouve-t-il ? | `ia_qubo_diagnostic_V2.json` |
| `precision_modeles.py` | erreur et corrélation de rang de chaque modèle selon le nombre de données (10 à 500), sur V2 et sur le cache CST | `ia_qubo_precision.json` |
| `qaoa_contraintes.py` | QAOA $p = 2$ sur des QUBO à contrainte **modélisée** (14 qubits) ou en **boîte noire** (11 qubits) | `ia_qubo_qaoa_contraintes.json` |
| `experiences_echelle.py` | hasard, NSGA-II, `quadz`, `fm` (×2), `bocs`, `hybride` sur le problème CST, 8 graines, budget 300 | `ia_qubo_echelle.json` (+ `_brut`, et la 1ʳᵉ campagne sans filtre : `ia_qubo_echelle_sansfiltre*.json`) |

`__init__.py` est vide.

---

## 4. Qui utilise quoi : interactions avec le reste du code

```
quantique/qubo.py (surrogate_to_qubo, all_bitstrings, qubo_energy) ─► modeles.py, boucle.py, qaoa_contraintes.py
quantique/qaoa.py ──────────────────────────────────────────────────► qaoa_contraintes.py
optim/qubo_loop.py (fit_quadratic, fit_linear, predict, z_of,
                    grow, penalty_*, lin_bits) ──────────────────────► modeles.py, boucle.py, qaoa_contraintes.py
optim/problem.py (CountedProblem), optim/problem_v2.py ─────────────► experiences_v2_ia.py, diagnostic_v2_ia.py,
optim/experiences_v2.py (setup, state, summary, CONS)                  precision_modeles.py, qaoa_contraintes.py
optim/nsga2.py, optim/pareto.py ────────────────────────────────────► experiences_echelle.py
aero/naca4.py (naca4), aero/xfoil_runner.py (run_xfoil) ────────────► probleme_cst.py
ser/po2d.py (echo_width, radar_directions, to_db) ──────────────────► probleme_cst.py
```

Rien en dehors de `ia_qubo/` n'importe `ia_qubo/` : c'est un **lot d'expériences autonome**. Ses JSON sont
lus par le cours `Cours/6_IA_QUBO/` et la fiche projet.

---

## 5. Les résultats à connaître

### 5.1 Sur le problème V2 (front exact connu)
*(`resultats/ia_qubo_V2.json`, 20 graines, budget 250 ; médiane du nombre de simulations, S1 / S2)*

| Méthode | Front complet | HV 99 % | Succès |
|---|---|---|---|
| **`quadz_modele` (boucle V2)** | **75** / 142,5 | **62,5** / **75** | 100 % / 100 % |
| `fm_modele` (FMQA) | 95 / 152,5 | 85 / 112,5 | 100 % / 95 % |
| `bocs_modele` (BOCS) | 100 / 187,5 | 82,5 / 125 | 100 % / 80 % |
| `mlp_modele` (réseau) | jamais | — | 5 % / 5 % |
| `hybride_modele` (réseaux + affine) | 102,5 / **90** | 72,5 / 82,5 | 100 % / 100 % |
| toutes les variantes **boîte noire** (`*_bn`) | jamais | jamais | 0 % |

**Lecture** :
- Sur ce petit problème, **le modèle le plus simple (V2) reste le meilleur ou à égalité**. Les QUBO appris
  (FMQA, BOCS) ont beaucoup plus de paramètres que de données au début de la boucle.
- **La contrainte en boîte noire échoue partout** : le front est **au bord** de $C_{L,\max} \ge 1{,}2$, et un
  modèle qui apprend « infaisable = mauvaise note » repousse aussi les profils limites du front.
- Le **réseau de neurones**, pourtant **le plus précis** (§5.2), trouve **le moins bien** le front. Le
  diagnostic explique pourquoi (§5.3). L'**hybride** fonctionne, parce qu'il garde le modèle affine de la
  contrainte.

### 5.2 Précision des modèles
*(`resultats/ia_qubo_precision.json`, 500 profils d'entraînement ; erreur RMS / écart-type, et corrélation
de rang de Spearman)*

| Grandeur | Quadratique en z | FMQA | BOCS | Réseau |
|---|---|---|---|---|
| $C_D$ (V2) | 0,36 / 0,94 | 0,34 / 0,95 | 0,35 / 0,94 | **0,22 / 0,98** |
| SER (V2) | 0,13 / 0,99 | 0,14 / 0,99 | 0,12 / 0,99 | **0,07 / 1,00** |
| $C_{L,\max}$ (V2) | 0,52 / 0,92 | 0,53 / 0,93 | 0,54 / 0,92 | 0,49 / 0,94 |
| $C_D$ (CST) | 0,67 / 0,81 | 0,76 / 0,74 | 0,73 / 0,73 | **0,62 / 0,84** |

La SER se prédit très bien ; $C_D$ et surtout $C_{L,\max}$ sont plus difficiles. Le réseau est le plus
précis, mais l'écart avec le quadratique est faible.

### 5.3 Pourquoi un modèle plus précis ne trouve pas mieux le front
*(`resultats/ia_qubo_diagnostic_V2.json`)*

Un profil du front est **supporté** si une somme pondérée $w\,C_D + (1-w)\,\text{SER}$ des **vrais** objectifs
le désigne comme minimum. Sinon il est dans un « creux » non convexe du front. En S1, 2 des 4 profils du front
sont **non supportés** ; en S2, 6 sur 10.

- Un modèle **précis** reproduit fidèlement la somme pondérée : il **ne désigne jamais** les profils non
  supportés. Le réseau ne trouve les 2 profils non supportés de S1 que dans **58 %** des graines.
- Un modèle **grossier** (quadratique) se trompe un peu, et cette erreur lui fait atteindre ces profils
  **« par accident »** : 100 % des graines.

**Message pour le jury** : la limite n'est pas la précision du modèle, c'est la **somme pondérée**. Avec une
recherche précise, il faudrait une autre scalarisation (Tchebychev, contrainte epsilon...).

### 5.4 À l'échelle : le problème CST à 30 bits
*(`resultats/ia_qubo_echelle.json`, seconde campagne avec filtre, 8 graines, budget 300 ; hypervolume
relatif au front de référence, médiane ; 16 244 profils simulés au total)*

| Méthode | HV à 100 simulations | HV à 300 simulations | Profils simulés faisables |
|---|---|---|---|
| Hasard | 0,68 | 0,77 | 77 % |
| NSGA-II ($N = 16$) | 0,72 | 0,81 | 74 % |
| `quadz_modele` (boucle V2, QUBO de 33 qubits, recuit) | 0,77 | 0,88 | 42 % |
| `fm_modele` (FMQA) | **0,83** | 0,90 | 46 % |
| `fm_bn` (FMQA, boîte noire) | 0,78 | 0,83 | 83 % |
| `bocs_modele` (BOCS) | **0,83** | 0,90 | 43 % |
| `hybride_modele` (témoin non QUBO) | 0,81 | **0,91** | 38 % |

**Lecture** :
- **À grande échelle, les QUBO appris deviennent utiles** : FMQA et BOCS atteignent 0,83 d'hypervolume en
  100 simulations, contre 0,77 pour le modèle V2, 0,72 pour NSGA-II et 0,68 pour le hasard. Le modèle
  quadratique en $z$ (66 coefficients pour 10 variables) est trop rigide pour des formes CST.
- Toutes les boucles à modèle battent nettement NSGA-II et le hasard.
- Le témoin hybride, qui n'est pas un QUBO, finit légèrement devant : à budget égal, l'avantage vient
  surtout du **modèle**, pas du format QUBO. Mais FMQA/BOCS ont l'avantage de rester **compatibles avec un
  solveur quantique**, et c'est à cette taille (33 qubits, non énumérable) qu'un solveur quantique aurait
  un sens.
- Les boucles à modèle « gaspillent » plus de simulations sur des infaisables (38–46 %), car elles vont
  chercher **au bord** de la contrainte, là où est le front.
- La première campagne, **sans filtre géométrique**, est gardée pour comparaison : le modèle en $z$ y
  dépensait 59 % de son budget sur des géométries invalides. Le filtre est un post-traitement **classique et
  gratuit** des échantillons du solveur.

### 5.5 QAOA et contrainte en boîte noire
*(`resultats/ia_qubo_qaoa_contraintes.json`)*

Traiter la contrainte en boîte noire supprime 3 qubits d'écart (14 → 11) et fait passer la part de
l'objectif dans l'échelle des énergies de ~1–3 % à ~5–10 %. Mais le gain de QAOA ($p = 2$) sur le hasard
reste du même ordre (quelques unités). Voir `quantique/README_quantique.md`, §5.4.

---

## 6. Questions probables du jury

**« Qu'est-ce que FMQA / BOCS, et pourquoi les tester ? »**
Ce sont les deux méthodes de référence de la littérature pour **apprendre un QUBO** à partir de
simulations, puis le résoudre (souvent sur machine D-Wave). FMQA apprend des couplages factorisés (rang
faible), BOCS un modèle bayésien complet avec échantillonnage de Thompson. On voulait savoir si notre
modèle physique simple (quadratique en $z$) était battu par ces méthodes.

**« Alors, l'IA fait-elle mieux ? »**
Ça dépend de la taille. Sur le petit problème V2 (1 024 profils), **non** : le modèle simple est aussi bon ou
meilleur, car les modèles appris ont trop de paramètres pour le peu de données. Sur le problème CST à
30 bits, **oui** : FMQA et BOCS atteignent un meilleur hypervolume plus vite que le modèle V2, NSGA-II et le
hasard.

**« Pourquoi le réseau de neurones, plus précis, est-il moins bon ? »**
Parce que la boucle utilise une **somme pondérée**, qui ne peut pas désigner les profils dans les parties
non convexes du front (non supportés). Un modèle précis ne les vise donc jamais ; un modèle grossier les
atteint par ses erreurs. On l'a vérifié profil par profil (`diagnostic_v2_ia.py`).

**« Pourquoi la contrainte en boîte noire échoue-t-elle ? »**
Le front est **au bord** de la contrainte $C_{L,\max} \ge 1{,}2$. Donner une mauvaise note aux infaisables
apprend au modèle à **fuir** cette zone, alors que c'est là qu'il faut chercher. Un modèle explicite de la
contrainte, même grossier (affine), permet de s'en approcher.

**« Le problème CST, c'est encore un jouet ? »**
Non : 10⁹ profils, on ne peut plus énumérer, on ne connaît pas le front exact, et chaque évaluation est une
vraie simulation XFOIL + SER (16 244 au total, mises en cache). C'est la situation d'un vrai bureau d'études,
et c'est là que la question « solveur quantique ou non » commence à se poser (QUBO de 33 qubits).

**« Comment savez-vous que la nouvelle boucle ne change rien d'autre ? »**
`verifier_boucle()` : avec le modèle V2, la boucle générique simule **exactement** les mêmes profils, dans le
même ordre, que `optim/qubo_loop.py`, sur 3 graines.

---

## 7. Commandes

Depuis le dossier `Code/` (sous Windows, `python` au lieu de `python3`) :

```bash
python3 -c "from ia_qubo.boucle import verifier_boucle; verifier_boucle()"   # garde-fou, quelques secondes
python3 -m ia_qubo.experiences_v2_ia          # ~25 min sur 11 processus    -> resultats/ia_qubo_V2.json
python3 -m ia_qubo.experiences_v2_ia rapide   # 3 graines                   -> resultats/ia_qubo_V2_rapide.json
python3 -m ia_qubo.diagnostic_v2_ia           # ~10 min                     -> resultats/ia_qubo_diagnostic_V2.json
python3 -m ia_qubo.precision_modeles          # ~15 min                     -> resultats/ia_qubo_precision.json
python3 -m ia_qubo.qaoa_contraintes           # ~3 min                      -> resultats/ia_qubo_qaoa_contraintes.json
python3 -m ia_qubo.experiences_echelle        # ~1 h 45 à cache vide (XFOIL) -> resultats/ia_qubo_echelle.json
```

> **Attention sous Windows** : `probleme_cst.py` importe `fcntl` (verrou de fichier pour le cache), qui
> n'existe que sous Linux/macOS. Tout ce qui importe `probleme_cst` (`experiences_echelle.py`, la partie
> CST de `precision_modeles.py`) échouera sous Windows tant que ce verrou n'est pas remplacé. Les
> expériences sur le problème V2 ne sont pas concernées.
