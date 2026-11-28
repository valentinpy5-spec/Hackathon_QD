# Le dossier `quantique/` expliqué

> **Dernière mise à jour : 28/11/2026 à 16h09**, d'après le code du commit `a9861c4` (branche
> `branch_meddi`, avec la modification locale de `qaoa.py` : le commentaire PennyLane). Si un fichier
> de `quantique/` ou `optim/qubo_loop.py` a été modifié après cette date (`git log -- Code/quantique`),
> ce README n'est peut-être plus à jour.

Ce document explique **ce que fait chaque fichier du dossier `quantique/`**, **comment il s'insère
dans le reste du code** (surtout `optim/qubo_loop.py`) et **juste assez de théorie** pour comprendre
le code et répondre au jury. Les chiffres cités sortent des fichiers de `resultats/` (on dit lequel à
chaque fois), ils ne sont pas inventés.

> **En une phrase.** On transforme le problème « trouver le meilleur profil d'aile » en un
> **QUBO** (une fonction quadratique de bits 0/1), puis on cherche son minimum avec trois solveurs
> interchangeables : l'**énumération exacte**, le **recuit simulé** (classique) et **QAOA**
> (algorithme quantique, simulé exactement sur ordinateur classique).

---

## Sommaire

1. [Où se place la partie quantique dans la chaîne](#1-où-se-place-la-partie-quantique-dans-la-chaîne)
2. [La théorie minimale](#2-la-théorie-minimale)
3. [Les fichiers un par un](#3-les-fichiers-un-par-un)
4. [Qui utilise quoi : interactions avec le reste du code](#4-qui-utilise-quoi--interactions-avec-le-reste-du-code)
5. [Les résultats à connaître](#5-les-résultats-à-connaître)
6. [Questions probables du jury](#6-questions-probables-du-jury)
7. [Commandes](#7-commandes)

---

## 1. Où se place la partie quantique dans la chaîne

Le projet optimise un profil d'aile selon **deux objectifs en conflit** : la traînée $C_D$
(aérodynamique, calculée par XFOIL) et la **SER** (surface équivalente radar, furtivité). Chaque
évaluation d'un profil est une **simulation coûteuse** (~0,34 s par profil avec XFOIL en V2, bien plus
avec un vrai solveur CFD/électromagnétique). Le but est donc de **trouver le front de Pareto en simulant
le moins de profils possible**.

Un profil est décrit par des variables discrètes (des « niveaux » sur une grille), donc par des bits :

| Itération | Variables | Niveaux | Bits | Nombre de profils |
|---|---|---|---|---|
| V0/V1 | cambrure $m$, position $p$, épaisseur $t$ | 8 × 8 × 8 | 3 + 3 + 3 = **9** | 512 |
| V2 | $m$, $p$, $t$, rayon de bord d'attaque $I$ | 8 × 4 × 8 × 4 | 3 + 2 + 3 + 2 = **10** | 1 024 |

La boucle d'optimisation (`optim/qubo_loop.py`) fonctionne ainsi :

```
                ┌─────────────────────────────────────────────────────────────┐
                │                 optim/qubo_loop.py : qubo_loop()            │
                │                                                             │
 profils déjà   │  1. ajuster des modèles quadratiques     (fit_quadratic,    │
 simulés  ────► │     de C_D et SER, affine de C_L          fit_linear)       │
 (XFOIL + SER)  │                │                                            │
                │                ▼                                            │
                │  2. pour chaque poids w : construire un QUBO   (build_qubo) │
                │     w·C_D + (1-w)·SER + pénalités de contraintes            │
                │        └── utilise quantique/qubo.py : surrogate_to_qubo    │
                │                │                                            │
                │                ▼                                            │
                │  3. résoudre le QUBO              (ranked_candidates)       │
                │       solver="exact" → quantique/qubo.py   (énumération)    │
                │       solver="sa"    → quantique/recuit.py (recuit simulé)  │
                │       solver="qaoa"  → quantique/qaoa.py   (QAOA simulé)    │
                │                │                                            │
                │                ▼                                            │
                │  4. simuler (XFOIL) les meilleurs candidats non vus  ───────┼──► on recommence
                └─────────────────────────────────────────────────────────────┘
```

**Point clé à retenir** : le quantique n'intervient **qu'à l'étape 3**, comme *solveur d'un problème
d'optimisation combinatoire*. Ce n'est pas lui qui calcule l'aérodynamique ni la SER. Le QUBO est un
**modèle de substitution** (approximation) du vrai problème : c'est pour ça qu'on boucle et qu'on
re-simule les candidats proposés.

---

## 2. La théorie minimale

### 2.1 Le QUBO

**QUBO** = *Quadratic Unconstrained Binary Optimization*. On cherche le vecteur de bits
$b \in \{0,1\}^n$ qui minimise

$$E(b) = b^\top Q\, b + c = \sum_i Q_{ii}\, b_i + \sum_{i<j} Q_{ij}\, b_i b_j + c$$

- Comme $b_i^2 = b_i$, les termes linéaires se rangent **sur la diagonale** de $Q$.
- Dans le code, $Q$ est **triangulaire supérieure** (chaque paire $i<j$ n'apparaît qu'une fois).
- « Unconstrained » : il n'y a **pas de contrainte** ; les contraintes doivent être transformées en
  **pénalités** ajoutées à l'énergie (voir 2.5).

Pourquoi ce format ? Parce que c'est **exactement** le format qu'acceptent les machines de recuit
quantique (D-Wave) et l'algorithme QAOA.

### 2.2 Du modèle de substitution au QUBO (l'encodage binaire)

On ajuste un **polynôme de degré 2** sur les variables normalisées $z_i \in [-1, 1]$ :

$$f(z) = c_0 + \sum_i g_i z_i + \sum_{i \le j} M_{ij}\, z_i z_j \qquad (10 \text{ coefficients pour } d=3,\ 15 \text{ pour } d=4)$$

Chaque variable est un entier $x_i \in \{0, \dots, L-1\}$ écrit en binaire
($x_i = \sum_j 2^j b_{ij}$, **bit de poids faible en premier**), et $z_i = 2x_i/(L-1) - 1$.
Donc $z$ est une **fonction affine des bits** : $z = a_0 + A\,b$.

En substituant dans $f$, un polynôme de degré 2 en $z$ reste un polynôme de degré 2 en $b$ :
c'est un QUBO. **C'est tout le travail de `surrogate_to_qubo`.**

> Pourquoi un modèle quadratique ? Parce qu'un QUBO ne sait représenter **que** des interactions
> à deux bits. Un modèle de degré 3 donnerait des termes à 3 bits, qu'il faudrait réduire avec des
> qubits auxiliaires.

### 2.3 Le modèle d'Ising (lien avec la physique)

Un qubit mesuré vaut 0 ou 1 ; l'opérateur $Z$ de Pauli a pour valeurs propres $+1$ (pour $|0\rangle$)
et $-1$ (pour $|1\rangle$). Avec le changement de variable $b_i = (1 - s_i)/2$, $s_i \in \{-1, +1\}$,
le QUBO devient un **hamiltonien d'Ising** :

$$E = \sum_i h_i\, s_i + \sum_{i<j} J_{ij}\, s_i s_j + \text{offset}$$

- $h_i$ = « champ local » sur le qubit $i$ → porte **RZ** sur un circuit ;
- $J_{ij}$ = « couplage » entre qubits $i$ et $j$ → porte **ZZ** (= CNOT – RZ – CNOT).

L'énergie devient un opérateur $H_C$ **diagonal** : $H_C|x\rangle = E(x)|x\rangle$.
Trouver le minimum du QUBO = trouver **l'état fondamental** de $H_C$.

### 2.4 Les trois solveurs

| Solveur | Idée | Coût | Garantie |
|---|---|---|---|
| **Énumération** | calculer $E$ pour les $2^n$ chaînes et trier | $2^n$ (16 384 pour 14 qubits : ~4 ms) | exact |
| **Recuit simulé** | marche aléatoire qui retourne un bit à la fois, accepte parfois les mauvais mouvements (Metropolis), température qui baisse | quelques centaines de balayages | heuristique |
| **QAOA** | circuit quantique paramétré dont la mesure donne souvent des chaînes de basse énergie | $p$ couches de portes + optimisation classique des angles | heuristique ; exact si $p \to \infty$ |

**Recuit simulé.** À température $T$, on propose de retourner le bit $i$ ; la variation d'énergie est
$\Delta E = (1 - 2b_i)\,(Q_{ii} + \sum_{j} S_{ij} b_j)$ ($S$ = couplages symétrisés). On accepte si
$\Delta E \le 0$, sinon avec probabilité $e^{-\Delta E/T}$. On démarre chaud (on explore) et on
refroidit (on se fige dans un minimum). C'est l'**analogue classique du recuit quantique** : le recuit
quantique sort des minima locaux par effet tunnel au lieu de l'agitation thermique.

**QAOA** (*Quantum Approximate Optimization Algorithm*, Farhi et al. 2014). Sur $n$ qubits :

1. On part de $|+\rangle^{\otimes n}$ : **superposition uniforme** des $2^n$ solutions (portes de Hadamard).
2. On applique $p$ fois deux opérateurs :
   - **couche de coût** $U_C(\gamma) = e^{-i\gamma H_C}$ : comme $H_C$ est diagonal, chaque amplitude
     $|x\rangle$ est simplement multipliée par la phase $e^{-i\gamma E(x)}$ ;
   - **couche de mélange** $U_M(\beta) = e^{-i\beta \sum_k X_k}$ : une rotation $R_X(2\beta)$ sur chaque
     qubit, qui fait **interférer** les solutions qui diffèrent d'un bit.

   $$|\gamma, \beta\rangle = U_M(\beta_p)U_C(\gamma_p)\cdots U_M(\beta_1)U_C(\gamma_1)\,|+\rangle^{\otimes n}$$

3. Un **optimiseur classique** (ici COBYLA) règle les $2p$ angles $(\gamma, \beta)$ pour minimiser
   l'énergie moyenne $\langle \gamma,\beta | H_C | \gamma,\beta\rangle$. C'est un algorithme
   **variationnel hybride** : le quantique prépare l'état, le classique ajuste les paramètres.
4. On **mesure** : on obtient la chaîne $x$ avec la probabilité $|\langle x|\gamma,\beta\rangle|^2$
   (règle de Born). Les bonnes solutions sont amplifiées, donc on les tire plus souvent qu'au hasard.

**Intuition « adiabatique ».** L'état de départ $|+\rangle^n$ est l'état fondamental de
$-\sum_k X_k$. Si on passe *très lentement* de $-\sum X$ à $H_C$, on reste dans l'état fondamental
(théorème adiabatique) et on finit sur la solution optimale. QAOA est une version **discrétisée** de ce
passage en $p$ pas : plus $p$ est grand, meilleur c'est (en théorie). La vitesse permise dépend du
**gap** (écart entre les deux plus basses énergies) le long du chemin : petit gap = problème dur.

### 2.5 Mettre des contraintes dans un QUBO (pénalités)

Le problème V2 a deux contraintes. Elles sont codées dans `optim/qubo_loop.py`, pas dans `quantique/`,
mais elles changent beaucoup le comportement de QAOA :

- **Épaisseur $t \ge 8\,\%$** = interdire l'indice 0 de $t$ (3 bits à 0). La pénalité exacte
  $P(1-b_0)(1-b_1)(1-b_2)$ est **cubique** ; on la rend quadratique avec **1 qubit auxiliaire** $y$
  (**réduction de Rosenberg** : on force $y = (1-b_1)(1-b_2)$ par une pénalité).
- **Portance max $C_{L,\max} \ge 1{,}2$** : on écrit $C_L$ comme un modèle **affine** en bits $L(b)$, on
  ajoute une **variable d'écart** $s \ge 0$ codée sur **3 qubits** et on pénalise $P\,(L(b) - 1{,}2 - s)^2$.

Total V2 : **10 bits de conception + 1 auxiliaire + 3 d'écart = 14 qubits** (V1 : 9 + 1 + 3 = 13).

**Le prix à payer** : pour que la contrainte soit respectée, la pénalité doit être **forte**, donc elle
**écrase** l'objectif dans l'échelle des énergies. C'est la raison principale pour laquelle QAOA
aide peu sur notre problème (voir §5.4).

---

## 3. Les fichiers un par un

### `__init__.py`
Juste la docstring du paquet. Permet `from quantique import qaoa`.

### `qubo.py` — construire et évaluer les QUBO *(utilisé partout)*

| Fonction | Rôle |
|---|---|
| `affine_bit_maps(bits_per_var, n_levels)` | renvoie $(a_0, A)$ tels que $z = a_0 + A\,b$ (l'encodage binaire de §2.2) |
| `quadratic_features(z)` | les monômes $[1, z_i, z_i z_j\ (i\le j)]$ : 10 colonnes pour $d=3$, 15 pour $d=4$. Sert à ajuster le modèle par moindres carrés |
| `surrogate_to_qubo(coef, bits, levels)` | **cœur du fichier** : coefficients du polynôme en $z$ → $(Q, c)$. On développe $f(a_0 + Ab)$, on obtient $b^\top (A^\top M A) b$ + termes linéaires, puis on replie les linéaires sur la diagonale et on met les couplages en triangulaire supérieure ($2P_{ij}$ car $P$ est symétrique) |
| `qubo_energy(Q, c, B)` | énergie de une ou plusieurs chaînes (lignes de `B`), vectorisée avec `einsum` |
| `all_bitstrings(n)` | matrice $(2^n, n)$ de toutes les chaînes ; ligne $x$ = bits de l'entier $x$, **bit $k$ = `(x >> k) & 1`**. C'est cette convention qui relie l'indice d'une amplitude QAOA à une chaîne de bits |
| `brute_force(Q, c)` | énumération exacte : toutes les chaînes + leurs énergies |
| `qubo_to_ising(Q, c)` | QUBO → $(h, J, \text{offset})$ avec $b = (1-s)/2$ |
| `ising_energy(h, J, offset, S)` | énergie d'Ising, pour vérifier que les deux formulations coïncident |

### `recuit.py` — recuit simulé

`simulated_annealing(Q, c, n_sweeps, T0, T1, n_reads, seed, return_states)` :
- températures géométriques de $T_0 = 2\max|Q|$ à $T_1 = 0{,}01\max|Q|$ (échelle adaptée au QUBO) ;
- un **balayage** = proposer de retourner chaque bit une fois (ordre aléatoire) ;
- `n_reads` redémarrages indépendants depuis des chaînes aléatoires (comme les « lectures » d'une
  machine D-Wave) ;
- `return_states=True` renvoie **tous** les états finaux, utile dans la boucle pour proposer
  plusieurs candidats. C'est ce que fait `neal.SimulatedAnnealingSampler` du SDK D-Wave Ocean.

### `qaoa.py` — QAOA simulé exactement *(le fichier quantique principal)*

On **ne fait pas tourner de vrai ordinateur quantique** : on simule le **vecteur d'état** exact
($2^n$ amplitudes complexes, 16 octets chacune → 256 Ko pour 14 qubits) avec NumPy.

| Fonction | Rôle | Équivalent circuit |
|---|---|---|
| `qaoa_state(gammas, betas, E, n)` | construit $\lvert\gamma,\beta\rangle$ : part de $2^{-n/2}$ partout, puis pour chaque couche `psi *= exp(-1j*g*E)` et `apply_mixer` | Hadamard, puis $p$ × (RZ/ZZ, RX) |
| `apply_mixer(psi, beta, n)` | $e^{-i\beta\sum X_k}$ : on voit `psi` comme un tenseur $(2,2,\dots,2)$ ; pour chaque qubit, $\cos\beta\cdot\psi - i\sin\beta\cdot X\psi$, où $X\psi$ = `np.flip` sur cet axe (échange 0 ↔ 1) | $R_X(2\beta)$ sur chaque qubit |
| `expectation(params, E, n)` | $\langle\psi|H_C|\psi\rangle = \sum_x |\psi_x|^2 E(x)$ : la fonction que l'optimiseur minimise | moyenne des mesures |
| `normalize(E)` | centre-réduit les énergies (voir encadré) | — |
| `optimize_qaoa(E, n, p, n_restarts, seed, init)` | boucle variationnelle : COBYLA depuis `n_restarts` angles aléatoires ($\gamma\in[0,\pi]$, $\beta\in[0,\pi/2]$) + éventuellement `init` ; garde le meilleur | optimiseur classique |
| `interp_init(params, p_new)` | angles optimaux à $p$ → point de départ à $p+1$ par interpolation linéaire (stratégie **INTERP**, Zhou et al. 2020) | — |
| `sample(psi, shots, seed)` | tire `shots` indices $x$ avec probas $|\psi_x|^2$ | mesures (règle de Born) |

**Pourquoi la simulation est si simple** : la couche de coût est **diagonale**, donc au lieu
d'appliquer des centaines de portes RZ/ZZ, on fait **une seule multiplication élément par élément**.
Le gros commentaire au milieu de `qaoa.py` montre **le même circuit écrit en PennyLane** (portes
Hadamard, MultiRZ, RX) : les états sont identiques à $10^{-16}$ près, mais PennyLane est ~10× plus
lent (~40 ms contre ~4 ms par circuit à 14 qubits, à cause du surcoût Python par porte), ce qui ferait
passer un run de la boucle de ~12 min à ~2 h. Ce bloc est là **pour montrer au jury que notre code
correspond bien à un vrai circuit quantique**.

> **Pourquoi `normalize` est indispensable.** $\gamma$ multiplie l'énergie dans une phase
> $e^{-i\gamma E}$. Si $E \approx 100$ (traînée en « counts »), un tout petit changement de $\gamma$
> fait tourner les phases de plusieurs tours : le paysage d'optimisation devient chaotique.
> Mesuré dans `experiences.py` (9 qubits, $p=1$) : probabilité de l'optimum **0,018 avec
> normalisation, 0,0045 sans** (le hasard donne 1/512 ≈ 0,002).

### `experiences.py` — expériences de référence du cours (V0)

`python -m quantique.experiences` → `resultats/experiences_quantique.json`. Trois parties :

- **A. QUBO jouet à 3 qubits** « pas deux voisins actifs » : $E = -b_0 - b_1 - b_2 + 2b_0b_1 + 2b_1b_2$,
  minimum $-2$ pour la chaîne $101$. On calcule l'Ising, le **spectre adiabatique**
  $H(s) = -(1-s)\sum X + s\,H_C$ (gap minimal 0,62 en $s \approx 0{,}67$), le paysage $p = 1$ et la
  convergence avec $p$. Sert de **démonstration pédagogique** (on peut tout vérifier à la main).
- **B. QUBO aéro à 9 qubits** (modèle quadratique de $C_D$ ajusté sur les 512 profils V0) : QAOA
  $p = 1 \dots 5$ et recuit simulé. Montre aussi la **limite du modèle de substitution** (voir §5.1).
- **C. La boucle itérative** « modèle → QUBO → simuler les candidats » sur $C_D$ seul, comparée à une
  recherche aléatoire. Le solveur y est l'énumération exacte (c'est la boucle qu'on généralise ensuite
  dans `optim/qubo_loop.py`).

Contient aussi `sum_x(n)` : la matrice $2^n \times 2^n$ de $\sum_k X_k$, utilisée seulement pour le spectre.

### `montee_en_taille.py` — que se passe-t-il quand le QUBO grandit ? (V2)

`python -m quantique.montee_en_taille` → `resultats/montee_en_taille.json`.

On garde **le même problème physique** (modèle quadratique de $0{,}5\,C_D + 0{,}5\,\text{SER}$, scénario
S1) et on **affine la grille** : $b$ bits par variable, $n = 3b$ qubits de **6 à 18**. Pour chaque $n$ :
énergies exactes, QAOA $p = 1..4$, recuit simulé, et le **coût du circuit** (couplages non nuls,
CNOT par couche $= 2 \times$ couplages, car chaque porte ZZ coûte 2 CNOT).

Les angles sont initialisés intelligemment (amélioration V2) : **INTERP** depuis $p-1$ et **transfert**
depuis la taille précédente (comme on ferait sur une vraie machine, où on ne peut pas optimiser
longtemps). La première version (angles aléatoires) plafonnait dès $p = 3$ ; ses résultats sont gardés
dans `montee_en_taille_init_aleatoire.json`.

Indicateurs calculés par `qaoa_stats` :
- `p_opt` : probabilité de mesurer l'optimum ; à comparer au hasard $1/2^n$ ;
- `p_top1` : probabilité de tomber dans le **1 % meilleur** ;
- `ratio` : rapport d'approximation $r = (E_{\max} - \langle E\rangle)/(E_{\max} - E_{\min})$ (1 = parfait) ;
- `shots95` : nombre de tirs pour voir l'optimum avec 95 % de chances, $\ln 0{,}05 / \ln(1 - p_{opt})$.

### `anatomie_qubo.py` — pourquoi QAOA n'apporte rien dans la boucle (V3)

`python -m quantique.anatomie_qubo` → `resultats/anatomie_qubo_V3.json` (~10 min).

C'est le **diagnostic** : on reconstruit les **vrais QUBO que la boucle V2 résout** (même code :
`fit_quadratic`, `fit_linear`, `build_qubo`, contraintes `CONS` de `optim/experiences_v2.py`) dans deux
états : **début** de boucle (modèles ajustés sur 30 profils, 10 graines) et **fin** (modèles ajustés sur
les 1 024 profils). On mesure :

- `obj_share` : **part de l'objectif dans l'étendue des énergies** (le reste = pénalités) ;
- `gap` : écart d'énergie entre les deux meilleurs profils, rapporté à l'étendue ;
- `same_as_substitute` : le minimum du QUBO est-il le vrai minimum du problème de substitution
  (contraintes imposées en dur) ? Si non : la pénalité est trop douce ou la variable d'écart discrète
  biaise le résultat ;
- `t_enum` : temps d'énumération des $2^{14}$ états ;
- QAOA $p = 1, 2, 3$ sur 6 QUBO, **avec** contraintes (14 qubits) et **sans** (10 qubits) pour isoler
  l'effet des pénalités.

---

## 4. Qui utilise quoi : interactions avec le reste du code

### Graphe des imports

```
quantique/qubo.py ◄──────────┬──── optim/qubo_loop.py   (quadratic_features, surrogate_to_qubo,
       ▲                     │                            all_bitstrings, qubo_energy, affine_bit_maps)
       │                     ├──── optim/bayes.py       (quadratic_features)
       │                     ├──── ia_qubo/modeles.py   (surrogate_to_qubo)
       │                     ├──── ia_qubo/boucle.py    (all_bitstrings, qubo_energy)
       │                     └──── ia_qubo/qaoa_contraintes.py
quantique/recuit.py ◄─────────────  optim/qubo_loop.py   (simulated_annealing)
quantique/qaoa.py   ◄──────┬──────  optim/qubo_loop.py   (normalize, optimize_qaoa, qaoa_state, sample)
                           └──────  ia_qubo/qaoa_contraintes.py

quantique/experiences.py     ──► utilise qubo, qaoa, recuit           ; lit resultats/base_aero_Re3e+06_a2.csv
quantique/montee_en_taille.py ─► utilise qubo, qaoa, recuit + optim.qubo_loop.fit_quadratic
                                                                       ; lit resultats/base_complete_S1_note.csv
quantique/anatomie_qubo.py   ──► utilise qubo, qaoa + optim.problem_v2, optim.qubo_loop, optim.experiences_v2
```

Les trois modules `qubo.py`, `recuit.py`, `qaoa.py` sont des **briques** (aucune dépendance au reste du
projet, seulement NumPy/SciPy). Les trois scripts `experiences.py`, `montee_en_taille.py`,
`anatomie_qubo.py` sont des **expériences** qui produisent des JSON lus par les cours et la fiche projet.

### Le chemin complet d'un solveur QAOA dans la boucle

Dans `optim/qubo_loop.py`, fonction `ranked_candidates(Q, c, solver="qaoa", ...)` :

```python
B  = all_bitstrings(n)               # qubo.py     : les 2^n chaînes (n = 14 en V2)
E  = qubo_energy(Q, c, B)            # qubo.py     : leurs énergies = la diagonale de H_C
En = qa.normalize(E)                 # qaoa.py     : centrer-réduire (sinon phases illisibles)
r  = qa.optimize_qaoa(En, n, p=2, n_restarts=3)    # boucle variationnelle COBYLA
psi = qa.qaoa_state(r.x[:2], r.x[2:], En, n)       # état final
xs = qa.sample(psi, shots=512)       # 512 mesures simulées
uniq = unique(xs) trié par E          # on recalcule l'énergie classiquement et on trie
→ decode() chaque chaîne en indices (m, p, t, I)    # on jette les 4 bits auxiliaires/d'écart
```

Puis `qubo_loop` garde, **pour chaque poids** $w \in \{0; 0{,}25; 0{,}5; 0{,}75; 1\}$, le meilleur profil
pas encore simulé, le simule, met à jour les modèles et recommence. Les poids $w$ balaient le front de
Pareto (somme pondérée $w\,C_D + (1-w)\,\text{SER}$, chaque objectif normalisé par son étendue).

**Comparaison des solveurs dans la boucle** : `optim/experiences_v2.py` lance la boucle avec
`solver="exact"`, `"sa"` et `"qaoa"` (et NSGA-II, et le hasard) → `resultats/pipeline_V2.json`.

**Suite V3 (`ia_qubo/`)** : on remplace le modèle quadratique en $z$ par des modèles appris (FMQA,
BOCS, réseau de neurones) mais on réutilise `surrogate_to_qubo`, `qubo_energy` et QAOA. Voir
`ia_qubo/README.md`.

---

## 5. Les résultats à connaître

### 5.1 Le modèle de substitution seul ne suffit pas → il faut boucler
*(`experiences_quantique.json`, partie B et C, grille V0 à 512 profils)*

- Le minimum du QUBO ajusté sur **tous** les profils donne un profil dont le **vrai** $C_D$ vaut 61,1
  counts, **37ᵉ** meilleur seulement ; le vrai minimum est 50,5. Un modèle quadratique global est une
  approximation grossière.
- En **boucle** (ajuster sur peu de points, proposer, simuler, réajuster), on trouve le vrai optimum en
  **25 simulations** (médiane sur 50 graines, quartiles 22–34), contre **256** en moyenne au hasard.

### 5.2 QAOA fonctionne, et s'améliore avec $p$
*(`experiences_quantique.json`, QUBO aéro 9 qubits ; hasard = 1/512 ≈ 0,002)*

| $p$ | P(optimum) | P(5 meilleurs) | Rapport d'approx. | Tirs moyens pour voir l'optimum |
|---|---|---|---|---|
| 1 | 0,018 | 0,09 | 0,87 | 56 |
| 2 | 0,062 | 0,22 | 0,93 | 16 |
| 3 | 0,097 | 0,30 | 0,95 | 10 |
| 5 | 0,142 | 0,44 | 0,97 | 7 |

Recuit simulé sur le même QUBO : 51 % des lectures trouvent l'optimum exact avec 20 balayages.
Jouet 3 qubits : P(optimum) = 0,31 ($p=1$), 0,54 ($p=2$), 0,67 ($p=3$).

### 5.3 Montée en taille : QAOA reste meilleur que le hasard mais l'optimum se dilue
*(`montee_en_taille.json`, $p = 4$)*

| $n$ qubits | Couplages | P(opt) QAOA | P(opt) hasard | Gain | P(1 % meilleur) | Recuit (succès/lecture) |
|---|---|---|---|---|---|---|
| 6 | 15 | 0,17 | 0,016 | ×11 | 0,17 | 0,56 |
| 9 | 36 | 0,078 | 0,002 | ×40 | 0,32 | 0,58 |
| 12 | 66 | 0,020 | 2,4·10⁻⁴ | ×84 | 0,51 | 0,20 |
| 15 | 105 | 0,0042 | 3,1·10⁻⁵ | ×140 | 0,58 | 0,08 |
| 18 | 153 | 0,0007 | 3,8·10⁻⁶ | ×180 | 0,59 | 0,00 |

Lecture : le **gain sur le hasard augmente** avec $n$ et QAOA concentre ~60 % de sa probabilité sur le
1 % meilleur, mais la probabilité de l'optimum **exact** chute. Le graphe d'interaction est **complet**
($n(n-1)/2$ couplages) : 306 CNOT par couche à 18 qubits, ce qui est lourd pour du matériel bruité actuel.

### 5.4 Dans la vraie boucle V2, QAOA n'apporte rien — et on sait pourquoi
*(`anatomie_qubo_V3.json`, QUBO à 14 qubits)*

- **L'objectif ne pèse que 1 à 2 % de l'étendue des énergies** : le reste vient des pénalités de
  contraintes. Le paysage que voit QAOA est **dominé par les contraintes**.
- **L'écart entre les deux meilleurs profils** vaut ~$10^{-4}$ de l'étendue : quasi-dégénérescence,
  donc très dur à distinguer pour une petite profondeur $p$.
- Résultat : sur le QUBO **complet**, QAOA ($p = 1..3$) trouve l'optimum seulement **1 à 5 fois plus
  souvent que le hasard**. Sur le **même QUBO sans contraintes** (10 qubits), le gain monte à
  **×30 à ×400**. Les pénalités sont donc la cause principale.
- Le minimum du QUBO n'est même pas toujours le minimum du problème de substitution (20 à 80 % selon
  le cas) : la pénalité de $C_L$ et la variable d'écart discrète introduisent un biais.
- **Énumérer les $2^{14}$ états prend ~4 ms**, contre **~0,34 s pour UNE simulation XFOIL** : sur un
  problème de cette taille, le solveur n'est **pas** le goulot d'étranglement ; c'est le nombre de
  simulations.

Dans la boucle complète (`pipeline_V2.json`, nombre de simulations pour atteindre 99 % de
l'hypervolume du front exact, médiane) :

| Méthode | Scénario S1 | Scénario S2 |
|---|---|---|
| Hasard | 632 | 740 |
| Boucle QUBO + exact | 62,5 | 75 |
| Boucle QUBO + recuit simulé | 62,5 | 90 |
| Boucle QUBO + QAOA (5 graines, budget 150) | 75 | 70 |

**Message à faire passer** : le gain vient de la **boucle à modèle de substitution** (÷10 simulations
par rapport au hasard), quel que soit le solveur. QAOA, quand il est utilisé comme solveur dans cette
boucle, fait **aussi bien** que l'exact, parce que la boucle tolère des candidats « presque optimaux » ;
mais il n'apporte pas d'avantage à cette taille.

V3 (`ia_qubo/qaoa_contraintes.py`) : traiter la contrainte en **boîte noire** (au lieu de variables
d'écart) passe à 11 qubits et fait monter la part de l'objectif à ~5–10 %, mais le gain de QAOA reste
du même ordre (quelques unités).

---

## 6. Questions probables du jury

**« Vous avez utilisé un vrai ordinateur quantique ? »**
Non. On simule **exactement** le vecteur d'état sur ordinateur classique (NumPy). C'est possible
jusqu'à ~25–30 qubits ; nous utilisons au plus 18. L'avantage : résultats **sans bruit**, on mesure ce
que l'algorithme donne idéalement, et on obtient toutes les probabilités. Le commentaire PennyLane de
`qaoa.py` montre le circuit équivalent porte par porte (Hadamard, RZ/ZZ, RX), qu'on pourrait envoyer
sur une machine réelle.

**« Si vous calculez déjà les $2^n$ énergies pour simuler QAOA, à quoi sert QAOA ? »**
En simulation, c'est inévitable : simuler un état quantique coûte $2^n$. Sur une vraie machine, on
n'aurait pas besoin d'énumérer. Notre étude sert à **mesurer ce que QAOA apporterait** (probabilité de
l'optimum, nombre de tirs), et la conclusion honnête est qu'à 14 qubits l'énumération classique
(~4 ms) suffit largement. Dans la boucle, on n'utilise d'ailleurs que les **512 tirs** mesurés, comme
sur une machine.

**« Pourquoi un QUBO ? »**
Parce que c'est le format natif du recuit quantique et de QAOA. Il impose un objectif **quadratique**
en bits, d'où le modèle de substitution de degré 2 et les astuces (Rosenberg, variables d'écart) pour
les contraintes.

**« Comment passez-vous d'un profil d'aile à des qubits ? »**
Chaque paramètre (cambrure, position, épaisseur, rayon de bord d'attaque) prend 4 ou 8 niveaux, codés
sur 2 ou 3 bits. Un qubit = un bit de cette écriture binaire. Les qubits auxiliaires (1) et d'écart (3)
servent uniquement aux contraintes et sont jetés au décodage.

**« Pourquoi QAOA n'est pas meilleur que le recuit simulé ou l'exact ? »**
(1) Problème petit : $2^{14}$ états s'énumèrent en millisecondes. (2) Les pénalités de contraintes
représentent 98 % de l'échelle d'énergie, l'objectif est « écrasé ». (3) Les meilleurs profils sont
quasi dégénérés. (4) À faible profondeur $p$, QAOA ne fait qu'amplifier modestement. On l'a démontré
expérimentalement avec `anatomie_qubo.py` (avec / sans contraintes).

**« Que se passerait-il à plus grande échelle ? »**
`montee_en_taille.py` montre que le **gain sur le hasard croît** avec $n$ (×11 à 6 qubits, ×180 à 18).
C'est là qu'un solveur quantique pourrait devenir intéressant, à condition (a) de réduire le poids des
pénalités, (b) d'avoir assez de profondeur $p$ sur du matériel peu bruité, car le graphe complet coûte
beaucoup de CNOT. La V3 (`ia_qubo/`, problème CST à 30 bits) explore cette direction.

**« Pourquoi normaliser les énergies ? Pourquoi COBYLA ? Pourquoi INTERP ? »**
- Normaliser : $\gamma$ multiplie l'énergie dans une phase ; sans normalisation le paysage est
  chaotique (P(opt) ÷4 mesuré).
- COBYLA : optimiseur sans gradient, robuste, standard pour les petits circuits variationnels (sur
  machine réelle on n'a pas le gradient exact, seulement des mesures bruitées).
- INTERP : partir des angles optimaux de $p-1$ évite de recommencer au hasard ; sans ça les résultats
  plafonnaient dès $p = 3$.

**« Quelle différence entre recuit simulé, recuit quantique et QAOA ? »**
Recuit simulé : classique, sort des minima locaux par **agitation thermique**. Recuit quantique
(D-Wave) : sort des minima par **effet tunnel**, en faisant évoluer lentement un hamiltonien
$-(1-s)\sum X + s\,H_C$. QAOA : version **discrète et paramétrée** de ce même chemin, sur un ordinateur
quantique à portes, avec un optimiseur classique dans la boucle.

---

## 7. Commandes

Depuis le dossier `Code/` (sous Windows, `python` au lieu de `python3`) :

```bash
python3 -m quantique.experiences        # jouet 3 qubits, aéro 9 qubits, boucle V0 -> resultats/experiences_quantique.json
python3 -m quantique.montee_en_taille   # QAOA de 6 à 18 qubits                     -> resultats/montee_en_taille.json
python3 -m quantique.anatomie_qubo      # anatomie des QUBO V2 (~10 min)            -> resultats/anatomie_qubo_V3.json
python3 -m optim.experiences_v2         # boucle complète avec les 3 solveurs (~1 h) -> resultats/pipeline_V2.json
```

Petit test interactif pour voir QAOA marcher sur le jouet à 3 qubits :

```python
import numpy as np
from quantique.qubo import brute_force
from quantique import qaoa as qa

Q = np.array([[-1., 2, 0], [0, -1, 2], [0, 0, -1]])   # « pas deux voisins actifs »
B, E = brute_force(Q, 0.0)                              # 8 chaînes, minimum -2 pour 101 (indice 5)
r = qa.optimize_qaoa(E, n=3, p=2)
psi = qa.qaoa_state(r.x[:2], r.x[2:], E, 3)
print(np.round(np.abs(psi)**2, 3))                      # la probabilité se concentre sur l'indice 5
```
