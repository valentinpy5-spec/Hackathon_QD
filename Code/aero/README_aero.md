# Le dossier `aero/` expliqué

> **Dernière mise à jour : 28/11/2026 à 16h43**, d'après le code du commit `30d54af` (branche
> `branch_meddi`). Si un fichier de `aero/` a été modifié après cette date (`git log -- Code/aero`), ce
> README n'est peut-être plus à jour.

Ce document explique **ce que fait chaque fichier du dossier `aero/`**, **comment il s'insère dans le
reste du code** et **juste assez de théorie aérodynamique** pour comprendre le code et répondre au jury.
Les chiffres cités viennent des fichiers de `resultats/` ou ont été recalculés avec le code du dossier
(on dit d'où à chaque fois).

> **En une phrase.** Ce dossier fabrique la **géométrie** des profils d'aile (NACA 4 chiffres, puis
> NACA modifié en V2) et calcule leur **aérodynamique** : portance $C_L$, traînée $C_D$ (premier objectif
> de l'optimisation), portance maximale $C_{L,\max}$ (contrainte) et l'incidence de vol $\alpha$ (qui
> sert ensuite au calcul de SER). On a deux simulateurs : un **modèle rapide écrit à la main** (V0-V1)
> et **XFOIL** (la référence depuis la V2).

---

## Sommaire

1. [Où se place l'aéro dans la chaîne](#1-où-se-place-laéro-dans-la-chaîne)
2. [La théorie minimale](#2-la-théorie-minimale)
3. [Les fichiers un par un](#3-les-fichiers-un-par-un)
4. [Qui utilise quoi : interactions avec le reste du code](#4-qui-utilise-quoi--interactions-avec-le-reste-du-code)
5. [Les résultats à connaître](#5-les-résultats-à-connaître)
6. [Questions probables du jury](#6-questions-probables-du-jury)
7. [Commandes](#7-commandes)

---

## 1. Où se place l'aéro dans la chaîne

L'aéro est **le premier maillon** : c'est elle qui définit la forme du profil, et la SER comme
l'optimisation en dépendent.

```
 indices de grille (im, ip, it[, iI])                       aero/design_space.py (V1)
        │                                                    optim/problem_v2.py : SpaceV2 (V2)
        ▼
 aero/naca4.py  ──► contour (X, Y) du profil  ─────────────────────────────┐
        │                                                                   │ même géométrie
        ├── V0-V1 : modèle rapide (aero/model.py)                           ▼
        │     panel.py (écoulement non visqueux)                     ser/ (SER du profil)
        │       + boundary_layer.py (couche limite)                         ▲
        │     ──► C_L, C_D, α                                               │
        │                                                                   │ α = attitude
        └── V2 : XFOIL (aero/xfoil_runner.py)                               │     vue par le radar
              croisière à C_L = 0,4        ──► C_D (objectif 1) et α ───────┘
              rampe d'incidence 4°..14°    ──► C_L,max (contrainte ≥ 1,2)
```

**Pourquoi deux simulateurs ?**
- **V0-V1 : modèle rapide maison** (~0,04 s par profil). Il a permis de construire et tester toute la
  chaîne d'optimisation sans dépendre d'un logiciel externe, et de **comprendre la physique** (on maîtrise
  chaque équation).
- **V2 : XFOIL** (Drela, MIT), le logiciel de référence pour les profils 2D. Plus précis (couplage
  visqueux fort, transition $e^N$), et il sait calculer le **décrochage**, donc $C_{L,\max}$, ce que le
  modèle rapide ne sait pas faire.

**Pourquoi c'est important pour le jury** : chaque appel au simulateur est une « simulation coûteuse ».
Tout le projet cherche à trouver le front de Pareto **en appelant le simulateur le moins possible**.
C'est pour ça qu'on compte les simulations (`AeroSimulator`, `CountedProblem`).

---

## 2. La théorie minimale

### 2.1 Le profil NACA 4 chiffres

Un profil = une **ligne moyenne** (la « cambrure », qui courbe le profil) + une **loi d'épaisseur**
répartie de part et d'autre de cette ligne. Trois paramètres, en fractions de corde :

| Paramètre | Sens | Exemple NACA 2412 |
|---|---|---|
| $m$ | cambrure maximale | 0,02 (le « 2 ») |
| $p$ | position de la cambrure max | 0,4 (le « 4 ») |
| $t$ | épaisseur relative max | 0,12 (le « 12 ») |

- La ligne moyenne est faite de **deux paraboles** raccordées en $x = p$.
- L'épaisseur suit une loi polynomiale fixe :
  $y_t = 5t\,(0{,}2969\sqrt{x} - 0{,}1260x - 0{,}3516x^2 + 0{,}2843x^3 - 0{,}1036x^4)$.
  Le terme en $\sqrt{x}$ donne un **nez arrondi** ; le coefficient $-0{,}1036$ ferme le bord de fuite.
- **Rayon de bord d'attaque** : $r_{BA} = 1{,}1019\,t^2$. Dans le NACA classique, il est **imposé par
  l'épaisseur**. C'est la grandeur clé du compromis : un gros nez supporte mieux les fortes incidences
  (portance max), mais renvoie davantage l'onde radar (SER).

Les points sont répartis **en cosinus** : plus serrés au bord d'attaque et au bord de fuite, là où la
courbure et les gradients de vitesse sont les plus forts.

### 2.2 V2 : le NACA 4 chiffres **modifié**

En V1, on ne pouvait pas avoir un nez fin **et** un profil épais : $r_{BA}$ dépendait de $t$. La série
**modifiée** (Abbott & von Doenhoff ; Ladson & Brooks, NASA 1975) ajoute deux paramètres à la loi
d'épaisseur :
- $I$ : **indice de rayon de bord d'attaque**, $r_{BA} = 1{,}1019\,(t\,I/6)^2$. $I = 6$ redonne le NACA
  standard ; $I = 2$ donne un nez 9 fois plus petit, $I = 8$ un nez 1,8 fois plus gros ;
- $T$ : position de l'épaisseur maximale (fixée à 0,3 en V2).

La loi d'épaisseur devient **deux arcs polynomiaux** (avant et arrière de $x = T$), raccordés avec une
épaisseur, une pente nulle et une **courbure continues** en $x = T$ (`_modified_coeffs` résout ces
conditions). C'est cette variable $I$ qui permet de piloter **séparément** la SER (nez) et la traînée
(épaisseur).

### 2.3 Le modèle rapide (V0-V1) : panneaux + couche limite

C'est l'approche classique « **non visqueux + couche limite** », en **couplage faible** (une seule passe).

**Étape 1 : écoulement non visqueux (méthode des panneaux de Hess-Smith, `panel.py`).**
L'air est supposé parfait (sans viscosité) et incompressible. On découpe le contour en $N$ panneaux
droits ; sur chacun on place une **source** d'intensité $\sigma_j$ (propre au panneau) et un **tourbillon**
d'intensité $\gamma$ (le même partout). On écrit :
- $N$ équations de **non-pénétration** : la vitesse normale est nulle au milieu de chaque panneau ;
- 1 **condition de Kutta** : l'écoulement quitte proprement le bord de fuite (vitesses égales en norme
  sur les deux panneaux du bord de fuite). C'est elle qui fixe la circulation, donc la portance.

On résout un système linéaire $(N+1) \times (N+1)$, on obtient la vitesse $V_t$ le long de la paroi, puis
$C_p = 1 - V_t^2$, et en intégrant : $C_L$, $C_M$. La traînée de pression intégrée doit valoir ~0
(paradoxe de d'Alembert) : c'est un **contrôle** de la méthode. Un écoulement non visqueux ne donne
**pas** de traînée : il faut la couche limite.

**Étape 2 : couche limite (`boundary_layer.py`).** Près de la paroi, la viscosité freine l'air sur une
fine couche. On la calcule face par face (extrados et intrados), du **point d'arrêt** au bord de fuite,
avec la vitesse extérieure $U_e$ donnée par les panneaux :

| Zone | Méthode | Ce qu'elle donne |
|---|---|---|
| Laminaire | **Thwaites** : $\theta^2 U_e^6 = \frac{0{,}45}{Re}\int_0^s U_e^5\,ds$ | épaisseur de quantité de mouvement $\theta$, facteur de forme $H$ |
| Transition | critère de **Michel** ($Re_\theta$ dépasse un seuil), ou **décollement laminaire**, ou déclenchement forcé | position $x_{tr}$ du passage laminaire → turbulent |
| Turbulente | méthode d'**entraînement de Head** + frottement de **Ludwig-Tillmann** (2 équations différentielles, RK45) | $\theta$, $H$, $c_f$ ; décollement si $H$ trop grand |

**Étape 3 : traînée (formule de Squire-Young).** La traînée de profil se lit sur l'état de la couche
limite au bord de fuite : $C_D = 2\,\theta_{BF}\,(U_{e,BF})^{(H_{BF}+5)/2}$, sommé sur les deux faces.

**Couplage faible** : la couche limite ne modifie **pas** l'écoulement non visqueux. Conséquence : $C_L$
est celui du fluide parfait (**surestimé** d'environ 18 %, cf. §5) et on ne peut pas prédire le
décrochage.

### 2.4 XFOIL (V2)

XFOIL résout la **même physique**, mais en **couplage visqueux fort** : la couche limite modifie
l'écoulement extérieur et on itère jusqu'à convergence. Il utilise des panneaux à tourbillon linéaire,
une couche limite intégrale à deux équations, une transition **$e^N$** (ici $N_{crit} = 9$, air calme)
et modélise le sillage. Il donne de bons $C_D$ jusqu'à l'approche du décrochage, et une estimation de
$C_{L,\max}$.

On le pilote **en boîte noire** : on lui écrit un fichier de coordonnées, on lui envoie des commandes
texte (`LOAD`, `PANE`, `OPER`, `VISC Re`, `CL 0.4` ou `ALFA a`...) et on relit sa polaire.

### 2.5 Le cas de vol

| Paramètre | V1 (modèle rapide) | V2 (XFOIL) |
|---|---|---|
| Reynolds | $3\cdot10^6$ | $3\cdot10^6$ |
| Point de fonctionnement | incidence imposée $\alpha = 2°$ (option : $C_L$ imposé) | **portance imposée $C_L = 0{,}4$** (croisière : portance = poids) |
| Contraintes | $t \ge 8\,\%$, $C_L \ge 0{,}3$, calcul valide | $t \ge 8\,\%$, **$C_{L,\max} \ge 1{,}2$** (manœuvre), croisière convergée |

**Pourquoi imposer $C_L$ en V2** : en croisière, l'avion doit porter son poids ; c'est donc $C_L$ qui est
fixé, et **l'incidence $\alpha$ devient une sortie** du calcul aéro. Cette incidence est l'**attitude**
sous laquelle le radar voit le profil : c'est le couplage aéro → SER.

**$C_{L,\max}$** : on lance XFOIL sur une rampe $\alpha = 4, 6, 8, 9, \dots, 14°$ et on garde le $C_L$
maximal atteint. Un nez trop fin **décroche tôt** : c'est ce qui empêche la SER d'être minimisée
« gratuitement » et qui crée un vrai compromis.

**Unité de traînée** : on parle en **counts** : 1 count $= 10^{-4}$ de $C_D$ (`CD_counts = CD * 1e4`).

---

## 3. Les fichiers un par un

### `__init__.py`
Expose les fonctions principales du modèle rapide : `naca4`, `naca4_from_code`, `leading_edge_radius`,
`solve`, `evaluate`, `evaluate_at_cl`, `alpha_for_cl`, `AeroResult`.

### `naca4.py` — géométrie *(utilisé par tout le projet)*

| Fonction | Rôle |
|---|---|
| `camber_line(x, m, p)` | ligne moyenne $y_c(x)$ et sa pente (deux paraboles raccordées en $p$) |
| `thickness(x, t, closed_te)` | demi-épaisseur NACA classique, bord de fuite fermé par défaut |
| `naca4(m, p, t, n_half)` | contour complet, $2\,n_{half}+1$ points en cosinus. L'épaisseur est appliquée **perpendiculairement** à la ligne moyenne (angle $\theta = \arctan(dy_c/dx)$) |
| `naca4_from_code("2412")` | raccourci : code à 4 chiffres → `naca4(0.02, 0.4, 0.12)` |
| `leading_edge_radius(t)` | $r_{BA} = 1{,}1019\,t^2$ |
| `_modified_coeffs(I, T)` | **V2** : résout les conditions de raccord des deux arcs de la loi modifiée (systèmes linéaires 2×2 et 3×3) |
| `thickness_modified(x, t, I, T)` | **V2** : demi-épaisseur modifiée, mise à l'échelle de $t$ |
| `naca4_modified(m, p, t, I, T, n_half)` | **V2** : contour du profil modifié, même convention |
| `leading_edge_radius_modified(t, I)` | **V2** : $r_{BA} = 1{,}1019\,(tI/6)^2$ |

**Convention importante** (la même partout, y compris dans `ser/` et XFOIL) : le contour part du
**bord de fuite extrados**, passe par le **bord d'attaque** et revient au **bord de fuite intrados** (sens
trigonométrique). La normale sortante d'un segment de tangente $(t_x, t_y)$ est $(t_y, -t_x)$.

### `panel.py` — méthode des panneaux de Hess-Smith

| Élément | Rôle |
|---|---|
| `Panels` (dataclass) | début, fin, milieu, longueur, tangente et normale de chaque panneau |
| `make_panels(X, Y)` | contour → panneaux |
| `influence_matrices(P)` | vitesse induite au milieu du panneau $i$ par une source unitaire et un tourbillon unitaire sur le panneau $j$ (formules analytiques en log et en angle, dans le repère local du panneau). Auto-influence traitée à part (le point est sur le panneau) |
| `solve(X, Y, alpha_deg, cache)` | assemble et résout le système $(N+1)\times(N+1)$ (non-pénétration + Kutta), renvoie un `InviscidSolution` : $\sigma$, $\gamma$, $V_t$, $C_p$, $C_L$, $C_{D,p}$ (≈ 0, contrôle), $C_M$ au quart de corde, et $C_L$ par Kutta-Joukowski (**deuxième contrôle** : doit égaler $C_L$ intégré). `cache` réutilise la matrice inversée quand seule l'incidence change (utile pour chercher l'$\alpha$ qui donne un $C_L$) |

Adimensionnement : corde $= 1$, $U_\infty = 1$, donc $C_p = 1 - V_t^2$.

### `boundary_layer.py` — couche limite intégrale

| Fonction | Rôle |
|---|---|
| `thwaites_closure(lam)` | corrélations $H(\lambda)$, $\ell(\lambda)$ de Thwaites ($\lambda$ = paramètre de gradient de pression) |
| `michel_transition(Re_x)` | seuil de transition de Michel |
| `H1_of_H`, `H_of_H1` | facteurs de forme de Head (et l'inverse) |
| `cf_ludwieg_tillmann(H, Re_theta)` | frottement turbulent |
| `SurfaceBL` (dataclass) | résultat sur une face : $s$, $x$, $U_e$, $\theta$, $H$, $c_f$, zone turbulente, $x_{tr}$ et son mode, $x_{sep}$, drapeaux d'alerte |
| `march_surface(s, x, Ue, Re, x_trip, x_stop, free_transition)` | **cœur du fichier** : Thwaites, détection de la transition (position **interpolée** entre deux stations, sinon $C_D(m,p,t)$ serait en escalier et le modèle quadratique de la boucle QUBO en souffrirait), puis Head intégré par `solve_ivp`. S'arrête en $x = 0{,}985$ : au bord de fuite fermé, l'écoulement potentiel ralentit artificiellement vers 0, ce qui ferait exploser $\theta$ |
| `squire_young(theta, H, Ue)` | traînée d'une face |
| `split_at_stagnation(sol)` | trouve le **point d'arrêt** (changement de signe de $V_t$) et découpe la surface en extrados / intrados, orientés vers le bord de fuite |

### `model.py` — point d'entrée du modèle rapide

- `AeroResult` (dataclass) : $C_L$, $C_D$, $C_{D,f}$ (part de frottement ; $C_D - C_{D,f}$ = traînée de
  forme), $C_M$, positions de transition et de décollement, $r_{BA}$, `valid`, `flags`.
- `evaluate(m, p, t, alpha, Re, ...)` : enchaîne géométrie → panneaux → point d'arrêt → couche limite
  sur les deux faces → Squire-Young. **Validité** : un décollement turbulent **avant** $x = 0{,}9$ rend le
  calcul invalide (le modèle sort de son domaine) ; après $0{,}9$, il est toléré et signalé
  (`te_separation`).
- `alpha_for_cl(m, p, t, CL_target)` : cherche l'incidence qui donne $C_L$ visé (méthode de Brent sur le
  calcul non visqueux, avec le `cache`).
- `evaluate_at_cl(...)` : `alpha_for_cl` puis `evaluate`.

### `xfoil_runner.py` — pilotage d'XFOIL

| Fonction | Rôle |
|---|---|
| `find_xfoil()` | cherche `xfoil` dans le `PATH`, sinon la copie locale `Code/.tools/xfoil_pkg/` (installée par `installer_outils.sh`) |
| `xfoil_available()` | XFOIL est-il trouvé ? |
| `xfoil_env()` | XFOIL (version Linux) ne converge en visqueux qu'avec un affichage actif : sans écran, on lance un écran virtuel **Xvfb**. Sous **Windows** (`os.name == "nt"`, ajout V3), rien à faire |
| `run_xfoil(m, p, t, alphas, Re, ..., coords, cls)` | écrit le profil dans un dossier temporaire, envoie les commandes XFOIL, relit la polaire. `coords=(X, Y)` passe n'importe quelle géométrie (le NACA modifié en V2, les profils CST en V3) ; `cls=[0.4]` demande des points **à portance imposée**. Un `timeout` coupe XFOIL s'il se bloque, et on garde les points déjà écrits |
| `parse_polar(path)` | lit la polaire : `alpha, CL, CD, CDp, CM, xtr_up, xtr_lo` |

**Attention** : les incidences **non convergées sont absentes** du résultat. Dans `problem_v2.py`, on
teste donc `len(cr["alpha"]) == 1` pour savoir si la croisière a convergé.

### `design_space.py` — grille de conception V1 et simulateur compté

- `DesignSpace` : grille V1 $m \in \{0..7\,\%\}$, $p \in \{0{,}20..0{,}55\}$, $t \in \{6..20\,\%\}$,
  8 niveaux chacun → **3 × 3 bits = 9 bits, 512 profils**. Méthodes `values` (indices → valeurs),
  `encode` / `decode` (indices ↔ bits, **poids faible en premier**, la même convention que
  `quantique/qubo.py`), `all_indices`.
  **Pourquoi une grille discrète** : la formulation QUBO travaille sur des bits. NSGA-II, la boucle QUBO
  et l'énumération exhaustive (front exact) travaillent sur **la même grille**, sinon la comparaison n'a
  pas de sens.
- `FlightCase` : le cas de vol V1 (§2.5).
- `AeroSimulator` : enveloppe du modèle rapide avec **cache** et **compteur** `n_sims` (nombre de profils
  **distincts** simulés), la métrique de comparaison des méthodes. `constraints(r)` renvoie les
  contraintes au format $g \le 0$ (convention pymoo).

En V2, la grille est `SpaceV2` dans `optim/problem_v2.py` (4 variables, 10 bits, 1 024 profils).

### `geometry_export.py` — export vers un solveur SER 3D

- `extrude(X, Y, span, caps)` : extrude le profil selon l'envergure en une maquette « 2,5D » de
  triangles (deux triangles par panneau + les flancs). On réutilise **exactement** les points du calcul
  aéro : l'aéro et l'électromagnétisme voient la même forme.
- `normals`, `write_stl` (STL ASCII, lisible par les solveurs EM et ParaView), `write_metadata` (JSON :
  design, incidence, échelle, repère).
- L'incidence **n'est pas appliquée à la géométrie** : elle est écrite dans les métadonnées, c'est la
  direction radar qu'on tourne. Utilisé par `demo_profil.py` (→ `resultats/NACA2412.stl` et `.json`).
  Prévu comme interface avec le solveur de l'autre groupe, pas utilisé par l'optimisation.

---

## 4. Qui utilise quoi : interactions avec le reste du code

```
aero/naca4.py ◄──────┬── aero/panel.py ◄── aero/model.py ◄── aero/design_space.py (AeroSimulator)
                     │        aero/boundary_layer.py ◄──┘            ▲
                     │                                              ├── optim/problem.py   (V1 : evaluate, evaluate_at_cl,
                     │                                              │                        DesignSpace, FlightCase)
                     │                                              └── generer_base.py    (base exhaustive V0)
                     ├── aero/xfoil_runner.py ◄──┬── optim/problem_v2.py   (V2 : simulate_v2 → base_V2.csv)
                     │                           ├── ia_qubo/probleme_cst.py (V3 : profils CST à 30 bits)
                     │                           ├── validation.py, validation_xfoil_grille.py
                     │                           └── verifier_installation.py (find_xfoil)
                     ├── ser/model.py, ser/compare_po_mom.py   (même géométrie pour la SER)
                     ├── aero/geometry_export.py ◄── demo_profil.py
                     └── ia_qubo/probleme_cst.py
```

- **V1** (`optim/problem.py`) : une « simulation » = `evaluate` (ou `evaluate_at_cl`) du modèle rapide,
  puis la SER à l'incidence trouvée.
- **V2** (`optim/problem_v2.py : simulate_v2`) : géométrie `naca4_modified`, puis **deux appels XFOIL**
  (croisière à $C_L = 0{,}4$ → $C_D$, $\alpha$ ; rampe 4°..14° → $C_{L,\max}$), puis la SER. Les 1 024 profils
  sont calculés **une fois** et mis en cache dans `resultats/base_V2.csv` (~6 min sur 10 cœurs). Les
  méthodes d'optimisation lisent ensuite cette table en **comptant** les profils qu'elles demandent : on
  mesure le coût comme si chaque lecture était une vraie simulation.
- **V3** (`ia_qubo/probleme_cst.py`) : profils **CST** (10 coefficients, 30 bits, ~10⁹ profils), toujours
  évalués par `run_xfoil(coords=...)`.
- La partie quantique ne voit **jamais** l'aéro directement : elle reçoit un modèle quadratique ajusté
  sur les $C_D$ déjà simulés (voir `quantique/README_quantique.md`).

---

## 5. Les résultats à connaître

### 5.1 Le modèle rapide est vérifié (`validation.py`)

`validation.py` contrôle chaque brique : convergence des panneaux en maillage, $C_L$ intégré = $C_L$ de
Kutta-Joukowski, traînée de pression ≈ 0 (d'Alembert), Thwaites contre **Blasius** sur plaque plane,
Head contre la loi turbulente en 1/7, sensibilité aux paramètres numériques, comparaison à XFOIL.
Exemple (recalculé) : NACA 2412 à $\alpha = 2°$, $Re = 3\cdot10^6$ : $C_L = 0{,}50$, $C_D = 69$ counts dont
55 de frottement, transition à 28 % (extrados) et 48 % (intrados) de corde, ~0,04 s.

### 5.2 Modèle rapide contre XFOIL sur les 512 profils V1
*(`resultats/base_aero_Re3e+06_a2.csv` contre `resultats/base_xfoil_a2.csv`, $\alpha = 2°$, 498 profils où
les deux ont convergé)*

| | Modèle rapide / XFOIL (médiane) | Corrélation de rang (Spearman) |
|---|---|---|
| $C_D$ | **+24 %** (le modèle rapide surestime) | **0,91** |
| $C_L$ | **+18 %** (pas de correction visqueuse) | — |

**Lecture** : le modèle rapide se trompe en valeur absolue (couplage faible), mais il classe
correctement la plupart des profils. Il suffisait pour construire la chaîne ; pour des résultats
crédibles et pour la contrainte $C_{L,\max}$, on est passé à XFOIL en V2.

### 5.3 La table XFOIL V2
*(`resultats/base_V2.csv`, 1 024 profils)*

- Croisière convergée pour **920** profils ; **702 faisables**. Les 322 autres ne le sont pas pour
  trois raisons, qui peuvent se cumuler : $t < 8\,\%$ (128), croisière non convergée (104),
  $C_{L,\max} < 1{,}2$ (233).
- $C_D$ en croisière : de **34 à 319 counts** (médiane 61,5).
- **Le nez pilote le décrochage** : part des profils avec $C_{L,\max} \ge 1{,}2$ selon $I$ :

| $I$ (rayon de nez) | 2 (le plus fin) | 4 | 6 (NACA standard) | 8 (le plus rond) |
|---|---|---|---|---|
| $C_{L,\max}$ médian | 1,18 | 1,47 | 1,56 | 1,58 |
| Part avec $C_{L,\max} \ge 1{,}2$ | **46 %** | 84 % | 95 % | 97 % |

C'est **le cœur du compromis** aéro / SER : la SER veut $I$ petit (§2.4 de `ser/README_SER.md`),
la manœuvre veut $I$ grand. Les profils du front de Pareto ont $I = 2$ ou $4$, avec un $C_{L,\max}$ juste
au-dessus de 1,2 en S1.

---

## 6. Questions probables du jury

**« Pourquoi avoir écrit votre propre modèle aéro alors qu'XFOIL existe ? »**
Pour démarrer la chaîne sans dépendance externe, comprendre et maîtriser chaque équation, et pouvoir
tester très vite (~0,04 s par profil). Il a été **vérifié** brique par brique (`validation.py`) et
**comparé** à XFOIL sur 512 profils (classement correct, corrélation de rang 0,91). Pour les résultats
finaux (V2), on utilise XFOIL, qui est la référence pour les profils 2D.

**« Quelles sont les limites du modèle rapide ? »**
Couplage **faible** : la couche limite ne modifie pas l'écoulement extérieur, donc $C_L$ est surestimé
(~18 %) et le **décrochage n'est pas prédit**. Squire-Young et les corrélations (Thwaites, Michel, Head)
sont des modèles intégraux approchés. Traînée surestimée d'environ 24 % par rapport à XFOIL.

**« Et les limites d'XFOIL ? »**
2D, incompressible (on reste à bas Mach), et $C_{L,\max}$ est **approché** : près du décrochage, le
décollement est massif et XFOIL perd en précision. On l'utilise donc pour **classer** les profils par
rapport à un seuil (1,2), pas pour une valeur certifiée. Certains profils ne convergent pas (104 sur
1 024 en croisière) : on les déclare **infaisables**, comme le ferait un ingénieur avec un calcul raté.

**« Pourquoi une grille discrète et pas des variables continues ? »**
La formulation QUBO (partie quantique) travaille sur des **bits**. Pour comparer équitablement NSGA-II,
la boucle QUBO et le front exact (énumération), toutes les méthodes travaillent sur **la même grille**.
Avec 1 024 profils, on peut aussi calculer le **front de Pareto exact**, ce qui permet de mesurer
objectivement la qualité de chaque méthode.

**« Pourquoi le NACA modifié en V2 ? »**
Dans le NACA classique, le rayon de nez est imposé par l'épaisseur ($r = 1{,}1019\,t^2$). Or le nez pilote
à la fois la SER (un nez fin réfléchit moins) et le décrochage. La série modifiée ajoute l'indice $I$, qui
**découple** le nez de l'épaisseur : on peut explorer le vrai compromis.

**« Pourquoi imposer $C_L = 0{,}4$ et pas l'incidence ? »**
En croisière, l'avion doit porter son poids : c'est la portance qui est fixée. L'incidence devient un
**résultat**, et c'est elle qui donne l'attitude vue par le radar. Comparer les profils à incidence
égale serait injuste : ils ne porteraient pas tous le même poids.

**« Comment comptez-vous le coût de calcul ? »**
Chaque profil **distinct** demandé par une méthode compte pour une simulation (`AeroSimulator` en V1,
`CountedProblem` en V2), même si on le relit dans la table pré-calculée. C'est la métrique de toutes les
comparaisons : avec un vrai solveur CFD, chaque simulation coûterait des minutes ou des heures.

---

## 7. Commandes

Depuis le dossier `Code/` (sous Windows, `python` au lieu de `python3`) :

```bash
python3 demo_profil.py                 # un profil (NACA 2412) de bout en bout + export STL/JSON
python3 validation.py                  # vérifications V1 à V5 du modèle rapide (V5 si XFOIL installé)
python3 validation_xfoil_grille.py     # XFOIL sur les 512 profils V1, comparé au modèle rapide
python3 -m optim.problem_v2            # table XFOIL des 1 024 profils V2 -> resultats/base_V2.csv
python3 verifier_installation.py       # vérifie qu'XFOIL tourne (un profil complet)
```

Petit test interactif (modèle rapide, sans XFOIL) :

```python
from aero import evaluate, evaluate_at_cl

r = evaluate(0.02, 0.4, 0.12, alpha=2.0)        # NACA 2412, α = 2°, Re = 3e6
print(r.CL, r.CD * 1e4, r.x_tr_up, r.valid)     # ≈ 0,50 ; 69 counts ; transition à 28 % ; True

r = evaluate_at_cl(0.02, 0.4, 0.12, 0.4)        # croisière à C_L = 0,4
print(r.alpha, r.CD * 1e4)                      # ≈ 1,19° ; 68 counts
```

Avec XFOIL (s'il est installé) :

```python
from aero.naca4 import naca4_modified
from aero.xfoil_runner import run_xfoil

X, Y = naca4_modified(0.02, 0.4, 0.12, I=4.0, T=0.3, n_half=120)
cr = run_xfoil(0.02, 0.4, 0.12, [], 3e6, coords=(X, Y), cls=[0.4])   # croisière à C_L = 0,4
print(cr["alpha"], cr["CD"] * 1e4)                                    # tableau vide = non convergé
```
