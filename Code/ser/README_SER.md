# Le dossier `ser/` expliqué

> **Dernière mise à jour : 28/11/2026 à 16h09**, d'après le code du commit `30d54af` (branche
> `branch_meddi`). Si un fichier de `ser/` a été modifié après cette date
> (`git log -- Code/ser`), ce README n'est peut-être plus à jour.

Ce document explique **ce que fait chaque fichier du dossier `ser/`**, **comment il s'insère dans le
reste du code** et **juste assez de théorie électromagnétique** pour comprendre le code et répondre au
jury. Les chiffres cités viennent des fichiers de `resultats/` ou ont été recalculés avec le code du
dossier (on dit d'où à chaque fois).

> **En une phrase.** Pour chaque profil d'aile, on calcule sa **SER** (surface équivalente radar,
> « à quel point l'avion est visible au radar »), vue par un radar à 10 GHz situé devant l'avion. On
> utilise deux méthodes : l'**optique physique** (rapide, approchée) et la **méthode des moments**
> (lente, exacte, la référence depuis la V2). La SER est le **deuxième objectif** de l'optimisation,
> en face de la traînée $C_D$.

---

## Sommaire

1. [Où se place la SER dans la chaîne](#1-où-se-place-la-ser-dans-la-chaîne)
2. [La théorie minimale](#2-la-théorie-minimale)
3. [Les fichiers un par un](#3-les-fichiers-un-par-un)
4. [Qui utilise quoi : interactions avec le reste du code](#4-qui-utilise-quoi--interactions-avec-le-reste-du-code)
5. [Les résultats à connaître](#5-les-résultats-à-connaître)
6. [Questions probables du jury](#6-questions-probables-du-jury)
7. [Commandes](#7-commandes)

---

## 1. Où se place la SER dans la chaîne

Le projet cherche des profils d'aile qui ont **à la fois** peu de traînée et peu de SER. Une
« simulation » d'un profil, dans `optim/problem_v2.py : simulate_v2`, enchaîne :

```
 indices (im, ip, it, iI)
        │
        ▼
 aero/naca4.py : naca4_modified(m, p, t, I)    ──► contour (X, Y) du profil
        │
        ├──► aero/xfoil_runner.py (XFOIL)
        │       croisière à C_L = 0,4  ──► C_D (objectif 1)  et  incidence α
        │       rampe 4°..14°          ──► C_L,max (contrainte ≥ 1,2)
        │                                         │
        │                                         │  α = l'ATTITUDE vue par le radar
        ▼                                         ▼
 ser/model.py : evaluate_ser(m, p, t, alpha=α, I=I)            (optique physique)
        └──► ser/po2d.py : echo_width(...)  ──► SER_S1, SER_S2 (objectif 2), en dBm
                                                    │
 ser/compare_po_mom.py grille  (méthode des moments, calculée une fois pour les 1 024 profils)
        └──► resultats/ser_mom_V2.csv  ──► relu par optim/problem_v2.py : with_objective(..., "MoM")
```

**Deux points clés :**
- **Couplage aéro → SER** : la SER dépend de l'incidence $\alpha$ du profil en vol, et cette incidence
  est **calculée par XFOIL**. On ne peut donc pas calculer la SER sans l'aérodynamique.
- **Le conflit** : la SER veut un **nez fin** (petit rayon de bord d'attaque), mais un nez trop fin
  **décroche tôt** ($C_{L,\max}$ trop faible) et la traînée en souffre. C'est ce qui crée un vrai
  front de Pareto. En V2, on a ajouté la variable $I$ (indice de rayon de bord d'attaque, indépendant
  de l'épaisseur) précisément pour piloter ce compromis.

`model.py` a été écrit pour avoir **la même interface que le solveur électromagnétique de l'autre
groupe** : on pourra le remplacer sans rien changer ailleurs.

---

## 2. La théorie minimale

### 2.1 Qu'est-ce que la SER ?

La **SER** (*RCS, Radar Cross Section*) d'un objet est la surface fictive qui, si elle renvoyait
l'énergie radar de manière isotrope, produirait le même écho que l'objet. Plus elle est petite, moins
l'objet est détectable. On l'exprime en **dB** car elle varie sur plusieurs ordres de grandeur :
$-10$ dB = 10 fois moins visible.

**Monostatique** : l'émetteur et le récepteur sont au même endroit (cas d'un radar classique). On
regarde donc l'onde renvoyée **dans la direction d'où elle vient**.

### 2.2 Pourquoi en 2D ?

On étudie une **section** d'aile (un profil), supposée infiniment longue en envergure. En 2D, la
grandeur naturelle est la **largeur équivalente** (*echo width*) $\sigma_{2D}$, en **mètres** (c'est
une SER par mètre d'envergure). Dans le code : $\text{SER} = 10\log_{10}(\sigma_{2D} / 1\,\text{m})$, en
**dBm** (« dB par mètre »).

Pour passer à une vraie aile 3D d'envergure $b$, vue perpendiculairement :
$\sigma_{3D} = k\,b^2\,\sigma_{2D}/\pi$ (fonction `sigma_3d_broadside`, non utilisée par l'optimisation
car elle ne change pas le classement des profils).

### 2.3 Le scénario radar

| Paramètre | Valeur | Pourquoi |
|---|---|---|
| Fréquence | 10 GHz (bande X, radar de conduite de tir) | λ = 3 cm, très petit devant la corde |
| Corde | 1 m | fixe le rapport corde/λ ≈ 33 |
| Conducteur | parfait (métal) | hypothèse standard, pas de matériau absorbant |
| Approche | de face (le radar est devant l'avion) | menace principale en pénétration |

La SER-objectif est une **moyenne sur un secteur d'élévations** (et pas une seule direction), car on ne
sait pas exactement où est le radar. Deux scénarios :

| Scénario | Secteur d'élévation | Interprétation |
|---|---|---|
| **S1** (« note », sujet du hackathon) | $[-20°, +20°]$ | radar à peu près dans l'axe de vol |
| **S2** (« sol ») | $[-60°, 0°]$ | radar au sol, vu **sous** la trajectoire |

La moyenne se fait sur $\sigma$ **linéaire** (pas en dB), puis on passe en dB : les directions où
l'écho est fort pèsent davantage, ce qui est réaliste.

### 2.4 Méthode 1 : l'optique physique (PO) — `po2d.py`

Hypothèse **haute fréquence** : l'objet est grand devant la longueur d'onde. Alors, sur un conducteur
parfait :
- les parties **éclairées** par le radar (normale tournée vers le radar, $n \cdot s > 0$) portent un
  courant $J = 2\,n \times H_{\text{inc}}$ ;
- les parties **dans l'ombre** ne portent aucun courant.

On découpe le contour en petites **facettes** droites ; chacune contribue à l'écho, et on somme les
contributions **avec leur phase** (elles interfèrent) :

$$\sigma_{2D} = k \left| \sum_{\text{éclairées}} (n_j \cdot s)\, L_j\, \mathrm{sinc}(k L_j\, t_j \cdot s)\, e^{2ik\, s \cdot r_j} \right|^2$$

- $k = 2\pi/\lambda$ ; $L_j$, $r_j$, $t_j$, $n_j$ = longueur, milieu, tangente, normale de la facette ;
- $(n_j \cdot s)\,L_j$ : surface « apparente » de la facette vue du radar ;
- $\mathrm{sinc}(\dots)$ : une facette inclinée renvoie moins dans la direction du radar ; elle renvoie
  au maximum quand elle lui fait face (**réflexion spéculaire**) ;
- $e^{2ik\,s\cdot r_j}$ : le **facteur 2** vient de l'aller-retour de l'onde.

**Intuition physique** : l'écho vient surtout des zones qui font **face** au radar. En approche de
face, c'est le **bord d'attaque** (le nez). Pour un nez circulaire de rayon $r$, la PO donne
$\sigma_{2D} \approx \pi r$ (comme un cylindre). **Gros nez = grosse SER**. C'est tout le compromis.

En PO 2D, les deux polarisations (TM et TE) donnent la même formule.

**Avantage** : très rapide (~5 ms par profil, mesuré). **Limite** : elle suppose que le nez est grand
devant λ. Or les nez fins de la V2 ont $k\,r_{BA} \approx 0{,}16$ à $0{,}4$ : on est **hors du domaine
de validité** de la PO. Elle ignore aussi la diffraction par le bord de fuite et les ondes rampantes.
D'où la méthode des moments.

### 2.5 Méthode 2 : la méthode des moments (MoM) — `mom2d.py`

C'est une méthode **exacte** (à la discrétisation près) : on résout les équations de Maxwell sur le
contour au lieu de supposer la forme du courant.

**Polarisation TM** : le champ électrique est parallèle à l'envergure (axe $z$). Sur un conducteur
parfait, le champ électrique **total** tangent doit être nul sur la surface. Le champ diffusé est
produit par le courant inconnu $J_z$ sur le contour, via la fonction de Green 2D (la fonction de Hankel
$H_0^{(2)}$). D'où l'**équation intégrale du champ électrique (EFIE)** :

$$E_z^{\text{inc}}(r) = \frac{k\eta}{4} \oint J_z(r')\, H_0^{(2)}(k|r - r'|)\, dl' \qquad \text{pour tout } r \text{ sur le contour}$$

**Discrétisation** (« méthode des moments » au sens le plus simple) :
- on découpe le contour en $N$ segments, avec un courant **constant** $J_n$ sur chacun ;
- on impose l'équation **au milieu** de chaque segment (*point matching*, collocation) ;
- on obtient un **système linéaire** $Z\,J = E$, avec $Z$ une matrice pleine $N \times N$ :
  - hors diagonale : $Z_{mn} = \frac{k\eta}{4} L_n H_0^{(2)}(k R_{mn})$ ;
  - sur la diagonale (un segment agit sur lui-même, la Hankel est singulière) : formule analytique
    $Z_{nn} = \frac{k\eta}{4} L_n \left[1 - \frac{2j}{\pi}\left(\ln\frac{\gamma k L_n}{4} - 1\right)\right]$,
    avec $\gamma = 1{,}781$ (constante d'Euler exponentiée) ;
- une fois $J$ trouvé, l'écho lointain est
  $\sigma_{2D} = \frac{k\eta^2}{4}\left|\sum_n J_n L_n e^{jk\,s\cdot r_n}\right|^2$.

**Coût** : $N \approx 1\,000$ segments par profil (λ/15 partout, raffiné à $r_{BA}/4$ au nez), soit une
matrice $1\,000 \times 1\,000$ à remplir et inverser : **~1 s par profil et par secteur** (mesuré), soit
~200 fois plus que la PO.

**Validation** : sur un **cylindre** conducteur, il existe une solution exacte en série (**Mie**,
fonction `mie_tm`). On l'utilise pour vérifier la MoM. **Limite connue** : l'EFIE a des **résonances
internes** parasites à certaines fréquences (les fréquences propres de la cavité intérieure), qu'on
détecte justement en comparant à Mie.

### 2.6 PO ou MoM : laquelle on utilise ?

| | Optique physique | Méthode des moments |
|---|---|---|
| Hypothèse | haute fréquence (objet ≫ λ) | aucune (Maxwell exact, discrétisé) |
| Coût par profil | ~5 ms | ~1 s par secteur |
| Validité sur nez fins | douteuse ($k\,r_{BA} < 1$) | bonne |
| Rôle | calcul « en ligne » dans `simulate_v2`, problème CST de la V3 | **référence** de la V2, calculée une fois pour les 1 024 profils |

---

## 3. Les fichiers un par un

### `__init__.py`
Fichier vide, fait de `ser/` un paquet Python (`from ser.model import ...`).

### `po2d.py` — optique physique 2D

| Fonction | Rôle |
|---|---|
| `wavenumber(freq_hz)` | $k = 2\pi f / c_0$ (210 rad/m à 10 GHz) |
| `facets(X, Y)` | milieux, tangentes, **normales sortantes** et longueurs des facettes du contour. Même convention que `aero/panel.py` : contour parcouru dans le sens trigonométrique (BF extrados → BA → BF intrados), normale $(t_y, -t_x)$ |
| `echo_width(X, Y, s_dirs, k)` | **cœur du fichier** : la formule de §2.4, vectorisée pour plusieurs directions radar à la fois (matrice directions × facettes) ; `lit = ns > 0` sélectionne les facettes éclairées |
| `radar_directions(alpha_deg, elev_deg)` | direction vers le radar **dans le repère du profil**. Le profil vole à l'incidence α ; un radar à l'élévation ε au-dessus de la trajectoire est vu dans la direction $s = -(\cos(\alpha-\varepsilon), \sin(\alpha-\varepsilon))$. C'est ici que l'attitude aéro entre dans la SER |
| `to_db(sigma)` | $10\log_{10}$, protégé contre $\log 0$ |
| `sigma_3d_broadside(...)` | passage 2D → 3D pour une aile d'envergure $b$ (pas utilisé dans l'optimisation) |

### `model.py` — point d'entrée « profil + attitude → SER »

- `RadarCase` (dataclass) : le scénario radar (10 GHz, corde 1 m, secteur ±20°, 81 directions,
  200 points par demi-contour). Ses propriétés `k` et `elevations` donnent le nombre d'onde et la
  grille d'élévations.
- `evaluate_ser(m, p, t, alpha, case, I, T)` : construit la géométrie (NACA 4 chiffres classique en
  V1, **modifiée** avec $I$ en V2), la met à l'échelle de la corde, calcule $\sigma_{2D}$ sur tout le
  secteur et renvoie un `SERResult` :
  - `ser_db` : **l'objectif**, moyenne linéaire sur le secteur, en dBm ;
  - `ser_nose_db` : radar exactement dans l'axe ($\varepsilon = 0$) ;
  - `ser_max_db` : pire direction du secteur ;
  - `r_le` : rayon de bord d'attaque (m) ; `pattern` : $\sigma_{2D}(\varepsilon)$ complet.

**Astuce pour les secteurs asymétriques** (S2 = $[-60°, 0°]$) : `RadarCase` ne sait faire qu'un secteur
centré $[-e, +e]$. Dans `optim/problem_v2.py : ser_sector`, on prend $e$ = demi-largeur et on décale
l'incidence : `alpha - centre`. Décaler toutes les élévations de $+c$ revient à décaler α de $-c$.

### `mom2d.py` — méthode des moments (V2)

| Fonction | Rôle |
|---|---|
| `resample(X, Y, h_max, refine)` | rééchantillonne le contour en segments de longueur ≤ `h_max`, avec un facteur de raffinement local (au nez) |
| `solve_tm(X, Y, k, s_dirs)` | remplit $Z$ (Hankel `scipy.special.hankel2`, terme propre analytique sur la diagonale), construit le second membre $E = e^{jk\,s\cdot r}$ pour **toutes** les directions d'un coup, résout $ZJ = E$ (`np.linalg.solve`, une seule factorisation pour toutes les directions) et renvoie $\sigma_{2D}$ |
| `mie_tm(ka, a)` | solution exacte d'un cylindre conducteur (série de Bessel/Hankel) pour **valider** `solve_tm` |

Le calcul utilise la **réciprocité** : la phase à l'émission et au retour est la même, d'où
`far = sum(L * J * E)`.

### `compare_po_mom.py` — comparaison PO / MoM (V2)

- `mom_contour(m, p, t, I)` : contour adapté à la MoM : λ/15 partout, **raffiné à $r_{BA}/4$ au nez**
  (`refine` passe progressivement de $r/4$ au nez à λ/15 au-delà de 3 % de corde).
- `ser_both(m, p, t, I, alpha, lo, hi)` : SER moyenne du secteur $[lo, hi]$ par **les deux méthodes**,
  sur les mêmes directions. Renvoie `(PO dB, MoM dB, nombre de segments)`.
- `_grid_one(row)` : pour un profil de `base_V2.csv`, calcule PO et MoM sur S1 et S2.
- En script :
  - `python -m ser.compare_po_mom` : tableau de comparaison sur 12 profils tests (épaisseur × $I$),
    avec un test de **convergence** (λ/15 contre λ/25) ;
  - `python -m ser.compare_po_mom grille` : MoM des **1 024 profils V2** en parallèle →
    `resultats/ser_mom_V2.csv`.

---

## 4. Qui utilise quoi : interactions avec le reste du code

```
aero/naca4.py  (naca4, naca4_modified, leading_edge_radius[_modified])
      │
      ▼
ser/po2d.py ◄────── ser/model.py ◄────────┬── optim/problem.py      (V1 : evaluate_ser, RadarCase)
      ▲                                    └── optim/problem_v2.py   (V2 : ser_sector → SER_S1, SER_S2)
      │
      ├──────────── ser/compare_po_mom.py ◄── verifier_installation.py  (ser_both sur un profil)
      │                    │
      │             ser/mom2d.py
      │                    │
      │                    └──► resultats/ser_mom_V2.csv ──► optim/problem_v2.py : with_objective(source="MoM")
      │                                                         └──► optim/experiences_v2.py, quantique/anatomie_qubo.py,
      │                                                              ia_qubo/* (objectif SER de toutes les comparaisons V2)
      └──────────── ia_qubo/probleme_cst.py   (V3 : profils CST à 30 bits, PO seulement, secteur S2)
```

- **V1** (`optim/problem.py`) : SER par PO, profil NACA classique, attitude α donnée par le modèle aéro
  rapide. Les deux scénarios s'appellent `S1_note` et `S2_sol`.
- **V2** (`optim/problem_v2.py`) : `simulate_v2` calcule la SER PO pour S1 et S2 et la stocke dans
  `base_V2.csv` ; `with_objective(table, "S1" ou "S2", "PO" ou "MoM")` choisit **quelle SER sert
  d'objectif**. Toutes les expériences V2 utilisent `"MoM"`.
- **V3** (`ia_qubo/probleme_cst.py`) : le problème à 30 bits (~10⁹ profils) ne peut pas utiliser la MoM
  (trop lente pour des milliers d'évaluations) : il utilise la PO, justifiée par la corrélation de
  rang PO/MoM de 0,999 (§5.2).
- La partie quantique (`quantique/`) **ne voit jamais la SER directement** : elle reçoit un modèle
  quadratique ajusté sur les valeurs de SER déjà simulées (voir `quantique/README_quantique.md`).

---

## 5. Les résultats à connaître

### 5.1 Validation de la MoM sur le cylindre (série de Mie)
*(recalculé avec `solve_tm` et `mie_tm`, cylindre discrétisé à λ/20, 10 GHz)*

| $ka$ | Mie (exact) | MoM | PO | $\pi a$ (limite optique) |
|---|---|---|---|---|
| 1 | −17,34 dBm | −17,34 dBm | −18,06 dBm | −18,24 dBm |
| 5 | −11,17 dBm | −11,16 dBm | −11,04 dBm | −11,25 dBm |
| 20 | −5,23 dBm | −5,29 dBm | −5,23 dBm | −5,23 dBm |

La MoM colle à la solution exacte. La PO est bonne pour les gros objets ($ka = 20$) mais se trompe
de ~0,7 dB pour $ka = 1$. **C'est exactement le régime de nos nez fins.**

### 5.2 PO contre MoM sur les 1 024 profils V2
*(`resultats/ser_mom_V2.csv`)*

| | Scénario S1 | Scénario S2 |
|---|---|---|
| Écart moyen MoM − PO | +1,05 dB | +0,66 dB |
| Écart-type de l'écart | 1,32 dB | 0,98 dB |
| Corrélation de rang (Spearman) | **0,999** | **0,999** |

Écart moyen MoM − PO (S1) selon le rayon de nez :

| $I$ (rayon de nez) | 2 (le plus fin) | 4 | 6 (NACA standard) | 8 (le plus rond) |
|---|---|---|---|---|
| MoM − PO | **+2,5 dB** | +1,0 dB | +0,45 dB | +0,22 dB |

**Lecture** : la PO **sous-estime** la SER des nez fins (jusqu'à 2,5 dB), comme prévu par la théorie
(hors domaine optique). Mais elle **garde presque parfaitement le classement** des profils. Pour une
optimisation, c'est le classement qui compte : la PO suffit pour explorer, et la MoM sert de référence
pour les chiffres. Le gain de SER des nez fins existe donc bien, mais il est **moins fort** que ce que
la PO laissait croire.

### 5.3 Ce que donne le compromis (front de Pareto V2, SER par MoM)
*(`resultats/pipeline_V2.json`, 702 profils faisables sur 1 024)*

- **S1** : front exact de **4 profils**. Moins de traînée : $m = 2\,\%$, $t = 8\,\%$, $I = 4$ →
  38 counts, −17,3 dBm. Moins de SER : $m = 5\,\%$, $t = 10\,\%$, $I = 2$ → 58,7 counts, −18,1 dBm.
- **S2** : front de **10 profils**, de 38 counts / −15,1 dBm jusqu'à −18,5 dBm (mais 319 counts).
- Les profils à faible SER ont un **nez fin** ($I = 2$) **et** plus de cambrure, donc une incidence de
  croisière plus **négative** : l'attitude, imposée par l'aéro, change ce que le radar voit. C'est le
  couplage aéro → SER du §1.
- Sur S1, les 4 profils du front ont $C_{L,\max}$ entre 1,20 et 1,34, donc proche de la limite de 1,2 :
  la contrainte de manœuvre empêche d'affiner le nez davantage. Aucun profil du front n'a le nez le
  plus rond ($I = 6$ ou $8$).

---

## 6. Questions probables du jury

**« Pourquoi une SER 2D et pas une vraie SER 3D ? »**
Le hackathon optimise un **profil** (une section d'aile). En 2D, la SER par unité d'envergure suffit à
classer les profils ; le passage en 3D pour une aile droite vue de face est un simple facteur
$k b^2/\pi$ qui ne change pas le classement (`sigma_3d_broadside`). `aero/geometry_export.py` exporte
déjà la géométrie extrudée (STL) pour un futur solveur 3D.

**« L'optique physique est-elle valable ? »**
Pas sur les nez les plus fins ($k\,r_{BA} < 1$). C'est pour ça qu'on a écrit la **méthode des moments**
en V2, validée contre la solution exacte de Mie. Résultat : la PO sous-estime la SER des nez fins
(jusqu'à +2,5 dB d'écart), mais garde le classement (corrélation de rang 0,999). La MoM est la
référence pour tous les résultats V2.

**« Pourquoi la polarisation TM seulement ? »**
C'est la plus simple à écrire (courant scalaire selon l'envergure, EFIE avec $H_0^{(2)}$). La TE demande
une équation avec des dérivées normales (hypersingulière), plus délicate. En PO, TM et TE donnent la
même formule. C'est une limite assumée : compléter avec la TE est une piste d'amélioration.

**« Pourquoi moyenner sur un secteur ? Pourquoi en linéaire ? »**
On ne sait pas exactement où est le radar. Une seule direction rendrait l'optimisation sensible à un
creux d'interférence accidentel. La moyenne linéaire donne plus de poids aux directions où l'écho est
fort, ce qui correspond au risque réel de détection.

**« Pourquoi l'incidence α intervient-elle ? »**
Le radar voit le profil **dans son attitude de vol**. Cette attitude est l'incidence nécessaire pour
voler à $C_L = 0{,}4$, calculée par XFOIL. Deux profils différents ne présentent pas leur nez sous le
même angle : la SER dépend donc de l'aérodynamique.

**« Comment savez-vous que la MoM est convergée ? »**
(1) Validation sur le cylindre contre Mie (écart < 0,1 dB). (2) `compare_po_mom.py` sans argument
recalcule chaque profil test avec une discrétisation plus fine (λ/25 au lieu de λ/15). (3) Le nez est
raffiné à $r_{BA}/4$, donc même le nez le plus fin est décrit par plusieurs segments.

**« Pourquoi pas un vrai solveur électromagnétique du commerce ? »**
Le module `ser/model.py` a été écrit avec **la même interface** que le solveur de l'autre groupe, pour
pouvoir le remplacer sans changer l'optimisation. Nos solveurs maison (PO et MoM 2D) permettent de
tester toute la chaîne sur les 1 024 profils en quelques minutes.

---

## 7. Commandes

Depuis le dossier `Code/` (sous Windows, `python` au lieu de `python3`) :

```bash
python3 -m ser.compare_po_mom          # PO contre MoM sur 12 profils tests (+ test de convergence)
python3 -m ser.compare_po_mom grille   # MoM des 1 024 profils V2 -> resultats/ser_mom_V2.csv (lit base_V2.csv)
python3 verifier_installation.py       # test rapide de toute la chaîne, dont une SER PO et MoM
```

Petit test interactif :

```python
from ser.model import evaluate_ser
from ser.compare_po_mom import ser_both

r = evaluate_ser(0.02, 0.4, 0.12, alpha=1.0)           # NACA 2412, α = 1°, secteur S1
print(r.ser_db, r.ser_nose_db, r.ser_max_db)           # ≈ -12,9 / -13,2 / -12,1 dBm (~5 ms)

po, mom, n = ser_both(0.02, 0.4, 0.12, 6.0, 1.0, -20, 20)
print(po, mom, n)                                       # ≈ -12,87 dBm (PO), -12,58 dBm (MoM), 1 022 segments (~1 s)
```
