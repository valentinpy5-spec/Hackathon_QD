# Le dossier `resultats/` (et `figures/`) expliqué

> **Dernière mise à jour : 28/11/2026 à 17h05**, d'après le contenu du dossier au commit `30d54af`
> (branche `branch_meddi`). Si un fichier a été ajouté ou régénéré après cette date
> (`git log -- Code/resultats`), ce README n'est peut-être plus à jour.

Ce dossier contient **toutes les données produites par le code** : les tables de simulations (CSV), les
résultats des comparaisons de méthodes (JSON) et le cache du problème à 30 bits. Les cours
(`Cours/*/tools/make_figures*.py`) et la fiche projet (`Fiche_Projet/tools/chiffres.py`) **relisent ces
fichiers** pour leurs figures et leurs chiffres : **un chiffre présenté au jury doit pouvoir être retrouvé
ici**.

Pour chaque fichier : **qui le produit**, **qui le lit**, **ce qu'il contient**, et les chiffres clés.
Les explications détaillées des résultats sont dans les README des dossiers de code (lien entre
parenthèses).

---

## 1. Vue d'ensemble : du calcul aux figures

```
 aero/ + ser/ (simulateurs)
        │
        ├─ V0-V1 : modèle rapide ──► base_aero_*.csv (512) ──► base_complete_S1_note / S2_sol.csv (512, + SER)
        │                            base_xfoil_a2.csv (512, contrôle XFOIL)
        │
        ├─ V2 : XFOIL ─────────────► base_V2.csv (1 024) ──┐
        │       méthode des moments ► ser_mom_V2.csv (1 024)┤
        │                                                   ▼
        │                     optim/ : CountedProblem lit ces tables (simulation « comptée »)
        │                                                   │
        │          ┌────────────────────────────────────────┼──────────────────────────────┐
        │          ▼                                        ▼                              ▼
        │   pipeline.json (V1)                      pipeline_V2.json (V2)          pipeline_L1_V3*.json (V3)
        │   check_pymoo.json                        anatomie_qubo_V3.json          diagnostic_L1_V3.json
        │                                           trace_boucle_V2.json           ia_qubo_V2.json, ...
        │
        └─ V3 : CST 30 bits ───────► cache_cst.jsonl (simulations à la demande) ──► ia_qubo_echelle*.json

 quantique/ (sans simulateur) ─────► experiences_quantique.json, montee_en_taille*.json
                                              │
                                              ▼
                     Cours/*/tools/make_figures*.py, Fiche_Projet/tools/chiffres.py
```

**Les tables servent de cache** : `build_table_v2()` relit `base_V2.csv` s'il existe au lieu de relancer
XFOIL (≈ 6 min sur 10 cœurs). Pour tout recalculer, supprimer le fichier.

---

## 2. Les tables de simulations (CSV)

| Fichier | Itération | Lignes | Produit par | Contenu |
|---|---|---|---|---|
| `base_aero_Re3e+06_a2.csv` | V0 | 512 | `generer_base.py` | **modèle aéro rapide** à $\alpha = 2°$, $Re = 3\cdot10^6$ : `CL`, `CD`, `CD_f` (frottement), `CM`, transitions (`x_tr_*`, `tr_mode_*`), décollements, `r_le`, `valid`, `flags`, indices `im, ip, it`, `bits`, contraintes `g` |
| `base_aero_Re3e+06_CL0.4.csv` | V0 | 512 | `generer_base.py` | même chose à **portance imposée** $C_L = 0{,}4$ (l'incidence `alpha` devient une sortie) |
| `base_complete_S1_note.csv` | V1 | 512 | `optim/experiences_pipeline.py` (via `build_table`) | **simulation V1 complète** = aéro rapide + SER (optique physique), scénario S1 (élévations $[-20°, 20°]$) : `CD_counts`, `SER_db`, `SER_nose_db`, `SER_max_db`, `cv` (violation), `feasible`. 392 faisables |
| `base_complete_S2_sol.csv` | V1 | 512 | idem | même chose, scénario S2 (radar au sol, $[-60°, 0°]$) |
| `base_xfoil_a2.csv` | V1 | 512 | `validation_xfoil_grille.py` | **XFOIL** sur la grille V1 à $\alpha = 2°$ : `xf_ok`, `xf_CL`, `xf_CD`, `xf_CM`, transitions. 498 convergés. Sert à valider le modèle rapide (`aero/README_aero.md`, §5.2) |
| **`base_V2.csv`** | V2 | 1 024 | `python -m optim.problem_v2` | **la table de référence V2** : indices `im, ip, it, iI` et valeurs `m, p, t, I`, `r_le`, croisière XFOIL à $C_L = 0{,}4$ (`alpha`, `CD`, `CD_counts`, `xtr_up`, `valid`), manœuvre (`CLmax`, `alpha_clmax`, `n_man`), SER par optique physique (`SER_S1`, `SER_S2`), `cv`, `feasible`. 920 croisières convergées, **702 faisables** |
| **`ser_mom_V2.csv`** | V2 | 1 024 | `python -m ser.compare_po_mom grille` | SER des 1 024 profils V2 par **optique physique** et par **méthode des moments**, S1 et S2 (`SER_S1_PO`, `SER_S1_MoM`, ...), `N` = nombre de segments MoM. Relu par `with_objective(..., "MoM")` : **c'est la SER-objectif de toutes les expériences V2 et V3** (`ser/README_SER.md`, §5.2) |

## 3. Le cache du problème à 30 bits

| Fichier | Produit par | Contenu |
|---|---|---|
| `cache_cst.jsonl` (≈ 7,7 Mo, 28 464 lignes) | `ia_qubo/probleme_cst.py : ProblemeSimule` | une ligne JSON par profil CST **réellement simulé** (XFOIL + SER PO), toutes méthodes, graines et campagnes confondues ; **valeurs brutes** d'XFOIL (le seuil de traînée aberrante est appliqué à la relecture). Une même simulation n'est jamais refaite. Le supprimer = refaire ~1 h 45 de calcul (`ia_qubo/README_ia_qubo.md`, §2.6) |

## 4. Les résultats des expériences (JSON)

### Partie quantique

| Fichier | Itération | Produit par | Contenu, chiffres clés |
|---|---|---|---|
| `experiences_quantique.json` | V0 | `python -m quantique.experiences` | `toy` (QUBO à 3 qubits : énergies, Ising, spectre adiabatique, paysage QAOA $p = 1$), `aero` (QUBO de $C_D$ à 9 qubits : QAOA $p = 1..5$, P(optimum) 0,018 → 0,142 ; recuit), `boucle` (boucle mono-objectif : 25 simulations contre 256 au hasard) |
| `montee_en_taille.json` | V2 | `python -m quantique.montee_en_taille` | QAOA de 6 à 18 qubits, $p = 1..4$ : P(optimum), P(1 % meilleur), rapport d'approximation, coût en CNOT, recuit. À 18 qubits, $p = 4$ : 59 % des mesures dans le 1 % meilleur |
| `montee_en_taille_init_aleatoire.json` | V2 | 1ʳᵉ version du même script (angles initiaux aléatoires) | gardé pour la comparaison du cours quantique (plafonnement dès $p = 3$) |
| `anatomie_qubo_V3.json` | V3 | `python -m quantique.anatomie_qubo` | anatomie des QUBO de la boucle V2 : part de l'objectif dans l'énergie (1–2 %), écart entre les deux meilleurs profils, QAOA avec contraintes (gain ×1 à ×5 sur le hasard) et sans (×30 à ×400), spectre |
| `trace_boucle_V2.json` | V3 | `python -m quantique.trace_boucle_v2` | une exécution de la boucle (S2, graine 0) pas à pas : candidats, prédictions, vraies valeurs, $Q$ complet du premier QUBO, recuit à deux températures. Front complet en 180 simulations |

(Détails : `quantique/README_quantique.md`, §5.)

### Comparaison des méthodes d'optimisation

Format commun : pour chaque scénario (`S1`/`S2` ou `S1_note`/`S2_sol`), une liste de **traces** par méthode
et par graine (chaque point = `[n_sims, fraction du front exact, hypervolume relatif]`) et un `summary` :
`full_front` et `hv99` = quartiles [25 %, médiane, 75 %] du nombre de simulations pour retrouver tout le
front / 99 % de l'hypervolume (`nan` = pas atteint par assez de graines dans le budget), `success` = part des
graines qui retrouvent le front complet, et pour V2/V3 l'arrêt réaliste `stop_sims`, `stop_frac`, `stop_hv`.

| Fichier | Itération | Produit par | Contenu, chiffres clés |
|---|---|---|---|
| `pipeline.json` | V1 | `python -m optim.experiences_pipeline` | hasard, NSGA-II, boucle QUBO exact / recuit / QAOA sur les 512 profils V1. Front complet (médiane S1) : boucle 35, NSGA-II 106,5, hasard 454 |
| `check_pymoo.json` | V1 | `python -m optim.check_pymoo` | simulations pour retrouver le front, 20 graines, notre NSGA-II (`ours`) contre pymoo. Médianes : 106,5 contre 95 (S1), 106 contre 110,5 (S2) |
| **`pipeline_V2.json`** | V2 | `python -m optim.experiences_v2` | la comparaison principale : hasard, NSGA-II ($N = 8$, $16$), boucle exact / recuit / QAOA sur la table XFOIL + SER MoM ; contient aussi le **front exact** (`front`) et tous les profils faisables (`feasible`). HV 99 % (médiane S1) : boucle exacte 62,5, hasard 632 ; NSGA-II ne retrouve jamais le front complet |
| `pipeline_L1_V3.json` | V3 (L1) | `python -m optim.experiences_l1` | boucle QUBO contre l'ablation bayésienne (krigeage, EI, ParEGO, contraintes strictes), 20 graines. La boucle QUBO fait aussi bien ou mieux ; les contraintes strictes échouent |
| `pipeline_L1_V3_rapide.json` | V3 (L1) | `python -m optim.experiences_l1 rapide` | même chose, 4 graines (test rapide) |
| `diagnostic_L1_V3.json` | V3 (L1) | `python -m optim.diagnostic_l1` | `constraint` : effet du poids de pénalité et part des candidats infaisables ; `quality` : erreur des modèles selon le nombre de points ; `front` : ce que prédisent les modèles de $C_{L,\max}$ sur les profils du front |

(Détails : `optim/README_optim.md`, §5.)

### Modèles appris (lot L6)

| Fichier | Produit par | Contenu, chiffres clés |
|---|---|---|
| `ia_qubo_V2.json` | `python -m ia_qubo.experiences_v2_ia` | 9 méthodes (`quadz`, `fm`, `bocs`, `mlp` × contrainte modélisée / boîte noire, + `hybride`) sur V2, 20 graines. Le modèle V2 (`quadz_modele`) reste le meilleur ou à égalité ; toutes les variantes boîte noire échouent |
| `ia_qubo_diagnostic_V2.json` | `python -m ia_qubo.diagnostic_v2_ia` | pour chaque profil du front : supporté ou non, fréquence à laquelle chaque modèle le trouve. Explique pourquoi le réseau, plus précis, trouve moins bien le front |
| `ia_qubo_precision.json` | `python -m ia_qubo.precision_modeles` | erreur et corrélation de rang de chaque modèle pour 10 à 500 profils d'entraînement, sur V2 (`v2`) et CST (`cst`) |
| `ia_qubo_qaoa_contraintes.json` | `python -m ia_qubo.qaoa_contraintes` | QAOA $p = 2$ sur QUBO à contrainte modélisée (14 qubits) ou en boîte noire (11 qubits) : part de l'objectif et gain sur le hasard |
| `ia_qubo_echelle.json` | `python -m ia_qubo.experiences_echelle` | problème CST à 30 bits, 2ᵉ campagne (avec filtre géométrique), 7 méthodes, 8 graines, budget 300 : hypervolume au fil des simulations (`hv_median`, quartiles), front de référence. HV à 100 simulations : FMQA/BOCS 0,83, modèle V2 0,77, NSGA-II 0,72, hasard 0,68 |
| `ia_qubo_echelle_brut.json` | idem | les listes de profils simulés par chaque exécution (pour retracer ou recalculer) |
| `ia_qubo_echelle_sansfiltre.json`, `..._sansfiltre_brut.json` | 1ʳᵉ campagne (sans filtre) | gardés pour la comparaison : le modèle V2 y dépensait 59 % de son budget sur des géométries invalides |

(Détails : `ia_qubo/README_ia_qubo.md`, §5.)

## 5. Démonstration et export

| Fichier | Produit par | Contenu |
|---|---|---|
| `NACA2412.stl` | `demo_profil.py` (via `aero/geometry_export.py`) | maquette 2,5D extrudée du NACA 2412, en triangles (STL ASCII), pour un solveur SER 3D ou ParaView |
| `NACA2412.json` | idem | métadonnées associées : $m, p, t$, incidence, corde, envergure, repère, note sur la direction radar |

---

## 6. Le dossier `figures/`

Figures PNG produites par les scripts **V0** à la racine de `Code/` (les figures des cours sont produites
ailleurs, par `Cours/*/tools/`) :

| Fichier | Produit par | Contenu |
|---|---|---|
| `demo_NACA2412.png` | `demo_profil.py` | géométrie, $C_p$ et couche limite du NACA 2412 |
| `polaire_NACA2412.png` | `demo_profil.py` | polaire du NACA 2412 par le modèle rapide, de −4° à 10° ($C_L(\alpha)$ et $C_L(C_D)$, points décollés marqués) |
| `V2_V3_plaque_plane.png` | `validation.py` | validations V2 et V3 du modèle rapide : Thwaites contre Blasius, Head contre la loi turbulente en 1/7, sur plaque plane |
| `analyse_quadratique.png` | `analyse_quadratique.py` | la sortie aéro est-elle « QUBO-compatible » ? Qualité de deux modèles de $C_D$ sur la base V0 : quadratique en $(m, p, t)$ (10 coefficients, celui de la boucle) et QUBO direct sur les 9 bits (46 coefficients), avec l'effet d'un petit échantillon |

---

## 7. Règles (rappel du `Code/README.md`)

- Un nouveau résultat s'appelle `resultats/<nom>_V3.<ext>` (suffixe de l'itération).
- Les fichiers `.log` ne sont pas archivés.
- Quand une itération est close, `resultats/` est copié dans `archives/<Vk>_<nom>/resultats/` (voir
  `archives/README.md`) ; une archive n'est jamais modifiée.
- Ne pas régénérer un fichier lu par les cours sans relancer leurs figures : les chiffres des PDF ne
  correspondraient plus.
