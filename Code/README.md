# Code

Chaîne complète : géométrie du profil → aérodynamique → SER → optimisation bi-objectif
(NSGA-II ou boucle QUBO avec solveur exact, recuit simulé ou QAOA).
Toutes les commandes se lancent **depuis ce dossier** (`Code/`).

Ce dossier contient l'état de la dernière itération close (**V2**) ; les versions
précédentes sont figées dans `archives/` (voir `archives/README.md`).

## Installation

```bash
pip install -r requirements.txt   # NumPy, SciPy, Matplotlib (+ pymoo, optionnel)
./installer_outils.sh             # Debian/Ubuntu : .venv avec pymoo, XFOIL et Xvfb locaux, sans droits admin
python3 verifier_installation.py  # un profil complet (XFOIL, SER) et une boucle QUBO, sans toucher à resultats/
```
XFOIL est indispensable à partir de la V2 (simulateur de référence). Sous Debian/Ubuntu, le script
ci-dessus le récupère ; ailleurs, installer XFOIL et le mettre dans le `PATH`
(`aero/xfoil_runner.py` le cherche d'abord dans le `PATH`, puis dans `.tools/`).
Sans écran (serveur, SSH), `xfoil_runner` lance tout seul un écran virtuel Xvfb.

## Arborescence

La colonne « depuis » donne l'itération qui a créé le fichier ; « rôle en V2 » dit s'il sert encore.

| Fichier | Depuis | Contenu | Rôle en V2 |
|---|---|---|---|
| `aero/naca4.py` | V0 | géométrie NACA 4 chiffres ; en V2 : série **modifiée** (indice $I$ de rayon de bord d'attaque) | actif |
| `aero/panel.py`, `aero/boundary_layer.py`, `aero/model.py` | V0 | modèle rapide : panneaux de Hess-Smith, Thwaites/Michel/Head, Squire-Young | figures V0-V1 des cours |
| `aero/design_space.py` | V0 | grille 8×8×8 (9 bits), encodage binaire, simulateur compté | V1 |
| `aero/geometry_export.py` | V0 | extrusion 2,5D → STL + JSON (interface vers un solveur SER 3D) | à reprendre en V3 |
| `aero/xfoil_runner.py` | V0, revu V1-V2 | pilotage d'XFOIL ($\alpha$ ou $C_L$ imposé, coordonnées libres, Xvfb) | **actif** |
| `ser/po2d.py`, `ser/model.py` | V1 | SER 2D par optique physique, secteur angulaire, attitude de vol | **actif** (contrôle) |
| `ser/mom2d.py` | V2 | SER 2D exacte : méthode des moments, polarisation TM, validée sur Mie | **actif** (référence) |
| `ser/compare_po_mom.py` | V2 | comparaison optique physique / méthode des moments ; SER des 1 024 profils | **actif** |
| `optim/pareto.py` | V1 | dominance, tri non dominé, encombrement, hypervolume 2D | **actif** |
| `optim/nsga2.py` | V1 | NSGA-II écrit à la main (variables entières, contraintes de Deb) | **actif** |
| `optim/qubo_loop.py` | V1, généralisé V2 | boucle QUBO bi-objectif : modèles quadratiques, somme pondérée, contraintes, solveurs exact / recuit / QAOA | **actif** |
| `optim/problem.py` | V1 | problème V1 (modèle rapide, 512 profils) ; `CountedProblem` (compteur de simulations) | `CountedProblem` actif |
| `optim/problem_v2.py` | V2 | problème V2 : table XFOIL des 1 024 profils, contraintes, scénarios S1/S2 | **actif** |
| `optim/experiences_pipeline.py` | V1 | comparaison des méthodes V1 → `resultats/pipeline.json` | figures V1 |
| `optim/experiences_v2.py` | V2 | comparaison des méthodes V2, arrêt réaliste → `resultats/pipeline_V2.json` | **actif** |
| `optim/check_pymoo.py` | V1 | vérification croisée de NSGA-II contre pymoo | V1 |
| `quantique/qubo.py`, `qaoa.py`, `recuit.py` | V0 | QUBO / Ising, QAOA par vecteur d'état, recuit simulé | **actif** |
| `quantique/montee_en_taille.py` | V2 | QAOA de 6 à 18 qubits, initialisation INTERP → `resultats/montee_en_taille.json` | **actif** |
| `quantique/experiences.py` | V0 | expériences du cours quantique → `resultats/experiences_quantique.json` | figures V0 |
| `validation.py` | V0 | validations du modèle rapide (V1 à V5 du cours aéro) | V0 |
| `validation_xfoil_grille.py` | V1 | XFOIL sur les 512 profils V1, comparé au modèle rapide | V1 |
| `verifier_installation.py` | V2 | test de la chaîne sur un exemple, en quelques secondes | **actif** |
| `demo_profil.py`, `generer_base.py`, `analyse_quadratique.py`, `plot_style.py` | V0 | démonstration sur un profil, base exhaustive V0, qualité d'un QUBO, style | V0 |
| `resultats/` | — | tables et résultats de toutes les itérations (voir ci-dessous) | |
| `figures/` | V0 | figures PNG des scripts V0 | |

## Lancer la V2

```bash
python3 -m optim.problem_v2            # table XFOIL des 1 024 profils (≈ 6 min sur 10 cœurs) -> resultats/base_V2.csv
python3 -m ser.compare_po_mom grille   # SER par méthode des moments            -> resultats/ser_mom_V2.csv
python3 -m optim.experiences_v2        # comparaison des méthodes (≈ 1 h)        -> resultats/pipeline_V2.json
python3 -m optim.experiences_v2 PO rapide   # moins de graines, SER par optique physique (réécrit pipeline_V2.json)
python3 -m quantique.montee_en_taille  # QAOA de 6 à 18 qubits                   -> resultats/montee_en_taille.json
```
`base_V2.csv` sert de cache : s'il existe, `build_table_v2` le relit sans relancer XFOIL (le supprimer
pour tout recalculer). `ser_mom_V2.csv` est relu par `with_objective(..., "MoM")`.
Les cours et la fiche projet relisent `resultats/` pour leurs figures et leurs chiffres
(`Cours/*/tools/make_figures*.py`, `Fiche_Projet/tools/chiffres.py`).

## Résultats

| Fichier | Itération | Produit par |
|---|---|---|
| `base_aero_Re3e+06_a2.csv`, `base_aero_Re3e+06_CL0.4.csv`, `NACA2412.*`, `experiences_quantique.json` | V0 | `generer_base.py`, `demo_profil.py`, `quantique.experiences` |
| `base_complete_S1_note.csv`, `base_complete_S2_sol.csv`, `pipeline.json`, `base_xfoil_a2.csv`, `check_pymoo.json` | V1 | `optim.experiences_pipeline`, `validation_xfoil_grille.py`, `optim.check_pymoo` |
| `base_V2.csv`, `ser_mom_V2.csv`, `pipeline_V2.json`, `montee_en_taille.json` | V2 | voir « Lancer la V2 » |
| `montee_en_taille_init_aleatoire.json` | V2 | première version de la montée en taille (angles aléatoires), gardée pour la comparaison du cours quantique |

## Ajouter du code pour une nouvelle itération

- Nouveau module ou script : suffixe `_v3` s'il remplace un fichier existant (`problem_v3.py`), pour
  que la V2 reste relançable depuis `Code/` tant que les cours en dépendent.
- Nouveau résultat : `resultats/<nom>_V3.<ext>`.
- Le docstring du fichier dit l'itération, ce qui change par rapport à la précédente et la commande à lancer
  (voir `optim/problem_v2.py`).
