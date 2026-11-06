# Archives du code, par itération

Le code **actif** est dans `Code/` (itération en cours). Chaque itération close est
**figée** ici, avec ses résultats, pour qu'on puisse toujours relancer, citer ou
comparer une version antérieure.

Le journal complet (pourquoi chaque itération, ce qui a changé, les résultats) est dans
la fiche projet : `Fiche_Projet/Fiche_Projet_Iterations.pdf`.

## Règles

1. **Clore une itération** = copier `aero/`, `ser/`, `optim/`, `quantique/`, les scripts
   et `resultats/` (sans les `.log`) dans `archives/<Vk>_<nom>/`, puis ajouter une ligne ci-dessous.
2. **Rendre un module obsolète** pendant une itération = le retirer de `Code/` et le
   laisser dans l'archive de l'itération précédente ; noter dans le tableau
   « Remplacements » ce qui le remplace et pourquoi.
3. Une archive n'est **jamais modifiée** après sa création.
4. Pour relancer une archive : `cd archives/<Vk>_<nom>/code && python3 <script>.py`
   (chaque archive est autonome ; les chemins vers `resultats/` sont relatifs à `code/`,
   il faut créer un lien `code/resultats -> ../resultats`).

## Itérations figées

| Itération | Dossier | Contenu | Résultats principaux |
|---|---|---|---|
| V0 — modèle de la note d'intention (24-30 oct. 2026) | `V0_note_intention/` | `aero/` (panneaux, couche limite, grille), `quantique/` (QUBO mono-objectif $C_D$, QAOA, recuit), scripts V0 | base 512 profils ($\alpha=2°$ et $C_L=0{,}4$), $R^2$ quadratique 0,91, boucle $C_D$ seule : 25 simulations contre 256 au hasard |
| V1 — chaîne complète conforme à la note (4-5 nov. 2026) | `V1_chaine_complete/` | + `ser/` (optique physique 2D), `optim/` (problème, Pareto, NSGA-II, boucle QUBO bi-objectif), XFOIL local | fronts exacts S1/S2 dégénérés ($t=t_{\min}$), boucle 1,2 à 3 fois plus économe que NSGA-II, $C_D$ du modèle rapide +27 % face à XFOIL |
| V2 — problème reformulé (5-6 nov. 2026) | `V2_probleme_reformule/` | + NACA 4 chiffres modifiée (rayon de bord d'attaque), XFOIL de référence (`optim/problem_v2.py`), manœuvre $C_{L,\max}\ge1{,}2$, SER par méthode des moments (`ser/mom2d.py`), comparaison avec arrêt réaliste (`optim/experiences_v2.py`), montée en taille QAOA (`quantique/montee_en_taille.py`) | vrai compromis traînée/SER (épaisseur et rayon de BA) ; boucle exacte : front en 75 (S1) et 142 (S2) simulations, NSGA-II 0 à 5 % de réussite en 250 ; QAOA $p=4$ à 18 qubits : 59 % des mesures dans le 1 % meilleur |

## Remplacements

| Élément | Itération | Remplacé par | Pourquoi |
|---|---|---|---|
| `aero/xfoil_runner.py` (version V0, `PLOP G F`) | V1 | version V1 (copie locale d'XFOIL + écran virtuel Xvfb) | la compilation Debian d'XFOIL plante sans affichage graphique |
| `quantique/recuit.py` (sans `return_states`) | V1 | version V1, compatible | la boucle bi-objectif a besoin de plusieurs candidats par lecture |
| `quantique/experiences.py` → `boucle()` (mono-objectif) | V1 | `optim/qubo_loop.py` (bi-objectif, contraintes) | toujours présent dans `Code/` car il produit les figures du cours quantique (V0) |
| `optim/experiences_pipeline.py` (comparaison V1) | V2 | `optim/experiences_v2.py` | toujours présent dans `Code/` : il produit `resultats/pipeline.json`, lu par les figures V1 du cours d'optimisation et par la fiche |
| `optim/problem.py` → `scenario`, `build_table` (modèle rapide, 512 profils) | V2 | `optim/problem_v2.py` (XFOIL, 1 024 profils) | `CountedProblem` de `problem.py` reste utilisé par la V2 |
| SER par optique physique comme référence | V2 | `ser/mom2d.py` (méthode des moments TM) | l'optique physique n'est plus valable sur les nez aigus ; elle reste en contrôle |
