# `contraintes/` — lot L7 de la V3 : traiter les contraintes de la boucle QUBO

Ce dossier remplace la pénalité à bits d'écart de la boucle par des formulations sans qubit
supplémentaire, ajoute un classifieur de convergence d'XFOIL, et exporte le problème au format
des solveurs contraints. La boucle de `ia_qubo/boucle.py` reçoit seulement de nouveaux modes
(`contrainte="lagrange"`, `"lagrange_clf"`, `"desequilibree"`) ; l'approche actuelle reste
identique (`ia_qubo.boucle.verifier_boucle()`).

Cours associé : `Cours/7_Contraintes/Cours_Contraintes.pdf`.

| Fichier | Rôle |
|---|---|
| `formulations.py` | `Contrainte` (h(b) >= 0 sous forme QUBO, `verifiee=False` pour une contrainte qui oriente seulement), pénalité déséquilibrée, lagrangien, `resout_lagrange` (montée duale sur les multiplicateurs + vérification classique des lectures) |
| `classifieur.py` | `LogistiqueQuadZ` : probabilité de convergence d'XFOIL, logit quadratique en `z` (donc contrainte QUBO) ; `auc` |
| `experiences_v2.py` | formulations sur le problème V2, 20 graines, sans XFOIL → `resultats/contraintes_V2.json` ; `... experiences_v2 seuils` : seuil du classifieur et diagnostic du front → `contraintes_V2_seuils.json` |
| `etude_cst.py` | rejeu de la seconde campagne CST à partir du cache, sans XFOIL : origine des C_L,max trop faibles, valeur du classifieur, formulations sur les mêmes QUBO → `resultats/contraintes_cst.json` |
| `qaoa_formulations.py` | QAOA (p = 2) selon la formulation, petites instances V2 → `resultats/contraintes_qaoa.json` |
| `export_cqm.py` | objectif + contraintes séparés au format LP (CPLEX, Gurobi, SCIP, `dimod` de D-Wave) et JSON → `resultats/cqm_exemple.lp`, `cqm_exemple.json` |
| `experiences_cst.py` | campagne XFOIL : `quadz_lagrange`, `quadz_lagrange_clf` × 8 graines, comparées aux exécutions de la seconde campagne L6 → `resultats/contraintes_cst_campagne.json` (+ `_brut.json`) |

Les contraintes géométriques exactement affines en `z` (non-croisement des faces) sont dans
`ia_qubo/probleme_cst.py` (`contraintes_geometriques`, `SPEC["geometrie_z"]`).

## Lancer

```bash
cd Code
python3 -m contraintes.experiences_v2          # ~5 min sur 11 processus
python3 -m contraintes.experiences_v2 seuils   # ~3 min
python3 -m contraintes.etude_cst               # ~10 min
python3 -m contraintes.qaoa_formulations       # ~3 min
python3 -m contraintes.export_cqm              # quelques secondes
python3 -m contraintes.experiences_cst         # ~1 h 10 (XFOIL, PC sur secteur, sans veille)
python3 -m contraintes.experiences_cst bilan   # relit les résultats bruts, sans XFOIL
```

XFOIL tourne sur un écran virtuel (Xvfb) : aucune fenêtre ne s'affiche. Pour les voir :
`XFOIL_FENETRES=1 python3 -m ...`.
