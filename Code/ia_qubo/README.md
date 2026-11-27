# `ia_qubo/` — lot L6 de la V3 : apprendre le QUBO

Ce dossier teste des **modèles appris** à la place du modèle quadratique en `z` de la boucle QUBO
(approche FMQA / BOCS), sur le problème V2 et sur un problème mis à l'échelle (forme CST, 30 bits).
Il ne modifie aucun fichier existant : il importe les briques de `optim/` et `quantique/`.
L'approche actuelle (`optim/qubo_loop.py`) reste la référence ; `boucle.verifier_boucle()`
vérifie que la boucle générique la reproduit à l'identique.

Cours associé : `Cours/6_IA_QUBO/Cours_IA_QUBO.pdf`.

| Fichier | Rôle |
|---|---|
| `modeles.py` | `QuadZ` (approche actuelle), `FactorizationMachine` (FMQA), `BayesQuadBits` (BOCS, tirage de Thompson), `MLPEnsemble` (réseau de neurones, pas un QUBO) ; encodage indices ↔ bits |
| `boucle.py` | `boucle_ia` (modèle × contrainte × solveur), recuit simulé vectorisé, recherche directe, `verifier_boucle` |
| `experiences_v2_ia.py` | 8 méthodes × S1, S2 × 20 graines sur le problème V2 → `resultats/ia_qubo_V2.json` |
| `precision_modeles.py` | précision et corrélation de rang des modèles selon le nombre de données → `resultats/ia_qubo_precision.json` |
| `qaoa_contraintes.py` | QAOA sur un QUBO à contrainte modélisée ou en boîte noire → `resultats/ia_qubo_qaoa_contraintes.json` |
| `probleme_cst.py` | problème mis à l'échelle : CST à 10 coefficients, XFOIL + SER (optique physique) en boîte noire, cache `resultats/cache_cst.jsonl` |
| `experiences_echelle.py` | 7 méthodes × 5 graines à l'échelle → `resultats/ia_qubo_echelle.json` |

## Lancer

```bash
cd Code
python3 -c "from ia_qubo.boucle import verifier_boucle; verifier_boucle()"   # quelques secondes
python3 -m ia_qubo.experiences_v2_ia          # ~25 min sur 11 processus
python3 -m ia_qubo.experiences_echelle        # ~1 h (XFOIL), réutilise le cache
python3 -m ia_qubo.precision_modeles          # ~15 min ; après experiences_echelle pour la partie CST
python3 -m ia_qubo.qaoa_contraintes           # ~3 min
```

Le réseau de neurones est écrit en NumPy (pas de dépendance supplémentaire).
