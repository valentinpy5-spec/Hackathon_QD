# Wrap-up du projet (V0 → V3)

Synthèse vérifiée du projet, pour toute l'équipe : un deck de 21 slides et tout ce qui permet d'en vérifier chaque chiffre.

**Le deck** : https://claude.ai/artifact/FaiZid1FUwDifQKqmTjdfn (Claude design, exportable en .pptx ou PDF). S'il ne s'ouvre pas, demandez à la personne qui l'a créé de vous le partager.

## La règle

Chaque chiffre du deck est recalculé depuis `Code/resultats/` et figure dans `registre.csv`. Un chiffre qui n'y est pas n'a rien à faire dans une présentation, pitch compris.

## Contenu

| Fichier | Rôle |
|---|---|
| `phase1_audit.md` | l'audit : inventaire V0–V3, thèse centrale, fidélité physique, faiblesses face à un jury, corrections du pitch |
| `registre.py` → `registre.csv` | recalcule chaque affirmation et lui donne un statut : reproduit, recalculé, écart, non vérifiable |
| `build.py` → `figures/` | régénère toutes les figures du deck depuis les données brutes |
| `reruns/` | journaux des relances : V3 L1 (traces identiques au JSON publié), audit statistique L6, `validation.py` |

État au 28/11 : 85 affirmations, 67 reproduites, 17 recalculées, 0 écart, 1 non vérifiable (XFOIL n'a pas été relancé ; ses sorties en cache sont recalculées).

## Commandes

Depuis la racine du repo, sans rien écrire dans `Code/resultats/` :

```bash
python wrapup/registre.py   # registre.csv + liste des écarts
python wrapup/build.py      # figures/ (environ 15 s)
```

`registre.py` lit `wrapup/reruns/ia_qubo_audit_echelle.json` pour les tests statistiques de L6. Pour le régénérer : `cd Code && python -m ia_qubo.audit_echelle`, **dans une copie de `Code/`**, car le script écrit dans `Code/resultats/`. Même précaution pour `python -m optim.experiences_l1`.

## Ce qu'il faut retenir

- La boucle « modèle quadratique → minimisation → simulation » retrouve le front exact en 75 simulations (S1) au lieu de 911 au hasard ; NSGA-II échoue.
- Face à l'optimisation bayésienne, elle gagne sur S1 (75 contre 125) et perd sur S2 (142,5 contre 127,5).
- Le gain vient du modèle quadratique, pas des qubits : la même boucle sans encodage QUBO fait aussi bien (S1) ou mieux (S2, 95).
- QAOA n'aide pas à cette taille : les pénalités occupent 98 % du spectre d'énergie, et QAOA ne fait que 4 à 5 fois mieux que le hasard.
- À 10⁹ profils (L6), la boucle devance NSGA-II, mais avec 8 graines ce n'est plus significatif après correction de Holm.
