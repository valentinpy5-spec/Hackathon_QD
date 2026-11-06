# Méthode de travail

Le projet avance par **itérations**. Une itération pose une question, change ce qu'il faut
pour y répondre, mesure, puis se clôt par une **décision** qui ouvre la suivante. Rien
n'est réécrit en silence : ce qu'une itération a produit reste lisible et relançable.

## 1. Le cycle d'une itération

| Étape | Ce qu'on fait | Où |
|---|---|---|
| Ouvrir | question, hypothèses, lots de travail, critère de clôture | chapitre de l'itération dans la fiche projet |
| Travailler | une branche par lot, du code testé, des résultats dans `Code/resultats/` | `Code/` |
| Documenter | tableau des changements (statut, résultat) ; chapitres ou encadrés « Évolution » dans les cours | `Fiche_Projet/`, `Cours/` |
| Clore | résultats, enseignements, décision ; code et résultats figés ; étiquette git | `Code/archives/`, `git tag` |

L'itération en cours est la **V3** : sa visée et ses lots (L1 à L5) sont au dernier chapitre de
`Fiche_Projet/Fiche_Projet_Iterations.pdf`.

## 2. Git

- `main` contient toujours un état qui tourne (`python3 Code/verifier_installation.py` passe).
- **Une branche par lot** : `v3/l1-bayes`, `v3/l2-grille`, `v3/l3-physique`, `v3/l4-bruit`, `v3/l5-interface-ser`.
- **Commits préfixés par l'itération** : `[V3] L1 : processus gaussien par objectif`.
- **Fusion dans `main` par demande de fusion** (pull request), relue par un autre membre.
- On ne versionne pas : `.venv/`, `.tools/` (recréés par `Code/installer_outils.sh`), les journaux
  `*.log`, les fichiers intermédiaires de LaTeX. On versionne les PDF des cours et de la fiche :
  ce sont les livrables.
- À la clôture : `git tag -a V3 -m "V3 close : ..."` puis `git push --tags`.

Le dépôt part du commit « First Commit », qui correspond à la **V2 close** (étiquette `V2`).
L'historique antérieur n'y figure pas : V0, V1 et V2 sont figées dans `Code/archives/` et
racontées dans la fiche projet.

## 3. Code

- Un fichier qui **remplace** un fichier existant prend le suffixe de l'itération
  (`optim/problem_v3.py`) : l'ancien reste relançable tant que les cours en dépendent.
- Un résultat nouveau s'appelle `Code/resultats/<nom>_V3.<ext>`.
- Le docstring de chaque script dit : l'itération, ce qui change, la commande à lancer, le fichier produit.
- Toute comparaison de méthodes passe par `CountedProblem` (`optim/problem.py`) : même table,
  même compteur de simulations, mêmes graines.
- Un module devenu inutile est retiré de `Code/` et noté dans le tableau « Remplacements »
  de `Code/archives/README.md` (il reste dans l'archive de l'itération précédente).

## 4. Cours et fiche projet

- Les couleurs des itérations sont définies une seule fois dans `Fiche_Projet/iterations.tex`
  (V0 brun, V1 framboise, V2 cyan, **V3 ambre**, V4 indigo réservé), et utilisées partout.
- Nouveau chapitre : `\iterbanner{V3}` sous le titre.
- Ajout ou correction dans un chapitre existant : encadré
  `\begin{evolution}{V3}{titre} ... \end{evolution}` ; on ne modifie pas le texte d'origine.
- Les figures à données sont produites par `Cours/<cours>/tools/make_figures*.py`, qui relisent
  `Code/resultats/` ; les nombres de la fiche par `Fiche_Projet/tools/chiffres.py`.
- Chaque chapitre finit par des exercices (corrigés en annexe) et un encadré `pistes`.
- Compilation : `tectonic <fichier>.tex` (ou `xelatex`, deux passes). Recompiler le PDF avant de committer.

## 5. Clore une itération (liste de contrôle)

1. Résultats, enseignements et décision écrits dans la fiche ; `python3 tools/chiffres.py` relancé.
2. Code et résultats copiés dans `Code/archives/Vk_<nom>/` (sans `__pycache__` ni `*.log`) ;
   ligne ajoutée dans `Code/archives/README.md`.
3. Cours et fiche recompilés.
4. Itération suivante déclarée dans `Fiche_Projet/iterations.tex` (nom, date, légende).
5. Commit `[Vk] Clôture`, étiquette `Vk`, push.
