# Hackathon quantique : compromis traînée / SER d'un profil de drone

Optimisation bi-objectif d'un profil d'aile : **traînée aérodynamique** contre
**surface équivalente radar (SER)**, sous contraintes. On compare une méthode classique
(NSGA-II) et une boucle quantique-classique (modèle quadratique → QUBO → recuit simulé
ou QAOA) sur le **nombre de simulations** nécessaires pour retrouver le front de Pareto exact.

Le projet avance par **itérations** (V0, V1, V2, …). Chacune a une couleur, utilisée dans
la fiche projet et dans tous les cours.

| Itération | Question | Statut |
|---|---|---|
| **V0** · modèle de la note d'intention | Peut-on construire un modèle aéro rapide et un QUBO à partir de lui ? | close |
| **V1** · chaîne complète conforme à la note | La chaîne donne-t-elle un compromis exploitable, et la boucle quantique bat-elle NSGA-II ? | close |
| **V2** · problème reformulé | Avec un rayon de bord d'attaque libre, une contrainte de manœuvre, XFOIL et une SER exacte, a-t-on un vrai compromis, et que vaut la boucle ? | close (6 nov. 2026) |
| **V3** · concurrent équitable, problème plus grand | Face à l'optimisation bayésienne et sur 16 384 profils, que reste-t-il de l'avantage de la boucle QUBO ? | **ouverte** |

## Par où commencer

1. **La fiche projet** : [`Fiche_Projet/Fiche_Projet_Iterations.pdf`](Fiche_Projet/Fiche_Projet_Iterations.pdf).
   Pour chaque itération : objectif, changements, résultats, enseignements, décision.
   Le dernier chapitre fixe la **visée de la V3** et ses lots de travail.
2. **Les cours**, un PDF par discipline, à lire selon son rôle dans l'équipe :

   | Cours | Contenu |
   |---|---|
   | [`Cours/1_Aerodynamique/`](Cours/1_Aerodynamique/Cours_Aerodynamique.pdf) | profils NACA, panneaux de Hess-Smith, couche limite intégrale, Squire-Young, validation et usage d'XFOIL |
   | [`Cours/2_SER/`](Cours/2_SER/Cours_SER.pdf) | SER 2D par optique physique, puis exacte par méthode des moments |
   | [`Cours/3_Optimisation_BiObjectif/`](Cours/3_Optimisation_BiObjectif/Cours_Optimisation_BiObjectif.pdf) | front de Pareto exact, NSGA-II, boucle QUBO bi-objectif, comparaisons |
   | [`Cours/4_Calcul_Quantique/`](Cours/4_Calcul_Quantique/Cours_Calcul_Quantique.pdf) | calcul quantique depuis zéro : qubit, circuits, QUBO, recuit, QAOA, montée en taille |

   Chaque chapitre porte le bandeau de l'itération qui l'a écrit ; un ajout ultérieur
   apparaît dans un encadré « Évolution » de la couleur de l'itération. Exercices corrigés
   et pistes d'amélioration à la fin de chaque chapitre.
3. **Le code** : [`Code/`](Code/README.md). Les itérations closes sont figées dans
   [`Code/archives/`](Code/archives/README.md).
4. **Le sujet** : [`Documents/Note_intention_projet.pdf`](Documents/Note_intention_projet.pdf),
   la note d'intention qui a servi à la sélection (la V2 s'en écarte volontairement).

## Arborescence

```
README.md                  ce fichier
METHODE_DE_TRAVAIL.md      itérations, git, cours : comment contribuer
Fiche_Projet/              journal des itérations (PDF + sources LaTeX)
Cours/                     quatre cours (PDF + sources LaTeX + scripts de figures)
Code/                      code actif (dernière itération close : V2)
  aero/  ser/  optim/  quantique/   modules
  resultats/               tables et résultats de toutes les itérations
  archives/                V0, V1, V2 figées
Documents/                 note d'intention du projet
```

## Installation rapide

```bash
cd Code
pip install -r requirements.txt   # NumPy, SciPy, Matplotlib, pymoo
./installer_outils.sh             # Debian/Ubuntu : XFOIL et Xvfb locaux, sans droits admin
python3 verifier_installation.py  # XFOIL, SER, boucle QUBO sur un exemple (quelques secondes)
```
Compiler un cours ou la fiche : `tectonic <fichier>.tex` (ou `xelatex`, deux passes).
Détails dans [`Code/README.md`](Code/README.md).

## Contribuer

Lire [`METHODE_DE_TRAVAIL.md`](METHODE_DE_TRAVAIL.md) : une branche par lot de travail,
commits préfixés par l'itération (`[V3] ...`), fusion par demande de fusion relue.
