# Wrap-up Hackathon_QD — Phase 1 : audit

Préparé le 28/11/2026, pour l'équipe, avant la construction du deck. À valider avant la Phase 2.

Fichiers associés :
- `wrapup/registre.csv` : le registre des affirmations (85 lignes), régénéré par `python wrapup/registre.py`.
- `wrapup/reruns/` : les journaux et sorties des relances.

## 1. Ce qui a été vérifié

| Vérification | Résultat |
|---|---|
| Registre des affirmations, recalculé depuis `Code/resultats/` | 85 lignes : 67 reproduites, 17 recalculées (valeurs nouvelles, sans valeur annoncée), 0 écart, 1 non vérifiable |
| `Fiche_Projet/chiffres_auto.tex` régénéré par `tools/chiffres.py` | 86 macros identiques |
| V3 L1 relancé (`optim.experiences_l1`, 5 méthodes × S1/S2 × 20 graines, 8 min 40) | traces identiques au bit près à `pipeline_L1_V3.json` |
| L6 : audit statistique relancé (`ia_qubo.audit_echelle`, sans XFOIL) | hypervolumes égaux aux publiés ; Wilcoxon et Holm recalculés |
| `validation.py` (panneaux, plaque plane, sensibilité) | reproduit (pente 6,900 /rad ; Thwaites 1,02 % ; Head 4,9 %) |
| MoM 2D TM contre la série de Mie, recalculé | écart max 0,004 dB (ka de 0,5 à 31) |
| Modèle rapide contre XFOIL, recalculé depuis `base_xfoil_a2.csv` | 498/512 convergés, biais CD +27 %, rang 0,91 (CD) et 0,935 (CL) |

Ce qui n'a pas pu être vérifié :
- **Les calculs XFOIL n'ont pas été relancés** : XFOIL n'est pas installé sur cette machine Windows. Leurs sorties CSV ont en revanche été recalculées.
- **Les chiffres V0** (25 simulations contre 256 ; P(optimum) de 0,018 à 0,142) sont écrits en dur dans la fiche, sans JSON derrière.
- **Aucune comparaison à des données NACA2412 publiées** (soufflerie, Abbott & von Doenhoff) n'existe dans le repo.

## 2. Inventaire des itérations

| | Question | Changement | Résultat (vérifié) | Décision |
|---|---|---|---|---|
| **V0** | Un modèle aéro rapide et un QUBO sont-ils possibles ? | Panneaux + couche limite (8 ms/profil), QUBO 9 qubits, boucle sur la seule traînée | Minimum trouvé en 25 simulations contre 256 au hasard *(en dur)* | Construire toute la chaîne de la note |
| **V1** | La chaîne de la note donne-t-elle un compromis, et la boucle bat-elle NSGA-II ? | SER par optique physique, NSGA-II, QUBO 13 qubits, XFOIL en validation | Boucle 35 / 67,5 simulations contre 106,5 / 106 pour NSGA-II (S1 / S2). Mais **compromis dégénéré** : tout le front est à t = 8 %, et la SER n'y varie que de 0,13 dB. Exact, recuit et QAOA font pareil : « le gain vient du modèle de substitution, pas du solveur quantique » | S'écarter de la note, reformuler le problème |
| **V2** | Avec un rayon de bord d'attaque libre, une contrainte de manœuvre, XFOIL et une SER exacte, a-t-on un vrai compromis ? | Indice I de rayon de bord d'attaque, C_L,max ≥ 1,2, XFOIL comme simulateur, SER par méthode des moments, 10 bits / 1 024 profils / 14 qubits | Vrai compromis : bord pointu = −5,0 dB (MoM) mais décroche. Front exact en 75 / 142,5 simulations (S1 / S2) contre 911 / 971,5 au hasard. NSGA-II échoue (5 % de succès). Recuit 97,5 / 235 (55 %), QAOA 105 / 60 % | Clore V2, ouvrir V3 |
| **V3 L1** | Face à l'optimisation bayésienne (BO), la boucle garde-t-elle son avantage ? (H1) | BO, krigeage, EI, ParEGO, et un **contrôle** : la même boucle sans encodage binaire | **H1 non confirmée.** S1 : boucle 75 = contrôle 75 < BO 125 (p < 0,001). S2 : contrôle 95 < BO 127,5 < boucle 142,5. L'encodage QUBO n'apporte rien | L3 (C_L,max) devient prioritaire |
| **V3 L6** | Peut-on apprendre le QUBO (FMQA, BOCS, réseau), et la boucle tient-elle à grande échelle ? | Profil CST à 30 bits (10⁹ profils), XFOIL à la demande, 8 graines, 300 simulations | Les trois QUBO se valent et battent NSGA-II (hypervolume normalisé sur le front : 0,62 contre 0,25). Mais p = 0,008 devient 0,055 après correction de Holm. Le témoin hybride non QUBO n'est pas significativement différent (p = 0,11) | Fait. Concurrent classique manquant sur CST (TuRBO, SAASBO) |
| L2 à L5 | Grande grille 14 bits ; robustesse C_L,max et SER ; QAOA bruité ; interface SER | — | à faire | — |

Le premier commit (06/11) contient déjà V0 à V2. L'historique git ne couvre donc que la V3 (15/11 → 28/11, PR #1 à #6). La chronologie de V0 à V2 repose sur la fiche.

## 3. Thèse centrale, formulée honnêtement

> **Ce qui marche, c'est la boucle « simuler peu → ajuster un modèle quadratique → minimiser ce modèle → simuler ses meilleurs candidats ».** Elle retrouve le front exact en 75 simulations là où le hasard en demande 911 et où NSGA-II échoue. Elle bat l'optimisation bayésienne sur un scénario (S1 : 75 contre 125) mais pas sur l'autre (S2 : 142,5 contre 127,5).
>
> **Ce qui ne marche pas encore, c'est la partie quantique.** Le même modèle sans encodage QUBO fait aussi bien (S1) ou mieux (S2, p < 0,001). Le QUBO de 14 qubits se résout exactement par énumération en 3,7 ms. Sur nos QUBO, QAOA ne fait que 4 à 5 fois mieux que le hasard, parce que les pénalités de contraintes occupent 98 % du spectre d'énergie.
>
> **Nuance à garder :** dans S1, c'est le modèle affine de C_L,max imposé par la formulation QUBO qui laisse passer le front. Les modèles plus précis le prédisent infaisable (PR #2). La forme QUBO a donc un effet, mais par la contrainte, pas par le solveur.

Ce constat est cohérent sur trois itérations : V1, puis V3 L1, puis le cours quantique (ch. 14, « pourquoi QAOA n'apporte rien »).

## 4. Fidélité physique, séparée du gain d'optimisation

- **SER :** la méthode des moments colle à Mie (0,004 dB recalculé). L'optique physique sous-estime la MoM de 1,05 dB en moyenne (rang 0,999), et de 2,5 à 4,3 dB sur les nez pointus. C'est pourquoi le −7,3 dB (PO) devient −5,0 dB en MoM.
- **Aéro :** depuis la V2, XFOIL est le simulateur. Le modèle rapide maison (V0–V1) surestime CD de 27 % mais classe bien (rang 0,91). XFOIL lui-même n'est comparé à aucune donnée expérimentale dans le repo. **C_L,max, la grandeur la moins bien prédite, décide du front**, qui est collé à la contrainte C_L,max ≥ 1,2.
- **Domaine :** 2D seulement (profil, SER 2D TM), incompressible, une fréquence (10 GHz).

## 5. Ce qu'un jury attaquera

1. **« Où est l'avantage quantique ? »** Il n'y en a pas : énumération en 3,7 ms, QAOA simulé, 4 à 5 fois le hasard avec contraintes.
2. **« Et sans QUBO ? »** Le contrôle fait aussi bien ou mieux. C'est la question la plus dangereuse si on ne la devance pas.
3. **« Et sur l'autre scénario ? »** Sur S2, la BO gagne.
4. **« 8 graines, 15 budgets testés ? »** Après Holm, rien n'est significatif à 5 % sur le CST 30 bits.
5. **« Votre front dépend d'une contrainte mal prédite. »** C_L,max ; une pénalité 10 fois plus forte fait tomber la boucle à 25 % de succès sur S2.
6. **« Le choix de normalisation change le tableau. »** L'hypervolume publié donne 0,74 contre 0,68 ; normalisé sur le front, 0,62 contre 0,25.
7. **2D, incompressible, une seule fréquence, pas de validation expérimentale.**
8. **Chiffres V0 sans traçabilité.**

## 6. Corrections à apporter au pitch de Meddi

Version lue : 1795877094-3857 (28/11 ; seul le visuel a changé par rapport à la précédente).

| Slide | Affirmation actuelle | Correction | Source |
|---|---|---|---|
| solution / resultats | « 12× fewer simulations », S1 seul, sans contrôle | Montrer S2 et le contrôle, ou promettre « boucle à modèle quadratique, prête pour le QUBO ». Contre la BO : 1,7× sur S1 ; la boucle perd sur S2 (0,89×) | registre : L1 S1/S2 |
| resultats | Barres NSGA-II (V2) et BO (V3) mélangées | Signaler les deux campagnes (même problème, protocole proche) | pipeline_V2 / pipeline_L1_V3 |
| compromis | « −7.3 dB … exact 2D RCS by method of moments » | −7,3 dB est la valeur en optique physique ; la MoM donne **−5,0 dB** (« divisé par 3 », pas par 5) | registre : SER t = 12 % |
| echelle | « 0.88–0.90 of hypervolume » | Ce sont des valeurs brutes ; en relatif 0,74 contre 0,68 (NSGA-II) et 0,64 (hasard) | ia_qubo_echelle.json |
| echelle | « 8 of 8, p = 0.008 » | Exact, mais c'est le meilleur de 15 budgets : après Holm p = 0,055. Le témoin hybride non QUBO fait jeu égal (p = 0,11) | reruns/ia_qubo_audit_echelle.json |
| position | « First to unite aero and stealth » | Contredit par son propre tableau (le krigeage couvre déjà aéro et radar). Au mieux « premiers avec un QUBO », à vérifier dans la littérature | — |
| position (note orateur) | « 20 runs per method » | 20 en L1, 8 sur la slide echelle | — |
| besoins | « Compute hours: 16,384-airfoil grid » | 16 384 × 0,343 s ≈ 94 min sur un cœur : ce n'est pas un besoin | registre |
| onemore | « QAOA … exact front in 100 % of runs, ready for the hardware » | Vrai sur V2 S1 avec 5 graines, mais en 105 simulations contre 75 en exact, et 60 % de succès sur S2. Le cours conclut que QAOA n'apporte rien à la boucle | pipeline_V2.json ; cours 4 ch14 ; PR #4 |

## 7. Plan du deck (Phase 2)

Les titres sont des conclusions. Visuels générés par `wrapup/build.py` depuis les données brutes.

| # | Titre (conclusion) | Visuel | Source |
|---|---|---|---|
| 1 | Traînée contre furtivité : ce que la boucle QUBO apporte vraiment | Profil, logo ISAE-SUPAERO, noms | — |
| 2 | Un profil de drone doit échapper au radar et bien voler, et chaque simulation coûte | Bord rond / bord pointu ; 0,34 s par XFOIL, 10⁹ profils ≈ 11 ans | anatomie_qubo_V3, base_V2 |
| 3 | La note d'intention misait sur un QUBO pour trouver le front avec peu de simulations | Schéma de la chaîne : géométrie → aéro → SER → modèle → QUBO → solveur | chaine_code.tex |
| 4 | Quatre itérations, chacune née de la limite de la précédente | Frise V0 → V3 (fiche + git) | fiche, git log |
| 5 | V0 : un QUBO de 9 qubits trouve le minimum de traînée en 25 simulations | Barre 25 contre 256 *(non vérifié)* | fiche (en dur) |
| 6 | V1 : la boucle bat NSGA-II, mais le compromis de la note est dégénéré | Fronts V1 ; t = 8 % partout, 0,13 dB | pipeline.json |
| 7 | V2 : un vrai compromis ; front exact en 75 simulations, NSGA-II échoue | Front V2 avec formes ; barres 911 / 75 / NSGA 5 % | pipeline_V2.json, base_V2 |
| 8 | V3 : face à la BO, la boucle gagne sur S1 et perd sur S2 | Boîtes à moustaches S1 / S2, 20 graines, toutes méthodes | pipeline_L1_V3.json |
| 9 | Le gain vient du modèle quadratique, pas des qubits | Boucle contre contrôle, S1 et S2, avec p | pipeline_L1_V3.json |
| 10 | Pourquoi QAOA n'aide pas : les pénalités prennent 98 % du spectre | Spectre d'énergie ; p_opt QAOA contre hasard, avec et sans contraintes | anatomie_qubo_V3.json |
| 11 | À 10⁹ profils, la boucle devance NSGA-II, mais 8 graines ne suffisent pas à le prouver | Hypervolume relatif en fonction des simulations, témoin hybride inclus ; p brut et Holm | ia_qubo_echelle + audit |
| 12 | Les solveurs sont validés ; la contrainte de manœuvre reste le maillon faible | MoM contre Mie ; PO contre MoM ; modèle rapide contre XFOIL | ser/mom2d, base_*_a2 |
| 13 | Où on en est : H1 non confirmée, L1 et L6 faits, L2 à L5 à faire | Tableau des lots | fiche |
| 14 | Limites : 2D, une fréquence, C_L,max mal prédit, peu de graines | Liste courte | §5 |
| 15 | Prochaines étapes, par impact / effort | Tableau : L3 C_L,max ; isoler la cause S2 du contrôle ; 20 graines CST + Holm ; TuRBO / SAASBO ; L2 ; L4 ; SER 3D | fiche, cours |

Annexes : registre des affirmations ; relancer le pipeline en 5 commandes ; carte du code ; glossaire ; corrections du pitch (§6).
