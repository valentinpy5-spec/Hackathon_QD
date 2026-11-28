"""Itération V3, lot L6 : pourquoi le modèle le plus précis (réseau de neurones) trouve-t-il moins bien le front ?

Hypothèse : les profils manqués sont des points NON SUPPORTÉS du front (aucune somme pondérée des vrais
objectifs ne les désigne). Un modèle précis ne les désigne donc jamais non plus ; l'erreur d'un modèle
grossier, elle, les fait atteindre « par accident ».

On rejoue 6 graines par méthode (quadz, fm, bocs, mlp ; contrainte modélisée) sur S1 et S2 en notant les
profils simulés, et l'on calcule pour chaque profil du front : supporté ou non, et la fréquence à laquelle
chaque méthode le trouve dans le budget.

    python3 -m ia_qubo.diagnostic_v2_ia        -> resultats/ia_qubo_diagnostic_V2.json  (~10 min)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.problem import CountedProblem  # noqa: E402
from optim.problem_v2 import SpaceV2, KEYS  # noqa: E402
from optim.experiences_v2 import setup  # noqa: E402
from ia_qubo.boucle import boucle_ia  # noqa: E402
from ia_qubo.experiences_v2_ia import SPEC  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "ia_qubo_diagnostic_V2.json")
MODELES = ("quadz", "fm", "bocs", "mlp")
SEEDS = range(6)


def _run(args):
    modele, sector, seed = args
    T = setup(sector, "MoM")[0]
    pb = CountedProblem(T, keys=KEYS)
    boucle_ia(pb, SPEC, modele=modele, contrainte="modele", solveur="exact", seed=seed, budget=250)
    return [list(i) for i in pb.seen]


def supportes(F, norm):
    Fn = norm(F)
    sup = set()
    for w in np.linspace(0, 1, 2001):
        sup.add(int(np.argmin(w * Fn[:, 0] + (1 - w) * Fn[:, 1])))
    return sup


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(m, s, g) for s in ("S1", "S2") for m in MODELES for g in SEEDS]
    with Pool(max(1, min(11, (os.cpu_count() or 2) - 1))) as pool:
        out = pool.map(_run, jobs, chunksize=1)
    res = {}
    for sec in ("S1", "S2"):
        T, feas, F, fr, exact, norm, ref, hv = setup(sec, "MoM")
        Ff = F[fr]
        idx = [tuple(feas[i][k] for k in KEYS) for i in fr]
        order = np.argsort(Ff[:, 0])
        sup = supportes(Ff, norm)
        pts = []
        for j in order:
            row = feas[fr[j]]
            d = dict(idx=list(idx[j]), CD=float(Ff[j, 0]), SER=float(Ff[j, 1]), CLmax=float(row["CLmax"]),
                     supporte=j in sup)
            for m in MODELES:
                seen = [set(map(tuple, o)) for jb, o in zip(jobs, out) if jb[0] == m and jb[1] == sec]
                d[m] = float(np.mean([idx[j] in s for s in seen]))
            pts.append(d)
        res[sec] = pts
        print(sec)
        for d in pts:
            print(f"  {d['idx']} C_D {d['CD']:6.1f} SER {d['SER']:7.2f} C_Lmax {d['CLmax']:.2f} "
                  f"{'supporté    ' if d['supporte'] else 'non supporté'} "
                  + " ".join(f"{m} {d[m]:.2f}" for m in MODELES))
    with open(OUT, "w") as f:
        json.dump(res, f)
    print("->", OUT)
