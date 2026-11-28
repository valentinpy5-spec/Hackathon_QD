"""Itération V3, lot L6 : les modèles appris sur le problème V2 (1 024 profils, front exact connu).

Même protocole que optim/experiences_v2.py et optim/experiences_l1.py : table XFOIL + SER par méthode des
moments, CountedProblem, 10 profils initiaux (les mêmes pour une graine donnée), 5 poids, budget 250,
critère « front exact retrouvé » et arrêt réaliste (stagnation de l'hypervolume sur 3 itérations).

Méthodes : <modele>_<contrainte>, modele dans {quadz, fm, bocs, mlp}, contrainte dans {modele, bn}
(bn = boîte noire), et hybride_modele (réseaux pour C_D et la SER, modèle affine pour C_L,max). quadz_modele est la boucle QUBO de la V2, à l'identique.

    python3 -m ia_qubo.experiences_v2_ia            # 20 graines -> resultats/ia_qubo_V2.json (~30 min)
    python3 -m ia_qubo.experiences_v2_ia rapide     # 3 graines  -> resultats/ia_qubo_V2_rapide.json
    python3 -m ia_qubo.experiences_v2_ia hybride_modele   # une seule méthode, fusionnée dans le JSON
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.problem import CountedProblem  # noqa: E402
from optim.problem_v2 import SpaceV2, KEYS  # noqa: E402
from optim.experiences_v2 import setup, state, summary  # noqa: E402
from ia_qubo.boucle import boucle_ia  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = SpaceV2()
SPEC = dict(bits=SP.bits, levels=SP.n_levels, keys=KEYS, t_var=2, mlp_epochs=1000, mlp_models=4)
METHODS = [f"{m}_{c}" for m in ("quadz", "fm", "bocs", "mlp") for c in ("modele", "bn")] + ["hybride_modele"]


def _run_one(args):
    method, sector, seed, budget = args
    T, feas, F, fr, exact, norm, ref, hv = setup(sector, "MoM")
    pb = CountedProblem(T, keys=KEYS)
    tr = []

    def cb(p):
        tr.append((p.n_sims, *state(p, exact, norm, ref, hv)))
        return tr[-1][1] >= 1.0 and len(tr) > 3 and tr[-1][2] <= tr[-4][2] + 1e-12
    modele, contrainte = method.split("_")
    t0 = time.time()
    boucle_ia(pb, SPEC, modele=modele, contrainte="modele" if contrainte == "modele" else "boite_noire",
              solveur="exact", seed=seed, budget=budget, stop_when=cb)
    return tr, time.time() - t0


if __name__ == "__main__":
    from multiprocessing import Pool
    quick = "rapide" in sys.argv
    names = [m for m in METHODS if m in sys.argv] or METHODS
    n_seeds, budget = (3 if quick else 20), 250
    jobs = [(m, sec, s, budget) for sec in ("S1", "S2") for m in names for s in range(n_seeds)]
    # les méthodes lentes (réseau de neurones) d'abord, pour équilibrer les processus
    jobs.sort(key=lambda j: 0 if j[0].startswith(("mlp", "hybride")) else 1)
    workers = max(1, min(12, (os.cpu_count() or 2) - 1))
    t0 = time.time()
    with Pool(workers) as pool:
        out = pool.map(_run_one, jobs, chunksize=1)
    res = {}
    for sec in ("S1", "S2"):
        T, feas, F, fr, exact, norm, ref, hv = setup(sec, "MoM")
        R = dict(n_feasible=len(feas), n_front=len(fr), n_seeds=n_seeds, budget=budget)
        for m in names:
            R[m] = [o[0] for j, o in zip(jobs, out) if j[0] == m and j[1] == sec]
            R[m + "_seconds"] = [o[1] for j, o in zip(jobs, out) if j[0] == m and j[1] == sec]
        R["summary"] = {m: summary(R[m], 3) for m in names}
        res[sec] = R
        print(f"{sec} : front exact {len(fr)} profils", flush=True)
        for m in names:
            v = R["summary"][m]
            print(f"   {m:13s} front complet {v['full_front']}, HV99 {v['hv99']}, succès {100 * v['success']:.0f} %, "
                  f"arrêt réaliste {v['stop_sims']} ; {sum(R[m + '_seconds']) / n_seeds:.0f} s par graine", flush=True)
    print(f"durée totale {time.time() - t0:.0f} s sur {workers} processus")
    path = os.path.join(HERE, "resultats", "ia_qubo_V2_rapide.json" if quick else "ia_qubo_V2.json")
    if os.path.exists(path) and set(names) != set(METHODS):
        # campagne partielle (ex. : python3 -m ia_qubo.experiences_v2_ia hybride_modele) : on fusionne
        old = json.load(open(path))
        for sec in res:
            for k, v in res[sec].items():
                if k == "summary":
                    old[sec].setdefault("summary", {}).update(v)
                elif k in names or k[:-len("_seconds")] in names:
                    old[sec][k] = v
        res = old
    with open(path, "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("->", path)
