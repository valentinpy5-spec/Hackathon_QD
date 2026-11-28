"""Itération V3, lot L7 : les formulations de contrainte sur le problème CST (30 bits), avec XFOIL.

Même protocole que la seconde campagne du lot L6 (ia_qubo/experiences_echelle.py) : filtre géométrique
avant simulation, 300 simulations, 8 graines, mêmes profils initiaux par graine (comparaison appariée).
Les méthodes de la seconde campagne ne sont PAS relancées : leurs exécutions sont relues dans
resultats/ia_qubo_echelle_brut.json, et le front de référence est recalculé sur l'union des deux.

Nouvelles méthodes (modèle en z, recuit simulé, 30 qubits au lieu de 33) :
    quadz_lagrange       C_L,max et faces sans croisement en termes lagrangiens, lectures vérifiées
    quadz_lagrange_clf   idem + classifieur de convergence d'XFOIL (P(succès) prédite >= 0,1)
    quadz_desequilibree  pénalité déséquilibrée sur C_L,max

Durée : ~1 h 10 sur 11 processus pour les deux méthodes lagrangiennes (XFOIL). PC sur secteur, sans veille.

    python3 -m contraintes.experiences_cst      -> resultats/contraintes_cst_campagne.json (+ _brut.json)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ia_qubo.boucle import boucle_ia  # noqa: E402
from ia_qubo import probleme_cst as P  # noqa: E402
from ia_qubo.experiences_echelle import post_traitement, BUDGET, N_SEEDS  # noqa: E402
from contraintes.experiences_v2 import wilcoxon_signe  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "contraintes_cst_campagne.json")
ANCIEN = os.path.join(HERE, "resultats", "ia_qubo_echelle_brut.json")
METHODS = ["quadz_lagrange", "quadz_lagrange_clf"]   # quadz_desequilibree : écartée (moins bonne hors ligne)
P_MIN = 0.1


def _run_one(args):
    method, seed = args
    pb = P.ProblemeSimule()
    t0 = time.time()
    modele, contrainte = method.split("_", 1)
    boucle_ia(pb, dict(P.SPEC, p_min=P_MIN), modele=modele, contrainte=contrainte, solveur="recuit",
              seed=seed, budget=BUDGET)
    return [list(i) for i in pb.seen[:BUDGET]], time.time() - t0


def raisons(cache, seen):
    rows = [cache[tuple(s)] for s in seen]
    n = len(rows)
    return dict(faisable=sum(r["feasible"] for r in rows) / n,
                echec_xfoil=sum(not r["valid"] for r in rows) / n,
                cl_faible=sum(r["valid"] and not r["feasible"] for r in rows) / n)


def bilan(runs, res):
    cache = P._charge_cache()
    ref = "quadz_modele"
    grille = res[ref]["grille"]
    out = {}
    for m in runs:
        tr = {}
        for b in (100, 200, 300):
            k = grille.index(b)
            a = [x[k] for x in _traces(res, m)]
            r0 = [x[k] for x in _traces(res, ref)]
            tr[str(b)] = dict(mediane=float(np.median(a)), mieux=int(sum(x > y for x, y in zip(a, r0))),
                              p=wilcoxon_signe(a, r0) if m != ref else float("nan"))
        rz = [raisons(cache, seen) for seen, _ in runs[m]]
        ext, ech6, ech = [], [], []
        for seen, _ in runs[m]:
            I = np.array(seen[10:])
            e = ((I == 0) | (I == 7)).sum(1)
            f = np.array([not cache[tuple(x)]["valid"] for x in seen[10:]])
            ext.append(float(((I == 0) | (I == 7)).mean()))
            if (e >= 6).any():
                ech6.append(float(f[e >= 6].mean()))
            ech.append(float(f[e < 6].mean()))
        out[m] = dict(hv=tr, raisons={k: float(np.median([r[k] for r in rz])) for k in rz[0]},
                      niveaux_extremes=float(np.median(ext)),
                      echec_si_6_extremes=float(np.median(ech6)) if ech6 else None,
                      echec_sinon=float(np.median(ech)),
                      minutes=float(np.mean([t for _, t in runs[m]]) / 60))
    return out


def _traces(res, m):
    return res[m]["traces"]


if __name__ == "__main__":
    from multiprocessing import Pool
    names = [m for m in METHODS if m in sys.argv] or METHODS
    brut = OUT.replace(".json", "_brut.json")
    nouveaux = json.load(open(brut)) if os.path.exists(brut) else {}
    jobs = [(m, s) for m in names for s in range(N_SEEDS)]
    if "bilan" not in sys.argv:
        workers = max(1, min(11, (os.cpu_count() or 2) - 1))
        t0 = time.time()
        with Pool(workers) as pool:
            out = pool.map(_run_one, jobs, chunksize=1)
        for m in names:
            nouveaux[m] = [o for j, o in zip(jobs, out) if j[0] == m]
        with open(brut, "w") as f:
            json.dump(nouveaux, f)
        print(f"durée {(time.time() - t0) / 60:.0f} min", flush=True)
    runs = dict(json.load(open(ANCIEN)))
    runs.update(nouveaux)
    res = post_traitement(runs)
    cache = P._charge_cache()
    from optim.pareto import pareto_front, hypervolume_2d, Normalizer
    vus = {tuple(s) for lst in runs.values() for seen, _ in lst for s in seen}
    feas = {k: r for k, r in cache.items() if r["feasible"] and k in vus}
    F = np.array([[r["CD_counts"], r["SER_db"]] for r in feas.values()])
    norm, refp = Normalizer(F), np.array([1.1, 1.1])
    for m, lst in runs.items():
        T = []
        for seen, _ in lst:
            tr = []
            for k in res[m]["grille"]:
                pts = [cache[tuple(s)] for s in seen[:k] if cache[tuple(s)]["feasible"]]
                Fk = np.array([[r["CD_counts"], r["SER_db"]] for r in pts]) if pts else None
                tr.append(hypervolume_2d(norm(Fk[pareto_front(Fk)]), refp) / res["hv_ref"] if pts else 0.0)
            T.append(tr)
        res[m]["traces"] = T
    res["bilan"] = bilan(runs, res)
    res["p_min"] = P_MIN
    with open(OUT, "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"front de référence (union des deux campagnes) : {len(res['front_ref'])} profils, "
          f"{res['n_simules_total']} profils simulés")
    for m, b in res["bilan"].items():
        print(f"   {m:20s} HV à 100/200/300 : " + " ".join(
            f"{b['hv'][k]['mediane']:.3f} ({b['hv'][k]['mieux']}/8, p={b['hv'][k]['p']:.3f})" for k in ("100", "200", "300"))
            + f" ; faisables {100 * b['raisons']['faisable']:.0f} %, échecs XFOIL {100 * b['raisons']['echec_xfoil']:.0f} %, "
              f"C_L faible {100 * b['raisons']['cl_faible']:.0f} % ; niveaux extrêmes {100 * b['niveaux_extremes']:.0f} % ; "
              f"{b['minutes']:.0f} min", flush=True)
    print("->", OUT)
