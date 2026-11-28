"""Itération V3, lot L7 : les formulations de contrainte sur le problème V2 (1 024 profils, front exact connu).

Même protocole que ia_qubo/experiences_v2_ia.py (table XFOIL + SER par méthode des moments, 10 profils
initiaux identiques par graine, 5 poids, budget 250, arrêt réaliste), sans aucun appel à XFOIL.
Méthodes <modele>_<contrainte> :
    modele         bits d'écart (V2), la référence
    lagrange       terme lagrangien sur C_L,max, lectures vérifiées, pas de bit d'écart
    lagrange_clf   idem + classifieur de convergence d'XFOIL (contrainte quadratique)
    desequilibree  pénalité déséquilibrée sur C_L,max, pas de bit d'écart
On ajoute la part de profils simulés faisables, le budget « utile ».

    python3 -m contraintes.experiences_v2          # 20 graines -> resultats/contraintes_V2.json (~5 min)
    python3 -m contraintes.experiences_v2 seuils   # seuil du classifieur (0,05 à 0,5) et diagnostic du
                                                   # front -> resultats/contraintes_V2_seuils.json (~3 min)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.problem import CountedProblem  # noqa: E402
from optim.problem_v2 import SpaceV2, KEYS  # noqa: E402
from optim.experiences_v2 import setup, state, summary  # noqa: E402
from ia_qubo.boucle import boucle_ia  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "contraintes_V2.json")
SP = SpaceV2()
SPEC = dict(bits=SP.bits, levels=SP.n_levels, keys=KEYS, t_var=2)
BUDGET_ECHEC = 251
METHODS = ["quadz_modele", "quadz_lagrange", "quadz_lagrange_clf", "quadz_desequilibree",
           "fm_modele", "fm_lagrange_clf"]


def _run_one(args):
    method, sector, seed, budget = args
    T, feas, F, fr, exact, norm, ref, hv = setup(sector, "MoM")
    pb = CountedProblem(T, keys=KEYS)
    tr = []

    def cb(p):
        tr.append((p.n_sims, *state(p, exact, norm, ref, hv)))
        return tr[-1][1] >= 1.0 and len(tr) > 3 and tr[-1][2] <= tr[-4][2] + 1e-12
    modele, contrainte = method.split("_", 1)
    t0 = time.time()
    boucle_ia(pb, SPEC, modele=modele, contrainte=contrainte, solveur="exact", seed=seed, budget=budget,
              stop_when=cb)
    rows = [pb.table[i] for i in pb.seen]
    raisons = dict(faisable=sum(r["feasible"] for r in rows),
                   echec_xfoil=sum((not r["valid"]) for r in rows),
                   cl_faible=sum(r["valid"] and not r["feasible"] for r in rows))
    return tr, time.time() - t0, raisons, pb.n_sims


def wilcoxon_signe(a, b):
    """Test de Wilcoxon apparié bilatéral exact (petits effectifs), a et b de même longueur."""
    from itertools import product
    a = np.where(np.isfinite(a), a, BUDGET_ECHEC)        # front jamais complet : compté budget + 1
    b = np.where(np.isfinite(b), b, BUDGET_ECHEC)
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    # rangs moyens en cas d'égalité
    for v in np.unique(np.abs(d)):
        m = np.abs(d) == v
        r[m] = r[m].mean()
    W = r[d > 0].sum()
    if n <= 16:
        tot = np.array([sum(rr for rr, s in zip(r, sg) if s) for sg in product((0, 1), repeat=n)])
    else:
        rng = np.random.default_rng(0)
        tot = (rng.integers(0, 2, (200000, n)) * r).sum(1)
    mu = r.sum() / 2
    return float(np.mean(np.abs(tot - mu) >= abs(W - mu) - 1e-9))


def premier(tr, k, thr):
    for row in tr:
        if row[k] >= thr - 1e-12:
            return row[0]
    return np.inf


def _run_seuil(args):
    pm, sec, s = args
    SPEC["p_min"] = pm
    return _run_one(("quadz_lagrange_clf", sec, s, 250))


def seuils():
    """Balayage du seuil p_min du classifieur, et probabilité de convergence prédite sur le front exact."""
    from multiprocessing import Pool
    from optim.problem_v2 import build_table_v2, with_objective  # noqa: F401
    from contraintes.classifieur import LogistiqueQuadZ, auc
    S = (0.05, 0.1, 0.2, 0.3, 0.5)
    jobs = [(pm, sec, s) for pm in S for sec in ("S1", "S2") for s in range(20)]
    with Pool(max(1, min(12, (os.cpu_count() or 2) - 1))) as pool:
        out = pool.map(_run_seuil, jobs, chunksize=1)
    res = dict(seuils=list(S))
    ref = json.load(open(OUT))
    for sec in ("S1", "S2"):
        R = {}
        r0 = [premier(t, 1, 1.0) for t in ref[sec]["quadz_modele"]]
        rl = [premier(t, 1, 1.0) for t in ref[sec]["quadz_lagrange"]]
        for pm in S:
            sel = [o for j, o in zip(jobs, out) if j[0] == pm and j[1] == sec]
            ff = [premier(o[0], 1, 1.0) for o in sel]
            hv = [premier(o[0], 2, 0.99) for o in sel]
            fin = lambda v: np.where(np.isfinite(v), v, BUDGET_ECHEC)  # noqa: E731
            R[str(pm)] = dict(front_median=float(np.median(fin(ff))), succes=float(np.mean(np.isfinite(ff))),
                              hv99_median=float(np.median(fin(hv))),
                              part_faisable=float(np.median([o[2]["faisable"] / o[3] for o in sel])),
                              p_vs_modele=wilcoxon_signe(ff, r0), p_vs_lagrange=wilcoxon_signe(ff, rl),
                              mieux_vs_modele=int(sum(fin(ff) < fin(r0))), pire_vs_modele=int(sum(fin(ff) > fin(r0))))
            print(sec, pm, R[str(pm)], flush=True)
        # diagnostic : classifieur appris sur toute la table, probabilité sur le front exact
        T, feas, F, fr, exact, norm, refp, hv_ = setup(sec, "MoM")
        I = np.array([[r[k] for k in KEYS] for r in T])
        v = np.array([r["valid"] for r in T])
        clf = LogistiqueQuadZ(SP.n_levels).fit(I, v)
        tab = {tuple(r[k] for k in KEYS): r for r in T}
        front = []
        for x in sorted(exact):
            nb = []
            for i in range(4):
                for d in (-1, 1):
                    y = list(x)
                    y[i] += d
                    if tuple(y) in tab:
                        nb.append(bool(tab[tuple(y)]["valid"]))
            front.append(dict(profil=list(x), proba=float(clf.proba(np.array([x]))[0]),
                              voisins=len(nb), voisins_echec=nb.count(False)))
        R["diagnostic"] = dict(n_echecs=int((~v).sum()), n_profils=len(T), auc_apprentissage=auc(clf.proba(I), v),
                               front=front)
        res[sec] = R
    with open(OUT.replace(".json", "_seuils.json"), "w") as f:
        json.dump(res, f)
    print("->", OUT.replace(".json", "_seuils.json"))


if __name__ == "__main__" and "seuils" in sys.argv:
    seuils()
elif __name__ == "__main__":
    from multiprocessing import Pool
    names = [m for m in METHODS if m in sys.argv] or METHODS
    n_seeds, budget = 20, 250
    jobs = [(m, sec, s, budget) for sec in ("S1", "S2") for m in names for s in range(n_seeds)]
    workers = max(1, min(12, (os.cpu_count() or 2) - 1))
    t0 = time.time()
    with Pool(workers) as pool:
        out = pool.map(_run_one, jobs, chunksize=1)
    res = {}
    for sec in ("S1", "S2"):
        R = dict(n_seeds=n_seeds, budget=budget)
        for m in names:
            sel = [o for j, o in zip(jobs, out) if j[0] == m and j[1] == sec]
            R[m] = [o[0] for o in sel]
            R[m + "_seconds"] = [o[1] for o in sel]
            R[m + "_raisons"] = [o[2] for o in sel]
            R[m + "_nsims"] = [o[3] for o in sel]
        R["summary"] = {m: summary(R[m], 3) for m in names}
        ref = [premier(tr, 1, 1.0) for tr in R["quadz_modele"]] if "quadz_modele" in names else None
        print(f"{sec}", flush=True)
        for m in names:
            v = R["summary"][m]
            ff = [premier(tr, 1, 1.0) for tr in R[m]]
            ns = np.array(R[m + "_nsims"], float)
            pf = np.median([r["faisable"] / n for r, n in zip(R[m + "_raisons"], ns)])
            p = wilcoxon_signe(ff, ref) if ref is not None and m != "quadz_modele" else float("nan")
            mieux = sum(a < b for a, b in zip(ff, ref)) if ref is not None else 0
            pire = sum(a > b for a, b in zip(ff, ref)) if ref is not None else 0
            v.update(part_faisable=float(pf), p_vs_ref=p, mieux=mieux, pire=pire)
            print(f"   {m:20s} front complet {v['full_front']}, succès {100 * v['success']:.0f} %, "
                  f"HV99 {v['hv99']}, faisables {100 * pf:.0f} %, vs réf. mieux {mieux} pire {pire} p={p:.3f} ; "
                  f"{np.mean(R[m + '_seconds']):.1f} s", flush=True)
        res[sec] = R
    print(f"durée totale {time.time() - t0:.0f} s sur {workers} processus")
    with open(OUT, "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("->", OUT)
