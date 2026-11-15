"""Itération V3, lot L1 : deux diagnostics qui expliquent les résultats de optim/experiences_l1.py.

1. Contrainte de manœuvre. Le front exact est au bord de la contrainte C_L,max >= 1,2. On mesure,
   pour la boucle QUBO (pénalité x2,5, x25 = réglage V2, x250, ou sans contrainte) et pour les
   variantes à modèle quadratique de optim/bayes.py, la part des profils PROPOSÉS qui violent la
   contrainte, et le nombre de simulations pour retrouver le front (arrêt dès le front complet).
2. Qualité des modèles. Erreur de prédiction du modèle quadratique et du krigeage sur les profils
   non vus, selon le nombre de profils d'entraînement tirés au hasard.

3. Profils du front. Ce que prédisent les modèles affine, quadratique et krigeage de C_L,max,
   entraînés sur toute la table, pour les profils du front de S1.

    python -m optim.diagnostic_l1            # -> resultats/diagnostic_L1_V3.json (quelques minutes)
    python -m optim.diagnostic_l1 front      # ne recalcule que le point 3
"""
import json
import os
import time

import numpy as np

from .problem import CountedProblem
from .problem_v2 import SpaceV2, KEYS
from .experiences_v2 import setup
from .qubo_loop import qubo_loop, z_of
from .bayes import bayes_loop, GPModel, QuadModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "diagnostic_L1_V3.json")
SP = SpaceV2()
BASE = {"t_min_index": True, "t_var": 2}
CL_MIN = 1.2
N_SEEDS, BUDGET = 20, 250

# nom -> ("qubo", cons) ou ("bayes", cons, options de bayes_loop)
CASES = {
    "qubo_x25": ("qubo", {**BASE, "cl_min": CL_MIN}),
    "qubo_x2.5": ("qubo", {**BASE, "cl_min": CL_MIN, "cl_weight": 2.5}),
    "qubo_x250": ("qubo", {**BASE, "cl_min": CL_MIN, "cl_weight": 250.0}),
    "qubo_sans": ("qubo", BASE),
    "quad_mean_lin": ("bayes", {**BASE, "cl_min": CL_MIN}, dict(model="quad", selection="mean", constraint="lin")),
    "quad_mean_pof": ("bayes", {**BASE, "cl_min": CL_MIN}, dict(model="quad", selection="mean", constraint="pof")),
    "quad_mean_sans": ("bayes", BASE, dict(model="quad", selection="mean")),
    "quad_ei_pof": ("bayes", {**BASE, "cl_min": CL_MIN}, dict(model="quad", selection="ei", constraint="pof")),
    "quad_ei_sans": ("bayes", BASE, dict(model="quad", selection="ei")),
}


def _constraint_one(args):
    name, sector, seed = args
    T, feas, F, fr, exact, norm, ref, hv = setup(sector, "MoM")
    pb = CountedProblem(T, keys=KEYS)
    case = CASES[name]

    def done(p):
        return exact <= p._set
    if case[0] == "qubo":
        qubo_loop(pb, n_init=10, solver="exact", seed=seed, budget=BUDGET, cons=case[1], stop_when=done,
                  bits=SP.bits, levels=SP.n_levels, keys=KEYS, lin_col="CLmax")
    else:
        bayes_loop(pb, seed=seed, budget=BUDGET, cons=case[1], stop_when=done,
                   levels=SP.n_levels, keys=KEYS, **case[2])
    prop = [pb.table[i] for i in pb.seen[10:]]                 # hors les 10 points initiaux
    viol = float(np.mean([not (r["CLmax"] >= CL_MIN) for r in prop])) if prop else 0.0
    return dict(n=pb.n_sims if exact <= pb._set else None, viol=viol)


def _quality_one(args):
    sector, n, rep = args
    T, *_ = setup(sector, "MoM")
    rows = [r for r in T if r["valid"] and np.isfinite(r["CLmax"]) and np.isfinite(r["CD_counts"])]
    Z = z_of(np.array([[r[k] for k in KEYS] for r in rows]), SP.n_levels)
    rng = np.random.default_rng(1000 * n + rep)
    tr = rng.permutation(len(rows))[:n]
    te = np.setdiff1d(np.arange(len(rows)), tr)
    out = {}
    for col in ("CD_counts", "SER_db", "CLmax"):
        y = np.array([r[col] for r in rows])
        for mname, M in (("quad", QuadModel), ("gp", GPModel)):
            mu, sd = M(Z[tr], y[tr]).predict(Z[te])
            err = mu - y[te]
            # erreur relative à l'écart-type de la grandeur, et part des erreurs couvertes par 2 écarts-types prédits
            out[f"{col}_{mname}"] = dict(rmse_rel=float(np.sqrt(np.mean(err**2)) / np.std(y)),
                                         cover=float(np.mean(np.abs(err) <= 2 * sd)))
    return out


def front_predictions(sector="S1"):
    """Ce que prédisent les trois modèles de C_L,max (entraînés sur toute la table) pour les profils du front."""
    from scipy.stats import norm
    from .qubo_loop import fit_linear, predict
    T, feas, F, fr, exact, *_ = setup(sector, "MoM")
    tab = {tuple(r[k] for k in KEYS): r for r in T}
    rows = [r for r in T if r["valid"] and np.isfinite(r["CLmax"])]
    I = np.array([[r[k] for k in KEYS] for r in rows])
    y = np.array([r["CLmax"] for r in rows])
    Z = z_of(I, SP.n_levels)
    E = np.array(sorted(exact))
    A = np.array(SP.all_indices())
    coef = fit_linear(I, y, levels=SP.n_levels)
    lin = predict(coef, E, SP.n_levels)
    mq, sq = QuadModel(Z, y).predict(z_of(E, SP.n_levels))
    mg, sg = GPModel(Z, y).predict(z_of(E, SP.n_levels))
    out = dict(profiles=[], rmse_lin=float(np.sqrt(np.mean((predict(coef, I, SP.n_levels) - y) ** 2))),
               rmse_quad=float(np.sqrt(np.mean((QuadModel(Z, y).predict(Z)[0] - y) ** 2))),
               share_feasible_lin=float(np.mean(predict(coef, A, SP.n_levels) >= CL_MIN)),
               share_feasible_true=float(np.mean([tab[tuple(a)]["CLmax"] >= CL_MIN for a in A])))
    for k, e in enumerate(E):
        r = tab[tuple(e)]
        out["profiles"].append(dict(
            m=r["m"], p=r["p"], t=r["t"], I=r["I"], CLmax=r["CLmax"],
            lin=float(lin[k]), lin_penalty=float(75.0 * max(CL_MIN - lin[k], 0.0) ** 2),
            quad=float(mq[k]), quad_std=float(sq[k]), quad_penalty=float(1 - norm.cdf((mq[k] - CL_MIN) / sq[k])),
            gp=float(mg[k]), gp_std=float(sg[k]), gp_penalty=float(1 - norm.cdf((mg[k] - CL_MIN) / sg[k]))))
    return out


if __name__ == "__main__":
    import sys
    if "front" in sys.argv:                  # ne recalcule que le tableau des profils du front
        res = json.load(open(OUT))
        res["front"] = front_predictions("S1")
        json.dump(res, open(OUT, "w"))
        for r in res["front"]["profiles"]:
            print({k: round(v, 3) for k, v in r.items()})
        print({k: round(v, 3) for k, v in res["front"].items() if k != "profiles"})
        sys.exit()
    from multiprocessing import Pool
    t0 = time.time()
    workers = max(1, min(10, (os.cpu_count() or 2) - 1))
    res = {"constraint": {}, "quality": {}, "n_seeds": N_SEEDS, "budget": BUDGET}
    with Pool(workers) as pool:
        jobs = [(name, sec, s) for sec in ("S1", "S2") for name in CASES for s in range(N_SEEDS)]
        out = pool.map(_constraint_one, jobs, chunksize=1)
        for sec in ("S1", "S2"):
            T, *_ = setup(sec, "MoM")
            res["constraint"][sec] = {"base_rate": float(np.mean([not (r["CLmax"] >= CL_MIN) for r in T]))}
            for name in CASES:
                o = [x for j, x in zip(jobs, out) if j[0] == name and j[1] == sec]
                res["constraint"][sec][name] = dict(n=[x["n"] for x in o], viol=[x["viol"] for x in o])
                n = np.array([x["n"] if x["n"] is not None else np.inf for x in o])
                print(f"{sec} {name:15s} médiane {np.median(n):6.1f}  réussite {100 * np.mean(np.isfinite(n)):3.0f} %  "
                      f"proposés violant la manœuvre {100 * np.mean([x['viol'] for x in o]):3.0f} %", flush=True)
        sizes, reps = (20, 40, 80, 160), 10
        jobs = [("S1", n, r) for n in sizes for r in range(reps)]
        out = pool.map(_quality_one, jobs, chunksize=1)
        for n in sizes:
            o = [x for j, x in zip(jobs, out) if j[1] == n]
            res["quality"][str(n)] = {k: dict(rmse_rel=[x[k]["rmse_rel"] for x in o], cover=[x[k]["cover"] for x in o])
                                      for k in o[0]}
            print(f"n = {n:3d} : " + ", ".join(f"{k} {np.median([x[k]['rmse_rel'] for x in o]):.2f}" for k in o[0]), flush=True)
    res["front"] = front_predictions("S1")
    with open(OUT, "w") as f:
        json.dump(res, f)
    print(f"durée {time.time() - t0:.0f} s ->", OUT)
