"""Itération V3, lot L1 : la boucle QUBO face à l'optimisation bayésienne, sur le problème V2.

Même protocole que optim/experiences_v2.py (table XFOIL des 1 024 profils, SER par méthode
des moments, CountedProblem, budget 250, critère « front exact retrouvé » et arrêt réaliste).
Les méthodes comparées sont l'ablation modèle x sélection de optim/bayes.py, plus la boucle
QUBO à solveur exact relancée sur les mêmes graines (comparaison appariée).

    python -m optim.experiences_l1                 # 20 graines -> resultats/pipeline_L1_V3.json
    python -m optim.experiences_l1 rapide          # 4 graines  -> resultats/pipeline_L1_V3_rapide.json
    python -m optim.experiences_l1 rapide S1       # un seul scénario
"""
import json
import os
import sys
import time

from .problem import CountedProblem
from .problem_v2 import SpaceV2, KEYS
from .experiences_v2 import setup, state, summary, CONS
from .qubo_loop import qubo_loop
from .bayes import bayes_loop

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = SpaceV2()

# nom -> (modèle, sélection, scalarisation, contrainte) ; None = boucle QUBO de la V2
METHODS = {
    "loop_exact": None,
    "quad_mean": ("quad", "mean", "sum", "lin"),      # contrôle : la boucle QUBO sans encodage binaire
    "gp_mean": ("gp", "mean", "sum", "lin"),
    "quad_mean_pof": ("quad", "mean", "sum", "pof"),  # contrainte stricte : échoue
    "quad_ei": ("quad", "ei", "sum", "pof"),
    "gp_ei": ("gp", "ei", "sum", "pof"),
    "parego": ("gp", "ei", "tcheby", "pof"),
    "gp_ei_dur": ("gp", "ei", "sum", "mean"),        # contrainte en dur sur la prédiction (SEGO)
}


def _run_one(args):
    method, sector, source, seed, budget = args
    T, feas, F, fr, exact, norm, ref, hv = setup(sector, source)
    pb = CountedProblem(T, keys=KEYS)
    tr = []

    def cb(p):
        tr.append((p.n_sims, *state(p, exact, norm, ref, hv)))
        # même règle que la V2 : front complet ET hypervolume stagnant depuis 3 itérations
        return tr[-1][1] >= 1.0 and len(tr) > 3 and tr[-1][2] <= tr[-4][2] + 1e-12

    t0 = time.time()
    if METHODS[method] is None:
        qubo_loop(pb, n_init=10, solver="exact", seed=seed, budget=budget, cons=CONS, stop_when=cb,
                  bits=SP.bits, levels=SP.n_levels, keys=KEYS, lin_col="CLmax")
    else:
        model, selection, scalar, constraint = METHODS[method]
        bayes_loop(pb, model=model, selection=selection, scalar=scalar, constraint=constraint,
                   n_init=10, seed=seed, budget=budget, cons=CONS, stop_when=cb,
                   levels=SP.n_levels, keys=KEYS)
    return tr, time.time() - t0


if __name__ == "__main__":
    from multiprocessing import Pool
    quick = "rapide" in sys.argv
    sectors = [s for s in ("S1", "S2") if s in sys.argv] or ["S1", "S2"]
    names = [m for m in METHODS if m in sys.argv] or list(METHODS)
    n_seeds, budget, source = (4 if quick else 20), 250, "MoM"
    workers = max(1, min(10, (os.cpu_count() or 2) - 1))
    jobs = [(m, sec, source, s, budget) for sec in sectors for m in names for s in range(n_seeds)]
    t0 = time.time()
    with Pool(workers) as pool:
        out = pool.map(_run_one, jobs, chunksize=1)
    res = {}
    for sec in sectors:
        T, feas, F, fr, exact, norm, ref, hv = setup(sec, source)
        R = dict(source=source, n_feasible=len(feas), n_front=len(fr), n_seeds=n_seeds, budget=budget)
        for m in names:
            R[m] = [o[0] for j, o in zip(jobs, out) if j[0] == m and j[1] == sec]
            R[m + "_seconds"] = [o[1] for j, o in zip(jobs, out) if j[0] == m and j[1] == sec]
        R["summary"] = {m: summary(R[m], 3) for m in names}
        res[sec] = R
        print(f"{sec} : front exact {len(fr)} profils parmi {len(feas)} faisables, {n_seeds} graines", flush=True)
        for m in names:
            v = R["summary"][m]
            print(f"   {m:11s} front complet {v['full_front']}, HV99 {v['hv99']}, succès {100 * v['success']:.0f} %, "
                  f"arrêt réaliste : {v['stop_sims']} sim., front {v['stop_frac']}, HV {v['stop_hv']} ; "
                  f"{sum(R[m + '_seconds']) / n_seeds:.0f} s par graine", flush=True)
    print(f"durée totale {time.time() - t0:.0f} s sur {workers} processus")
    out_path = os.path.join(HERE, "resultats", "pipeline_L1_V3_rapide.json" if quick else "pipeline_L1_V3.json")
    with open(out_path, "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("->", out_path)
