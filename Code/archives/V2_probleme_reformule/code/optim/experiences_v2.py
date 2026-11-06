"""Itération V2 : comparaison des méthodes sur le problème reformulé (table XFOIL, 1 024 profils, 10 bits).

Nouveautés par rapport à experiences_pipeline (V1) :
    - NSGA-II réglé (N = 8) en plus de N = 16 ;
    - critère d'arrêt RÉALISTE en plus du critère « front exact retrouvé » :
      on s'arrête quand l'hypervolume de l'archive ne progresse plus
      (NSGA-II : 5 générations ; boucle : 3 itérations), et on mesure alors
      la fraction du front exact et l'hypervolume obtenus ;
    - boucle QUBO à 14 qubits : 10 bits de conception + 1 auxiliaire (t >= 8 %)
      + 3 d'écart (C_L,max >= 1,2, contrainte linéarisée).

    python3 -m optim.experiences_v2          -> resultats/pipeline_V2.json
"""
import json
import os
import sys
import time

import numpy as np

from .problem import CountedProblem
from .problem_v2 import build_table_v2, with_objective, SpaceV2, KEYS
from .pareto import pareto_front, hypervolume_2d, Normalizer
from .nsga2 import nsga2
from .qubo_loop import qubo_loop

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "pipeline_V2.json")
SP = SpaceV2()
CONS = {"t_min_index": True, "t_var": 2, "cl_min": 1.2}


def setup(sector, source="MoM"):
    T = with_objective(build_table_v2(), sector, source)
    feas = [r for r in T if r["feasible"]]
    F = np.array([[r["CD_counts"], r["SER_db"]] for r in feas])
    fr = pareto_front(F)
    exact = {tuple(feas[i][k] for k in KEYS) for i in fr}
    norm = Normalizer(F)
    ref = np.array([1.1, 1.1])
    return T, feas, F, fr, exact, norm, ref, hypervolume_2d(norm(F[fr]), ref)


def state(pb, exact, norm, ref, hv_exact):
    rows = [pb.table[i] for i in pb.seen if pb.table[i]["feasible"]]
    if not rows:
        return 0.0, 0.0
    F = np.array([[r["CD_counts"], r["SER_db"]] for r in rows])
    return len(exact & pb._set) / len(exact), hypervolume_2d(norm(F), ref) / hv_exact


def stop_point(trace, patience):
    """Premier point où l'HV n'a pas progressé depuis `patience` relevés : (n_sims, frac, hv)."""
    for k in range(patience, len(trace)):
        if trace[k][2] <= trace[k - patience][2] + 1e-12:
            return trace[k]
    return trace[-1]


def _run_one(args):
    method, sector, source, s, budget = args
    T, feas, F, fr, exact, norm, ref, hv = setup(sector, source)
    return run(method, T, exact, norm, ref, hv, [s], budget)[0]


def run(method, T, exact, norm, ref, hv, seeds, budget=250):
    traces = []
    for s in seeds:
        pb = CountedProblem(T, keys=KEYS)
        tr = []
        if method.startswith("nsga2"):
            N = int(method.split("_")[1])
            nsga2(pb, pop_size=N, n_gen=1000 // N + 30, seed=s, n_levels=SP.n_levels, max_sims=budget,
                  callback=lambda g, p: tr.append((p.n_sims, *state(p, exact, norm, ref, hv))))
        else:
            solver = method.split("_")[1]

            def cb(p):
                tr.append((p.n_sims, *state(p, exact, norm, ref, hv)))
                # on continue jusqu'à ce que le front soit complet ET que l'HV ait stagné 3 itérations,
                # pour pouvoir mesurer les deux critères d'arrêt sur la même exécution
                return tr[-1][1] >= 1.0 and len(tr) > 3 and tr[-1][2] <= tr[-4][2] + 1e-12
            qubo_loop(pb, n_init=10, solver=solver, seed=s, budget=budget, cons=CONS,
                      stop_when=cb, bits=SP.bits, levels=SP.n_levels, keys=KEYS, lin_col="CLmax")
        traces.append(tr)
    return traces


def run_random(T, exact, norm, ref, hv, seeds):
    keys = list(CountedProblem(T, keys=KEYS).table.keys())
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        pb = CountedProblem(T, keys=KEYS)
        tr = []
        for j in rng.permutation(len(keys)):
            pb(keys[j])
            if pb.n_sims % 8 == 0:
                tr.append((pb.n_sims, *state(pb, exact, norm, ref, hv)))
            if exact <= pb._set:
                tr.append((pb.n_sims, *state(pb, exact, norm, ref, hv)))
                break
        out.append(tr)
    return out


def summary(traces, patience):
    def first(tr, k, thr):
        for row in tr:
            if row[k] >= thr - 1e-12:
                return row[0]
        return np.inf

    def q(v):
        return [float(np.percentile(v, x)) for x in (25, 50, 75)]
    stops = [stop_point(tr, patience) for tr in traces] if patience else None
    out = dict(full_front=q([first(tr, 1, 1.0) for tr in traces]),
               hv99=q([first(tr, 2, 0.99) for tr in traces]),
               success=float(np.mean([np.isfinite(first(tr, 1, 1.0)) for tr in traces])))
    if stops:
        out.update(stop_sims=q([s[0] for s in stops]), stop_frac=q([s[1] for s in stops]),
                   stop_hv=q([s[2] for s in stops]))
    return out


if __name__ == "__main__":
    # Usage : python3 -m optim.experiences_v2 [MoM|PO] [rapide]
    from multiprocessing import Pool
    source = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("MoM", "PO") else "MoM"
    quick = "rapide" in sys.argv
    n_seeds, n_seeds_q = (6, 2) if quick else (20, 5)
    methods = [("nsga2_8", n_seeds), ("nsga2_16", n_seeds), ("loop_exact", n_seeds), ("loop_sa", n_seeds),
               ("loop_qaoa", n_seeds_q)]
    # QAOA (14 qubits, environ 25 s par itération) est plafonné à 150 simulations
    jobs = [(m, sec, source, s, 150 if m == "loop_qaoa" else 250) for sec in ("S1", "S2") for m, ns in methods for s in range(ns)]
    jobs.sort(key=lambda j: j[0] != "loop_qaoa")          # les plus longs d'abord
    t0 = time.time()
    with Pool(10) as pool:
        out = pool.map(_run_one, jobs, chunksize=1)
    res = {}
    for sec in ("S1", "S2"):
        T, feas, F, fr, exact, norm, ref, hv = setup(sec, source)
        R = dict(source=source, n_feasible=len(feas), n_front=len(fr),
                 front=[{k: feas[i][k] for k in ("m", "p", "t", "I", "r_le", "alpha", "CD_counts", "SER_db", "CLmax")}
                        for i in sorted(fr, key=lambda i: F[i, 0])],
                 feasible=[[r["CD_counts"], r["SER_db"], r["t"], r["I"], r["m"]] for r in feas])
        R["random"] = run_random(T, exact, norm, ref, hv, range(n_seeds))
        for m, ns in methods:
            R[m] = [tr for j, tr in zip(jobs, out) if j[0] == m and j[1] == sec]
        R["summary"] = {k: summary(R[k], 5 if k.startswith("nsga2") else (3 if k.startswith("loop") else 0))
                        for k in ("random", "nsga2_8", "nsga2_16", "loop_exact", "loop_sa", "loop_qaoa")}
        res[sec] = R
        print(f"{sec} (SER {source}) : front exact {len(fr)} profils parmi {len(feas)} faisables", flush=True)
        for k, v in R["summary"].items():
            extra = f", arrêt réaliste : {v['stop_sims']} sim., front {v['stop_frac']}, HV {v['stop_hv']}" if "stop_sims" in v else ""
            print(f"   {k:11s} front complet {v['full_front']}, HV99 {v['hv99']}, succès {100 * v['success']:.0f} %{extra}", flush=True)
    print(f"durée totale {time.time() - t0:.0f} s")
    with open(OUT, "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
