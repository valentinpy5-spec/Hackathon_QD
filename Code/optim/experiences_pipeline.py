"""Chaîne complète : base exhaustive -> front exact -> NSGA-II -> boucle QUBO (exact, recuit, QAOA).

Produit resultats/pipeline.json, lu par les scripts de figures des cours.

    python3 -m optim.experiences_pipeline            # tout (quelques minutes)
    python3 -m optim.experiences_pipeline rapide     # moins de graines
"""
import json
import os
import sys
import time

import numpy as np

from .problem import scenario, build_table, CountedProblem
from .pareto import pareto_front, hypervolume_2d, Normalizer
from .nsga2 import nsga2
from .qubo_loop import qubo_loop

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats")


def setup(name):
    T = build_table(scenario(name), path=os.path.join(OUT, f"base_complete_{name}.csv"))
    feas = [r for r in T if r["feasible"]]
    F = np.array([[r["CD_counts"], r["SER_db"]] for r in feas])
    fr = pareto_front(F)
    exact = {(feas[i]["im"], feas[i]["ip"], feas[i]["it"]) for i in fr}
    norm = Normalizer(F)
    ref = np.array([1.1, 1.1])
    hv_exact = hypervolume_2d(norm(F[fr]), ref)
    return T, feas, F, fr, exact, norm, ref, hv_exact


def metrics(pb, exact, norm, ref, hv_exact):
    rows = [pb.table[i] for i in pb.seen if pb.table[i]["feasible"]]
    if not rows:
        return 0.0, 0.0
    F = np.array([[r["CD_counts"], r["SER_db"]] for r in rows])
    frac = len(exact & pb._set) / len(exact)
    return frac, hypervolume_2d(norm(F), ref) / hv_exact


def first(trace, key, thr):
    for n, frac, hv in trace:
        if (frac if key == "frac" else hv) >= thr:
            return n
    return None


def run_nsga(T, exact, norm, ref, hv, seeds, pop=16, n_gen=60):
    out = []
    for s in seeds:
        pb = CountedProblem(T)
        tr = []
        nsga2(pb, pop_size=pop, n_gen=n_gen, seed=s,
              callback=lambda g, p: tr.append((p.n_sims, *metrics(p, exact, norm, ref, hv))))
        out.append(tr)
    return out


def run_loop(T, exact, norm, ref, hv, seeds, solver, budget=150, n_init=10):
    out = []
    for s in seeds:
        pb = CountedProblem(T)
        tr = []

        def stop(p):
            tr.append((p.n_sims, *metrics(p, exact, norm, ref, hv)))
            return exact <= p._set
        qubo_loop(pb, n_init=n_init, solver=solver, seed=s, budget=budget,
                  cons={"t_min_index": True, "cl_min": 0.3}, stop_when=stop)
        out.append(tr)
    return out


def run_random(T, exact, norm, ref, hv, seeds):
    keys = list(CountedProblem(T).table.keys())
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        pb = CountedProblem(T)
        tr = []
        for j in rng.permutation(len(keys)):
            pb(keys[j])
            if pb.n_sims % 4 == 0 or exact <= pb._set:
                tr.append((pb.n_sims, *metrics(pb, exact, norm, ref, hv)))
            if exact <= pb._set:
                break
        out.append(tr)
    return out


def summary(traces):
    def med(v):
        v = [x if x is not None else np.inf for x in v]
        return [float(np.percentile(v, q)) for q in (25, 50, 75)]
    return dict(full_front=med([first(t, "frac", 1.0) for t in traces]),
                hv99=med([first(t, "hv", 0.99) for t in traces]),
                success=float(np.mean([first(t, "frac", 1.0) is not None for t in traces])))


if __name__ == "__main__":
    quick = len(sys.argv) > 1 and sys.argv[1] == "rapide"
    seeds = range(6 if quick else 20)
    seeds_q = range(3 if quick else 10)
    res = {}
    for name in ("S1_note", "S2_sol"):
        t0 = time.time()
        T, feas, F, fr, exact, norm, ref, hv = setup(name)
        R = dict(front=[feas[i] for i in fr], feasible=[[r["CD_counts"], r["SER_db"], r["t"], r["m"], r["p"]] for r in feas],
                 n_feasible=len(feas), n_front=len(fr))
        R["nsga2"] = run_nsga(T, exact, norm, ref, hv, seeds)
        R["random"] = run_random(T, exact, norm, ref, hv, seeds)
        R["loop_exact"] = run_loop(T, exact, norm, ref, hv, seeds, "exact")
        R["loop_sa"] = run_loop(T, exact, norm, ref, hv, seeds, "sa")
        R["loop_qaoa"] = run_loop(T, exact, norm, ref, hv, seeds_q, "qaoa")
        R["summary"] = {k: summary(R[k]) for k in ("nsga2", "random", "loop_exact", "loop_sa", "loop_qaoa")}
        res[name] = R
        print(f"{name} ({time.time() - t0:.0f} s) : front exact {len(fr)} profils parmi {len(feas)} faisables")
        for k, v in R["summary"].items():
            print(f"   {k:11s} simulations pour le front complet (q1, médiane, q3) = {v['full_front']}, "
                  f"HV 99 % = {v['hv99']}, succès {100 * v['success']:.0f} %")
    with open(os.path.join(OUT, "pipeline.json"), "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
