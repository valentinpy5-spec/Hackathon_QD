"""Vérification croisée : notre NSGA-II contre celui de pymoo, sur le même problème compté.

À lancer avec l'environnement du projet (pymoo y est installé) :
    .venv/bin/python -m optim.check_pymoo
"""
import json
import os

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from .problem import scenario, build_table, CountedProblem
from .pareto import pareto_front
from .nsga2 import nsga2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Hackathon(ElementwiseProblem):
    def __init__(self, pb):
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=1, xl=0, xu=7, vtype=int)
        self.pb = pb

    def _evaluate(self, x, out, *args, **kwargs):
        f, cv = self.pb.F(tuple(int(v) for v in np.round(x)))
        out["F"] = f
        out["G"] = [cv]          # g <= 0 : violation totale


def first_full_front(T, exact, runner, seeds):
    res = []
    for seed in seeds:
        pb = CountedProblem(T)
        trace = []
        runner(pb, seed, lambda p: trace.append((p.n_sims, exact <= p._set)))
        hit = [n for n, ok in trace if ok]
        res.append(hit[0] if hit else None)
    return res


def run_pymoo(pb, seed, log, pop=16, n_gen=60):
    class CB(Callback):
        def notify(self, algorithm):
            log(pb)
    algo = NSGA2(pop_size=pop, sampling=IntegerRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                 mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
                 eliminate_duplicates=True)
    minimize(Hackathon(pb), algo, ("n_gen", n_gen), seed=seed, callback=CB(), verbose=False)


def run_ours(pb, seed, log, pop=16, n_gen=60):
    nsga2(pb, pop_size=pop, n_gen=n_gen, seed=seed, callback=lambda g, p: log(p))


if __name__ == "__main__":
    out = {}
    for name in ("S1_note", "S2_sol"):
        T = build_table(scenario(name), path=os.path.join(HERE, "resultats", f"base_complete_{name}.csv"))
        feas = [r for r in T if r["feasible"]]
        F = np.array([[r["CD_counts"], r["SER_db"]] for r in feas])
        exact = {(feas[i]["im"], feas[i]["ip"], feas[i]["it"]) for i in pareto_front(F)}
        seeds = range(20)
        a = first_full_front(T, exact, run_ours, seeds)
        b = first_full_front(T, exact, run_pymoo, seeds)
        out[name] = dict(ours=a, pymoo=b)
        med = lambda v: float(np.median([x if x is not None else np.inf for x in v]))
        print(f"{name}: front exact {len(exact)} profils ; simulations pour le retrouver en entier (médiane sur 20 graines)"
              f" : notre NSGA-II {med(a)}, pymoo {med(b)}")
    with open(os.path.join(HERE, "resultats", "check_pymoo.json"), "w") as f:
        json.dump(out, f, indent=1)
