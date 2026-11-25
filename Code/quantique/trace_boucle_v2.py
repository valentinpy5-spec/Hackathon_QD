"""Itération V3 (cours quantique) : trace pas à pas de la boucle QUBO V2, pour l'expliquer.

On rejoue les premières itérations de optim.qubo_loop (scénario S2, graine 0, solveur exact,
mêmes réglages que optim/experiences_v2) en enregistrant tout ce qui se passe :
    - les profils initiaux, puis à chaque itération (jusqu'à ce que le front exact soit complet) les modèles, les 5 QUBO (un par poids w),
      le candidat retenu, sa prédiction, et sa vraie valeur une fois simulé ;
    - pour le premier QUBO (w = 0,5) : la matrice Q complète (14 x 14), la décomposition de
      l'énergie d'une chaîne en objectif + pénalité d'épaisseur + pénalité de manœuvre ;
    - le recuit simulé sur les 5 QUBO de la première itération, avec la température de
      quantique/recuit.py (réglée sur le plus grand coefficient de Q, donc sur les pénalités)
      puis avec une température réglée sur l'objectif.
La trace est vérifiée : les profils simulés sont exactement ceux de qubo_loop au même budget.

    python3 -m quantique.trace_boucle_v2      -> resultats/trace_boucle_V2.json   (~2 min)
"""
import json
import os
import sys
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.problem import CountedProblem  # noqa: E402
from optim.problem_v2 import build_table_v2, with_objective, SpaceV2, KEYS  # noqa: E402
from optim.qubo_loop import (fit_quadratic, fit_linear, build_qubo, ranked_candidates, decode,  # noqa: E402
                             predict, qubo_loop, lin_bits)
from optim.experiences_v2 import CONS  # noqa: E402
from optim.pareto import pareto_front  # noqa: E402
from quantique.qubo import all_bitstrings, qubo_energy  # noqa: E402
from quantique.recuit import simulated_annealing  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "trace_boucle_V2.json")
SP = SpaceV2()
BITS, LEV = SP.bits, SP.n_levels
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SECTOR, SEED, N_INIT, N_ITER = "S2", 0, 10, 60   # on s'arrête dès que le front est complet


def phys(idx):
    m, p, t, I = (SP.levels[k][idx[k]] for k in range(4))
    return dict(m=round(100 * m, 1), p=p, t=round(100 * t, 1), I=I)


def main():
    T = with_objective(build_table_v2(), SECTOR, "MoM")
    feas_all = [r for r in T if r["feasible"]]
    F = np.array([[r["CD_counts"], r["SER_db"]] for r in feas_all])
    exact = {tuple(feas_all[i][k] for k in KEYS) for i in pareto_front(F)}
    pb = CountedProblem(T, keys=KEYS)
    rng = np.random.default_rng(SEED)
    all_idx = list(product(*[range(L) for L in LEV]))
    init = [all_idx[j] for j in rng.permutation(len(all_idx))[:N_INIT]]
    for idx in init:
        pb(idx)
    out = dict(sector=SECTOR, seed=SEED, cons=CONS, weights=WEIGHTS, front=[phys(i) | {"idx": i} for i in exact],
               init=[dict(idx=i, **phys(i), feasible=pb.table[i]["feasible"], CD=pb.table[i]["CD_counts"],
                          SER=pb.table[i]["SER_db"], CLmax=pb.table[i]["CLmax"]) for i in init],
               iterations=[])
    B = None
    for it in range(1, N_ITER + 1):
        rows = [pb.table[i] for i in pb.seen]
        I = np.array([[r[k] for k in KEYS] for r in rows])
        ok = np.array([r["feasible"] for r in rows])
        val = np.array([r["valid"] for r in rows])
        cd = np.array([r["CD_counts"] for r in rows])
        ser = np.array([r["SER_db"] for r in rows])
        cl = np.array([r["CLmax"] for r in rows], dtype=float)
        vl = val & np.isfinite(cl)
        c_cd, c_ser = fit_quadratic(I[ok], cd[ok], levels=LEV), fit_quadratic(I[ok], ser[ok], levels=LEV)
        c_cl = fit_linear(I[vl], cl[vl], levels=LEV)
        s_cd, s_ser = max(np.ptp(cd[ok]), 1e-6), max(np.ptp(ser[ok]), 1e-6)
        rec = dict(it=it, n_sims=pb.n_sims, n_feasible=int(ok.sum()), s_cd=float(s_cd), s_ser=float(s_ser),
                   coef_cd=c_cd.tolist(), coef_ser=c_ser.tolist(), coef_cl=c_cl[:5].tolist(), cand=[])
        new = []
        for w in WEIGHTS:
            Q, c, n = build_qubo(c_cd, c_ser, c_cl, w, s_cd, s_ser, CONS, bits=BITS, levels=LEV)
            if B is None:
                B = all_bitstrings(n)
            E = qubo_energy(Q, c, B)
            ranked = ranked_candidates(Q, c, "exact", rng, bits=BITS)
            pick, rank = None, None
            for k, idx in enumerate(ranked):
                if idx not in pb._set and idx not in new:
                    pick, rank = idx, k
                    break
            new.append(pick)
            z = np.array([pick])
            rec["cand"].append(dict(w=w, idx=pick, **phys(pick), rank_in_qubo=rank,
                                    best_qubo_profile=ranked[0], already_simulated=rank > 0,
                                    pred_CD=float(predict(c_cd, z, levels=LEV)[0]),
                                    pred_SER=float(predict(c_ser, z, levels=LEV)[0]),
                                    pred_CLmax=float(predict(c_cl, z, levels=LEV)[0])))
            if it == 1:
                # ----- anatomie du premier QUBO de chaque poids : recuit simulé, deux températures
                Qo, co, n0 = build_qubo(c_cd, c_ser, c_cl, w, s_cd, s_ser, {}, bits=BITS, levels=LEV)
                o = int(np.argmin(E))
                prof_opt = decode(B[o], BITS)
                sa = {}
                for name, scale in (("penalites", np.abs(Q).max()), ("objectif", np.abs(Qo).max())):
                    _, _, fin, st = simulated_annealing(Q, c, n_sweeps=150, n_reads=40, seed=1,
                                                        T0=2 * scale, T1=0.01 * scale, return_states=True)
                    sa[name] = dict(T1=float(0.01 * scale),
                                    frac_opt=float(np.mean(np.abs(fin - E[o]) < 1e-9)),
                                    frac_profil_opt=float(np.mean([decode(s, BITS) == prof_opt for s in st])))
                rec["cand"][-1]["sa"] = sa
                rec["cand"][-1]["E_gap_top2_profiles"] = None
                if w == 0.5:
                    const, lin = lin_bits(c_cl, BITS, LEV)
                    b = B[o]
                    e_obj = float(qubo_energy(Qo, co, b[:n0])[0])
                    Qt, ct, _ = build_qubo(c_cd, c_ser, c_cl, w, s_cd, s_ser, {"t_min_index": True, "t_var": 2},
                                           bits=BITS, levels=LEV)
                    e_t = float(qubo_energy(Qt, ct, b[:n0 + 1])[0]) - e_obj
                    # même profil avec t à l'indice 0 (6 %) : la pénalité doit valoir P = 3
                    bt = b.copy()
                    bt[5:8] = 0
                    best_aux = min(float(qubo_energy(Qt, ct, np.r_[bt[:n0], y])[0]) for y in (0, 1))
                    e_obj_t = float(qubo_energy(Qo, co, bt[:n0])[0])
                    out["qubo_w05"] = dict(
                        Q=Q.tolist(), c=float(c), n=n, argmin_bits=b.astype(int).tolist(), argmin_idx=prof_opt,
                        argmin_phys=phys(prof_opt), E_total=float(E[o]), E_obj=e_obj, E_t=e_t,
                        E_cl=float(E[o]) - e_obj - e_t, Qmax=float(np.abs(Q).max()), Qobj_max=float(np.abs(Qo).max()),
                        L_const=float(const), L_coefs=[float(a) for _, a in lin],
                        L_argmin=float(const + sum(a * b[i] for i, a in lin)),
                        slack_bits=b[11:14].astype(int).tolist(),
                        t0_penalty=best_aux - e_obj_t,
                        n_couplings=int(sum(1 for i in range(n) for j in range(i + 1, n) if abs(Q[i, j]) > 1e-12)),
                        obj_range=float(np.ptp(qubo_energy(Qo, co, B[:2 ** n0, :n0]))), E_range=float(np.ptp(E)))
        for idx in new:
            pb(idx)
        for cnd in rec["cand"]:
            r = pb.table[cnd["idx"]]
            cnd.update(true_CD=r["CD_counts"], true_SER=r["SER_db"], true_CLmax=r["CLmax"],
                       feasible=r["feasible"], in_front=cnd["idx"] in exact)
        rec["front_found"] = len(exact & pb._set)
        out["iterations"].append(rec)
        if it > 4:                       # au-delà des 4 premières itérations, on ne garde que l'essentiel
            for cnd in rec["cand"]:
                for k in ("pred_CD", "pred_SER", "pred_CLmax", "best_qubo_profile"):
                    cnd.pop(k, None)
            rec.pop("coef_cd"); rec.pop("coef_ser"); rec.pop("coef_cl")
        print(f"it {it} : {pb.n_sims} simulations, {rec['front_found']}/{len(exact)} profils du front, "
              f"candidats faisables {sum(c['feasible'] for c in rec['cand'])}/5", flush=True)
        if rec["front_found"] == len(exact):
            break
    # ----- vérification : la trace reproduit qubo_loop
    pb2 = CountedProblem(T, keys=KEYS)
    qubo_loop(pb2, n_init=N_INIT, solver="exact", seed=SEED, budget=pb.n_sims, cons=CONS,
              bits=BITS, levels=LEV, keys=KEYS, lin_col="CLmax")
    out["trace_identique_a_qubo_loop"] = pb2._set == pb._set
    print("trace identique à qubo_loop :", out["trace_identique_a_qubo_loop"])
    for c in out["iterations"][0]["cand"]:
        print(f"  w={c['w']}: recuit (T réglée sur les pénalités) {c['sa']['penalites']}, "
              f"(T réglée sur l'objectif) {c['sa']['objectif']}")
    q = out["qubo_w05"]
    print("QUBO w=0,5 :", {k: q[k] for k in ("argmin_phys", "E_total", "E_obj", "E_t", "E_cl", "L_argmin",
                                             "slack_bits", "t0_penalty", "n_couplings", "Qmax", "Qobj_max",
                                             "obj_range", "E_range")})
    with open(OUT, "w") as fh:
        json.dump(out, fh, default=lambda o: o.item() if hasattr(o, "item") else list(o))
    print("->", OUT)


if __name__ == "__main__":
    main()
