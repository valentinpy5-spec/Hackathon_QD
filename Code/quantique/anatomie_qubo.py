"""Itération V3 : anatomie des QUBO de la boucle V2, pour comprendre pourquoi QAOA n'y apporte rien.

On reconstruit les QUBO que la boucle résout (même code : fit_quadratic, fit_linear, build_qubo,
contraintes CONS de experiences_v2) dans deux états de la boucle :
    "debut" : modèles ajustés sur 30 profils tirés au hasard (10 graines) ;
    "fin"   : modèles ajustés sur les 1 024 profils (le meilleur modèle possible).
Pour chaque QUBO on mesure :
    - la part de l'objectif dans l'étendue des énergies (le reste vient des pénalités) ;
    - l'écart d'énergie entre les deux meilleurs profils, rapporté à l'étendue ;
    - si le minimum exact du QUBO est le minimum du problème de substitution qu'il encode
      (objectif du modèle, C_L,max du modèle affine >= 1,2 imposé en dur, t >= 8 %) ;
    - les temps : énumération des 2^14 états, QAOA simulé.
Puis, sur 6 QUBO (S1, S2 x w = 0 ; 0,5 ; 1, état "debut", graine 0), QAOA p = 1, 2, 3 :
probabilité de l'optimum, des 1 % meilleurs états, chance de voir l'optimum en 512 tirs ;
même mesure sur le QUBO SANS contraintes (10 qubits) pour isoler l'effet des pénalités.

    python3 -m quantique.anatomie_qubo        -> resultats/anatomie_qubo_V3.json  (~10 min)
"""
import json
import os
import sys
import time
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.problem_v2 import build_table_v2, with_objective, SpaceV2, KEYS  # noqa: E402
from optim.qubo_loop import fit_quadratic, fit_linear, build_qubo, decode, predict  # noqa: E402
from optim.experiences_v2 import CONS  # noqa: E402
from quantique.qubo import all_bitstrings, qubo_energy  # noqa: E402
from quantique import qaoa as qa  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "anatomie_qubo_V3.json")
SP = SpaceV2()
BITS, LEV = SP.bits, SP.n_levels
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
ALL_IDX = np.array(list(product(*[range(L) for L in LEV])))


def models(T, rows):
    I = np.array([[r[k] for k in KEYS] for r in rows])
    ok = np.array([r["feasible"] for r in rows])
    val = np.array([r["valid"] for r in rows])
    cd = np.array([r["CD_counts"] for r in rows])
    ser = np.array([r["SER_db"] for r in rows])
    cl = np.array([r["CLmax"] for r in rows], dtype=float)
    vl = val & np.isfinite(cl)
    return dict(cd=fit_quadratic(I[ok], cd[ok], levels=LEV), ser=fit_quadratic(I[ok], ser[ok], levels=LEV),
                cl=fit_linear(I[vl], cl[vl], levels=LEV), scd=max(np.ptp(cd[ok]), 1e-6),
                sser=max(np.ptp(ser[ok]), 1e-6))


def anatomy(M, w, B):
    """Mesures sur le QUBO complet (14 qubits) de poids w."""
    Q, c, n = build_qubo(M["cd"], M["ser"], M["cl"], w, M["scd"], M["sser"], CONS, bits=BITS, levels=LEV)
    Qo, co, n0 = build_qubo(M["cd"], M["ser"], M["cl"], w, M["scd"], M["sser"], {}, bits=BITS, levels=LEV)
    t0 = time.perf_counter()
    E = qubo_energy(Q, c, B)
    t_enum = time.perf_counter() - t0
    Eo = qubo_energy(Qo, co, B[:, :n0])
    span = np.ptp(E)
    # meilleure énergie de chaque profil (minimum sur les qubits auxiliaires et d'écart)
    idx = [decode(b, BITS) for b in B]
    best = {}
    for e, ix in zip(E, idx):
        if ix not in best or e < best[ix]:
            best[ix] = e
    prof = sorted(best.values())
    ix_qubo = idx[int(np.argmin(E))]
    # problème de substitution « vrai » : objectif du modèle, contraintes imposées en dur
    f = M_obj(M, w, ALL_IDX)
    feas = (predict(M["cl"], ALL_IDX, levels=LEV) >= CONS["cl_min"]) & (ALL_IDX[:, CONS["t_var"]] > 0)
    ix_sub = tuple(int(v) for v in ALL_IDX[feas][np.argmin(f[feas])]) if feas.any() else None
    # pourquoi le minimum du QUBO diffère-t-il ? profil prédit infaisable (pénalité trop douce)
    # ou prédit faisable mais moins bon (biais de la variable d'écart discrète)
    k_q = int(np.flatnonzero((ALL_IDX == np.array(ix_qubo)).all(axis=1))[0])
    qubo_pred_feasible = bool(feas[k_q])
    qubo_cl_pred = float(predict(M["cl"], ALL_IDX[k_q:k_q + 1], levels=LEV)[0])
    return dict(E=E, Eo_states=Eo, n_qubits=n, t_enum=t_enum,
                obj_share=float(np.ptp(Eo) / span), gap=float((prof[1] - prof[0]) / span),
                same_as_substitute=bool(ix_sub is not None and tuple(ix_qubo) == ix_sub),
                qubo_pred_feasible=qubo_pred_feasible, qubo_cl_pred=qubo_cl_pred,
                n_distinct_profiles=len(best), Q_max_pen=float(np.abs(Q).max()), Q_max_obj=float(np.abs(Qo).max()),
                Qo=Qo, co=co, n0=n0)


def M_obj(M, w, I):
    return w * predict(M["cd"], I, levels=LEV) / M["scd"] + (1 - w) * predict(M["ser"], I, levels=LEV) / M["sser"]


def qaoa_probe(E, n, p, seed=1, shots=512):
    En = qa.normalize(E)
    t0 = time.perf_counter()
    r = qa.optimize_qaoa(En, n, p, n_restarts=3, seed=seed)
    psi = qa.qaoa_state(r.x[:p], r.x[p:], En, n)
    dt = time.perf_counter() - t0
    P = np.abs(psi) ** 2
    o = np.argsort(E)
    p_opt = float(P[o[0]])
    return dict(p=p, p_opt=p_opt, p_random=1 / len(E), p_top1=float(P[o[:max(1, len(E) // 100)]].sum()),
                p_seen_512=float(1 - (1 - p_opt) ** shots), seconds=dt,
                gammas=[float(g) for g in r.x[:p]], betas=[float(b) for b in r.x[p:]])


def main():
    out = dict(weights=WEIGHTS, cons=CONS, anatomy={}, qaoa={})
    for sector in ("S1", "S2"):
        T = with_objective(build_table_v2(), sector, "MoM")
        B = None
        res = {"debut": [], "fin": []}
        states = [("debut", s, [T[i] for i in np.random.default_rng(s).permutation(len(T))[:30]]) for s in range(10)]
        states.append(("fin", 0, T))
        for name, seed, rows in states:
            M = models(T, rows)
            for w in WEIGHTS:
                if B is None:
                    n = build_qubo(M["cd"], M["ser"], M["cl"], w, M["scd"], M["sser"], CONS, bits=BITS, levels=LEV)[2]
                    B = all_bitstrings(n)
                a = anatomy(M, w, B)
                res[name].append(dict(seed=seed, w=w, **{k: a[k] for k in (
                    "n_qubits", "t_enum", "obj_share", "gap", "same_as_substitute", "qubo_pred_feasible", "qubo_cl_pred",
                    "n_distinct_profiles",
                    "Q_max_pen", "Q_max_obj")}))
                if name == "debut" and seed == 0 and w in (0.0, 0.5, 1.0):
                    key = f"{sector}_w{w}"
                    Bo = all_bitstrings(a["n0"])
                    Eo = qubo_energy(a["Qo"], a["co"], Bo)
                    out["qaoa"][key] = dict(
                        complet=[qaoa_probe(a["E"], a["n_qubits"], p) for p in (1, 2, 3)],
                        sans_contraintes=[qaoa_probe(Eo, a["n0"], p) for p in (1, 2, 3)])
                    if sector == "S2" and w == 0.5:
                        Es = np.sort(a["E"])
                        span = Es[-1] - Es[0]
                        out["spectre"] = dict(
                            rang=[int(k) for k in np.unique(np.geomspace(1, len(Es), 120).astype(int)) - 1],
                            E_rel=[float((Es[k] - Es[0]) / span) for k in
                                   np.unique(np.geomspace(1, len(Es), 120).astype(int)) - 1],
                            obj_share=a["obj_share"])
                    print(key, "QAOA", [round(q["p_opt"], 5) for q in out["qaoa"][key]["complet"]],
                          "sans contraintes", [round(q["p_opt"], 5) for q in out["qaoa"][key]["sans_contraintes"]],
                          flush=True)
        out["anatomy"][sector] = res
        for name in res:
            r = res[name]
            print(f"{sector} {name}: part objectif médiane {np.median([x['obj_share'] for x in r]):.3f}, "
                  f"écart médian {np.median([x['gap'] for x in r]):.1e}, "
                  f"minimum QUBO = minimum du problème de substitution {np.mean([x['same_as_substitute'] for x in r]):.0%}, "
                  f"minimum QUBO prédit faisable {np.mean([x['qubo_pred_feasible'] for x in r]):.0%}, "
                  f"énumération {1e3 * np.median([x['t_enum'] for x in r]):.1f} ms", flush=True)
    log = os.path.join(HERE, "resultats", "base_V2.log")
    out["xfoil_s_par_profil"] = 351 / 1024 if not os.path.exists(log) else None
    if os.path.exists(log):
        txt = open(log).read()
        try:
            out["xfoil_s_par_profil"] = float(txt.split(" en ")[1].split(" s")[0]) / 1024
        except (IndexError, ValueError):
            out["xfoil_s_par_profil"] = 351 / 1024
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("->", OUT)


if __name__ == "__main__":
    main()
