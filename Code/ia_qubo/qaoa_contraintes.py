"""Itération V3, lot L6 : la contrainte en boîte noire rend-elle le QUBO plus facile pour QAOA ?

Sur des QUBO de début de boucle (S2, 30 profils tirés au hasard, 3 graines, poids 0 ; 0,5 ; 1), on compare :
    modele      : objectif quadratique en z + C_L,max affine avec 3 bits d'écart + Rosenberg (14 qubits, V2)
    boite_noire : modèle quadratique en z ajusté sur l'objectif pondéré, infaisables dégradés,
                  + Rosenberg pour l'épaisseur (11 qubits, pas de variables d'écart)
On mesure la part de l'objectif dans l'étendue des énergies et, pour QAOA p = 2 (3 départs),
la probabilité de l'optimum divisée par celle du hasard (comme quantique/anatomie_qubo.py).

    python3 -m ia_qubo.qaoa_contraintes        -> resultats/ia_qubo_qaoa_contraintes.json  (~3 min)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.problem_v2 import build_table_v2, with_objective, SpaceV2, KEYS  # noqa: E402
from optim.qubo_loop import fit_quadratic, fit_linear  # noqa: E402
from quantique.qubo import surrogate_to_qubo, all_bitstrings, qubo_energy  # noqa: E402
from quantique import qaoa as qa  # noqa: E402
from ia_qubo.boucle import ajoute_contraintes, PENALITE_INFAISABLE  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "ia_qubo_qaoa_contraintes.json")
SP = SpaceV2()
SPEC = dict(bits=SP.bits, levels=SP.n_levels, keys=KEYS, t_var=2)


def mesure(Q, c, Qobj, cobj):
    n = Q.shape[0]
    B = all_bitstrings(n)
    E = qubo_energy(Q, c, B)
    Eo = qubo_energy(Qobj, cobj, B[:, :Qobj.shape[0]])
    o = np.argsort(E)
    En = qa.normalize(E)
    r = qa.optimize_qaoa(En, n, 2, n_restarts=3, seed=1)
    P = np.abs(qa.qaoa_state(r.x[:2], r.x[2:], En, n)) ** 2
    return dict(n=n, part_objectif=float(np.ptp(Eo) / np.ptp(E)), gain=float(P[o[0]] * len(E)))


def main():
    T = with_objective(build_table_v2(), "S2", "MoM")
    out = {"modele": [], "boite_noire": []}
    for seed in range(3):
        rows = [T[i] for i in np.random.default_rng(seed).permutation(len(T))[:30]]
        I = np.array([[r[k] for k in KEYS] for r in rows])
        ok = np.array([r["feasible"] for r in rows])
        val = np.array([r["valid"] for r in rows])
        cd = np.array([r["CD_counts"] for r in rows], float)
        ser = np.array([r["SER_db"] for r in rows], float)
        cl = np.array([r["CLmax"] for r in rows], float)
        vl = val & np.isfinite(cl)
        s_cd, s_ser = np.ptp(cd[ok]), np.ptp(ser[ok])
        c_cd = fit_quadratic(I[ok], cd[ok], levels=SP.n_levels)
        c_ser = fit_quadratic(I[ok], ser[ok], levels=SP.n_levels)
        c_cl = fit_linear(I[vl], cl[vl], levels=SP.n_levels)
        for w in (0.0, 0.5, 1.0):
            Qo, co = surrogate_to_qubo(w * c_cd / s_cd + (1 - w) * c_ser / s_ser, SP.bits, SP.n_levels)
            Q, c = ajoute_contraintes(Qo.copy(), co, c_cl, SPEC, 1.2, modele_cl=True)
            out["modele"].append(dict(seed=seed, w=w, **mesure(Q, c, Qo, co)))
            f = np.full(len(rows), np.nan)
            f[ok] = w * cd[ok] / s_cd + (1 - w) * ser[ok] / s_ser
            f[~ok] = np.nanmax(f) + PENALITE_INFAISABLE * (np.nanmax(f) - np.nanmin(f))
            Qb, cb = surrogate_to_qubo(fit_quadratic(I, f, levels=SP.n_levels), SP.bits, SP.n_levels)
            Q2, c2 = ajoute_contraintes(Qb.copy(), cb, None, SPEC, 1.2, modele_cl=False)
            out["boite_noire"].append(dict(seed=seed, w=w, **mesure(Q2, c2, Qb, cb)))
            print(seed, w, out["modele"][-1], out["boite_noire"][-1], flush=True)
    for k, v in out.items():
        print(f"{k:12s} : {v[0]['n']} qubits, part de l'objectif médiane {np.median([x['part_objectif'] for x in v]):.3f}, "
              f"gain de QAOA sur le hasard médian {np.median([x['gain'] for x in v]):.1f} "
              f"(min {min(x['gain'] for x in v):.1f}, max {max(x['gain'] for x in v):.1f})")
    with open(OUT, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
