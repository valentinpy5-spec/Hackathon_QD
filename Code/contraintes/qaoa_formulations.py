"""Itération V3, lot L7 : la formulation des contraintes change-t-elle ce que voit QAOA ?

Mêmes QUBO de début de boucle que ia_qubo/qaoa_contraintes.py (problème V2, S2, 30 profils tirés au
hasard, 3 graines, poids 0 ; 0,5 ; 1). Le problème à résoudre est toujours le même : minimiser l'objectif
prédit sous C_L,max prédit >= 1,2 et t > t_min. Trois formulations :

    ecart          V2 : bits d'écart (3) + Rosenberg pour l'épaisseur (1) -> 14 qubits
    desequilibree  pénalité déséquilibrée sur C_L,max + Rosenberg       -> 11 qubits
    lagrange       objectif - lambda h_CL (lambda par montée duale, solveur exact) ; l'épaisseur n'est
                   PAS dans le QUBO : elle est vérifiée sur les échantillons     -> 10 qubits

Mesures (QAOA p = 2, 3 départs, vecteur d'état exact) :
    part_objectif   part de l'objectif dans l'étendue des énergies du QUBO
    p_optimum       probabilité d'obtenir la conception optimale DU PROBLÈME CONTRAINT (bits auxiliaires
                    marginalisés), divisée par celle du tirage uniforme d'une conception (1/1024)
    p_faisable      probabilité qu'un échantillon soit faisable (prédit) et d'épaisseur admise

    python3 -m contraintes.qaoa_formulations    -> resultats/contraintes_qaoa.json  (~3 min)
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
from ia_qubo.boucle import ajoute_contraintes  # noqa: E402
from ia_qubo.modeles import decode_bits  # noqa: E402
from contraintes.formulations import Contrainte, penalite_desequilibree, resout_lagrange, lagrangien  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "contraintes_qaoa.json")
SP = SpaceV2()
SPEC = dict(bits=SP.bits, levels=SP.n_levels, keys=KEYS, t_var=2)
ND = sum(SP.bits)


def mesure(Q, c, Bd, x_opt, faisable):
    """Bd : conceptions (2^ND, ND) ; x_opt : indice de la conception optimale contrainte ;
    faisable : masque des conceptions faisables (prédites). Les bits auxiliaires sont marginalisés."""
    n = Q.shape[0]
    B = all_bitstrings(n)
    E = qubo_energy(Q, c, B)
    En = qa.normalize(E)
    r = qa.optimize_qaoa(En, n, 2, n_restarts=3, seed=1)
    prob = np.abs(qa.qaoa_state(r.x[:2], r.x[2:], En, n)) ** 2
    design = (B[:, :ND] * (1 << np.arange(ND))).sum(1).astype(int)        # entier de la conception
    pd = np.bincount(design, weights=prob, minlength=2 ** ND)
    return dict(qubits=n, p_optimum=float(pd[x_opt] * 2 ** ND), p_faisable=float(pd[faisable].sum()))


def main():
    T = with_objective(build_table_v2(), "S2", "MoM")
    Bd = all_bitstrings(ND)
    Id = decode_bits(Bd, SP.bits)
    t_ok = Id[:, 2] != 0
    out = {f: [] for f in ("ecart", "desequilibree", "lagrange")}
    for seed in range(3):
        rows = [T[i] for i in np.random.default_rng(seed).permutation(len(T))[:30]]
        I = np.array([[r[k] for k in KEYS] for r in rows])
        ok = np.array([r["feasible"] for r in rows])
        val = np.array([r["valid"] for r in rows])
        cd, ser, cl = (np.array([r[k] for r in rows], float) for k in ("CD_counts", "SER_db", "CLmax"))
        vl = val & np.isfinite(cl)
        s_cd, s_ser = np.ptp(cd[ok]), np.ptp(ser[ok])
        c_cd = fit_quadratic(I[ok], cd[ok], levels=SP.n_levels)
        c_ser = fit_quadratic(I[ok], ser[ok], levels=SP.n_levels)
        c_cl = fit_linear(I[vl], cl[vl], levels=SP.n_levels)
        ct = Contrainte.depuis_z("CLmax", c_cl, SP.bits, SP.n_levels, seuil=1.2, s=max(np.ptp(cl[vl]), 1e-3))
        for w in (0.0, 0.5, 1.0):
            Qo, co = surrogate_to_qubo(w * c_cd / s_cd + (1 - w) * c_ser / s_ser, SP.bits, SP.n_levels)
            obj = qubo_energy(Qo, co, Bd)
            faisable = t_ok & (ct.h(Bd) >= 0)
            x_opt = int(np.argmin(np.where(faisable, obj, np.inf)))
            # 1. bits d'écart + Rosenberg
            Q, c = ajoute_contraintes(Qo.copy(), co, c_cl, SPEC, 1.2, modele_cl=True)
            out["ecart"].append(dict(seed=seed, w=w, part_objectif=float(np.ptp(obj) / np.ptp(qubo_energy(Q, c, all_bitstrings(Q.shape[0])))),
                                     **mesure(Q, c, Bd, x_opt, faisable)))
            # 2. pénalité déséquilibrée + Rosenberg
            Q, c = ajoute_contraintes(Qo.copy(), co, None, SPEC, 1.2, modele_cl=False)
            Q, c = penalite_desequilibree(Q, c, ct)
            out["desequilibree"].append(dict(seed=seed, w=w, part_objectif=float(np.ptp(obj) / np.ptp(qubo_energy(Q, c, all_bitstrings(Q.shape[0])))),
                                             **mesure(Q, c, Bd, x_opt, faisable)))
            # 3. lagrangien (lambda par montée duale, solveur exact), épaisseur vérifiée sur les échantillons
            def solve(Qx, cx):
                E = qubo_energy(Qx, cx, Bd)
                o = np.argsort(E)[:64]
                return Bd[o], E[o]

            def filtre(S):
                return decode_bits(S, SP.bits)[:, 2] != 0
            S, lam, _ = resout_lagrange(Qo, co, [ct], solve, filtre=filtre)
            Q, c = lagrangien(Qo, co, [ct], lam)
            r = dict(seed=seed, w=w, lam=float(lam[0]), part_objectif=float(np.ptp(obj) / np.ptp(qubo_energy(Q, c, Bd))),
                     optimum_retrouve_par_verification=bool(tuple(S[0]) == tuple(Bd[x_opt])),
                     **mesure(Q, c, Bd, x_opt, faisable))
            out["lagrange"].append(r)
            print(seed, w, {f: {k: round(v, 3) if isinstance(v, float) else v for k, v in out[f][-1].items()} for f in out}, flush=True)
    for f, v in out.items():
        print(f"{f:14s} : {v[0]['qubits']} qubits, part de l'objectif {np.median([x['part_objectif'] for x in v]):.3f}, "
              f"gain sur l'optimum contraint {np.median([x['p_optimum'] for x in v]):.1f} "
              f"(min {min(x['p_optimum'] for x in v):.1f}, max {max(x['p_optimum'] for x in v):.1f}), "
              f"P(faisable) {np.median([x['p_faisable'] for x in v]):.2f}")
    print("lagrangien : optimum contraint en tête des lectures vérifiées (solveur exact) :",
          sum(x["optimum_retrouve_par_verification"] for x in out["lagrange"]), "/", len(out["lagrange"]))
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("->", OUT)


if __name__ == "__main__":
    main()
