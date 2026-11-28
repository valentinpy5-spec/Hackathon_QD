"""Itération V3, lot L7 : diagnostic des contraintes sur le problème CST, sans aucun appel à XFOIL.

On rejoue les 8 exécutions de l'approche actuelle (quadz_modele) de la seconde campagne CST, lues dans
resultats/ia_qubo_echelle_brut.json : pour chaque profil proposé par la boucle, on réajuste les modèles
sur les profils simulés AVANT lui (comme la boucle à l'itération correspondante) et l'on compare la
prédiction à la vérité du cache.

    1. Les C_L,max trop faibles viennent-ils du modèle ou de la formulation ? Pour chaque profil proposé
       avec C_L,max < 1,2 : si le modèle affine prédisait déjà < 1,2, la pénalité n'a pas tenu
       (formulation) ; s'il prédisait >= 1,2, le modèle s'est trompé.
    2. Que vaudrait le classifieur de convergence ? Sur les profils proposés, probabilité prédite de
       succès contre le résultat d'XFOIL : AUC, échecs évités et succès sacrifiés selon le seuil.
       Et sur le front de référence : quelle probabilité le classifieur appris sur tout le cache donne-t-il
       à ses profils (risque d'exclure le front) ?
    3. Les QUBO eux-mêmes : à 60, 150 et 300 simulations, pour les 5 poids, on construit le QUBO avec
       bits d'écart (V2), pénalité déséquilibrée et lagrangien, on les résout par recuit, et l'on compare
       la part de l'objectif dans le spectre, la part des lectures prédites faisables, et l'objectif
       prédit du candidat retenu.

    python3 -m contraintes.etude_cst          -> resultats/contraintes_cst.json  (~5 min)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.qubo_loop import fit_linear, fit_quadratic, predict as predict_z  # noqa: E402
from quantique.qubo import surrogate_to_qubo, qubo_energy  # noqa: E402
from ia_qubo import probleme_cst as P  # noqa: E402
from ia_qubo.boucle import recuit, ajoute_contraintes, WEIGHTS  # noqa: E402
from ia_qubo.modeles import encode, decode_bits  # noqa: E402
from contraintes.classifieur import LogistiqueQuadZ, auc  # noqa: E402
from contraintes.formulations import (Contrainte, penalite_desequilibree, resout_lagrange,  # noqa: E402
                                      part_objectif, lagrangien)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "resultats", "contraintes_cst.json")
LV, BITS, KEYS = P.SPEC["levels"], P.SPEC["bits"], P.KEYS
CL_MIN = P.CL_MAX_MIN
GEO = [Contrainte.depuis_z(n, c, BITS, LV, seuil=sz, s=e, verifiee=False) for n, c, sz, e in P.SPEC["geometrie_z"]]


def donnees(cache, seen):
    rows = [cache[tuple(s)] for s in seen]
    I = np.array([[r[k] for k in KEYS] for r in rows])
    return (I, np.array([r["feasible"] for r in rows]), np.array([r["valid"] for r in rows]),
            np.array([r["CD_counts"] for r in rows], float), np.array([r["SER_db"] for r in rows], float),
            np.array([r["CLmax"] for r in rows], float))


def rejoue(cache, runs):
    """Parties 1 et 2 : prédictions au moment de la proposition."""
    cl_cas, conv = [], []
    for seen in runs:
        seen = [tuple(s) for s in seen]
        I, ok, val, cd, ser, cl = donnees(cache, seen)
        for m in range(10, len(seen), 5):
            vl = val[:m] & np.isfinite(cl[:m])
            if ok[:m].sum() < 4 or vl.sum() < 3:
                continue
            coef = fit_linear(I[:m][vl], cl[:m][vl], levels=LV)
            prop = slice(m, min(m + 5, len(seen)))
            pred = predict_z(coef, I[prop], LV)
            for j, pj in zip(range(prop.start, prop.stop), pred):
                if val[j] and np.isfinite(cl[j]):
                    cl_cas.append((m, float(pj), float(cl[j])))
            if (~val[:m]).sum() >= 3 and val[:m].sum() >= 3:
                clf = LogistiqueQuadZ(LV).fit(I[:m], val[:m])
                for j, pj in zip(range(prop.start, prop.stop), clf.proba(I[prop])):
                    conv.append((m, float(pj), bool(val[j])))
    return cl_cas, conv


def bilan_cl(cl_cas):
    a = np.array(cl_cas)
    faible = a[:, 2] < CL_MIN
    pred_ok = a[:, 1] >= CL_MIN
    err = a[:, 1] - a[:, 2]
    return dict(n_proposes_valides=len(a), n_cl_faible=int(faible.sum()),
                faible_mais_predit_ok=int((faible & pred_ok).sum()),
                faible_et_predit_faible=int((faible & ~pred_ok).sum()),
                erreur_mediane_abs=float(np.median(np.abs(err))),
                biais_median=float(np.median(err)),
                erreur_abs_par_budget={str(b): float(np.median(np.abs(err[(a[:, 0] >= b) & (a[:, 0] < b + 100)])))
                                       for b in (0, 100, 200)})


def bilan_conv(conv, cache, front):
    a = np.array([(m, p, v) for m, p, v in conv], float)
    p, v = a[:, 1], a[:, 2].astype(bool)
    out = dict(n=len(a), taux_echec=float(1 - v.mean()), auc=auc(p, v),
               auc_par_budget={str(b): auc(p[(a[:, 0] >= b) & (a[:, 0] < b + 100)], v[(a[:, 0] >= b) & (a[:, 0] < b + 100)])
                               for b in (0, 100, 200)}, seuils={})
    for s in (0.1, 0.2, 0.3, 0.5):
        rej = p < s
        out["seuils"][str(s)] = dict(echecs_evites=float((rej & ~v).sum() / max((~v).sum(), 1)),
                                    succes_sacrifies=float((rej & v).sum() / max(v.sum(), 1)),
                                    part_rejetee=float(rej.mean()))
    # front de référence : probabilité donnée par un classifieur appris sur tout le cache admissible
    adm = [(k, r) for k, r in cache.items() if r["geom_ok"] and r["t_max"] >= P.T_MIN]
    I = np.array([k for k, _ in adm])
    y = np.array([r["valid"] for _, r in adm])
    clf = LogistiqueQuadZ(LV).fit(I, y)
    out["front_proba_tout_cache"] = sorted(float(x) for x in clf.proba(np.array(front)))
    out["auc_tout_cache_apprentissage"] = auc(clf.proba(I), y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(I))
    a_, t_ = perm[:len(I) // 2], perm[len(I) // 2:]
    out["auc_tout_cache_test"] = auc(LogistiqueQuadZ(LV).fit(I[a_], y[a_]).proba(I[t_]), y[t_])
    return out


def filtre_b(S):
    return np.array([P.admissible(tuple(int(v) for v in x)) for x in decode_bits(S[:, :sum(BITS)], BITS)])


def qubos(cache, runs, n_points=(60, 150, 300)):
    """Partie 3 : les trois formulations sur les mêmes QUBO."""
    res = {f: [] for f in ("ecart", "desequilibree", "lagrange")}
    spec = dict(P.SPEC, t_var=None)
    for g, seen in enumerate(runs):
        seen = [tuple(s) for s in seen]
        I, ok, val, cd, ser, cl = donnees(cache, seen)
        for m in n_points:
            Im, okm, vl = I[:m], ok[:m], val[:m] & np.isfinite(cl[:m])
            s_cd, s_ser = np.ptp(cd[:m][okm]), np.ptp(ser[:m][okm])
            c_cd = fit_quadratic(Im[okm], cd[:m][okm], levels=LV)
            c_ser = fit_quadratic(Im[okm], ser[:m][okm], levels=LV)
            c_cl = fit_linear(Im[vl], cl[:m][vl], levels=LV)
            ct = Contrainte.depuis_z("CLmax", c_cl, BITS, LV, seuil=CL_MIN, s=max(np.ptp(cl[:m][vl]), 1e-3))
            rng = np.random.default_rng(1000 * g + m)
            for w in WEIGHTS:
                Qo, co = surrogate_to_qubo(w * c_cd / s_cd + (1 - w) * c_ser / s_ser, BITS, LV)
                T = float(np.abs(Qo).max())

                def solve(Q, c):
                    return recuit(Q, c, n_sweeps=200, n_reads=128, T0=2 * T, T1=0.01 * T,
                                  seed=int(rng.integers(1 << 30)))
                Balea = rng.integers(0, 2, (4000, sum(BITS))).astype(float)
                for f in res:
                    if f == "ecart":
                        Q, c = ajoute_contraintes(Qo.copy(), co, c_cl, spec, CL_MIN, modele_cl=True)
                        S, E = solve(Q, c)
                        S = S[np.argsort(E)]
                        Bx = np.hstack([Balea, rng.integers(0, 2, (len(Balea), Q.shape[0] - Qo.shape[0]))])
                        part = part_objectif(Qo, co, Q, c, np.vstack([Bx, S]))
                        nres = 1
                    elif f == "desequilibree":
                        Q, c = penalite_desequilibree(Qo.copy(), co, ct)
                        S, E = solve(Q, c)
                        S = S[np.argsort(E)]
                        part = part_objectif(Qo, co, Q, c, np.vstack([Balea, S]))
                        nres = 1
                    else:
                        ctrs = [ct] + GEO
                        S, lam, info = resout_lagrange(Qo, co, ctrs, solve, filtre=filtre_b)
                        Q, c = lagrangien(Qo, co, ctrs, lam)
                        part = part_objectif(Qo, co, Q, c, np.vstack([Balea, S]))
                        nres = 4
                    D = S[:, :Qo.shape[0]]
                    idx = [tuple(int(v) for v in x) for x in decode_bits(D, BITS)]
                    adm = np.array([P.admissible(x) for x in idx])
                    hok = ct.h(D) >= 0
                    # candidat retenu : comme la boucle, la première lecture admissible pas encore simulée
                    # (pour le lagrangien, les lectures sont déjà classées : prédites faisables d'abord)
                    deja = set(seen[:m])
                    j = next((k for k in range(len(idx)) if adm[k] and idx[k] not in deja), None)
                    # (aucune lecture admissible et nouvelle : la boucle tirerait un profil au hasard)
                    res[f].append(dict(graine=g, n=m, w=w, qubits=int(Q.shape[0]), part_objectif=part,
                                       lectures_predites_faisables=float(np.mean(hok & adm)),
                                       retenu_predit_faisable=bool(hok[j]) if j is not None else False,
                                       retenu_objectif=float(qubo_energy(Qo, co, D[j:j + 1])[0]) if j is not None else None,
                                       retenu_cl=float(ct.h(D[j:j + 1])[0] + CL_MIN) if j is not None else None,
                                       resolutions=nres))
    return res


def main():
    cache = P._charge_cache()
    runs = [seen for seen, _ in json.load(open(os.path.join(HERE, "resultats", "ia_qubo_echelle_brut.json")))["quadz_modele"]]
    front = json.load(open(os.path.join(HERE, "resultats", "ia_qubo_echelle.json")))["front_ref"]
    cl_cas, conv = rejoue(cache, runs)
    out = dict(cl=bilan_cl(cl_cas), convergence=bilan_conv(conv, cache, front))
    print(json.dumps(out, indent=1), flush=True)
    out["qubo"] = qubos(cache, runs)
    for f, L in out["qubo"].items():
        ob = [x["retenu_objectif"] for x in L if x["retenu_predit_faisable"]]
        print(f"{f:14s} qubits {L[0]['qubits']}, part objectif méd {np.median([x['part_objectif'] for x in L]):.3f}, "
              f"lectures prédites faisables {np.median([x['lectures_predites_faisables'] for x in L]):.2f}, "
              f"retenu prédit faisable {np.mean([x['retenu_predit_faisable'] for x in L]):.2f}, "
              f"objectif retenu méd (faisables) {np.median(ob):.3f}")
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("->", OUT)


if __name__ == "__main__":
    main()
