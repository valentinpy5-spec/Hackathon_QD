"""Itération V3, lot L7 : exporter un QUBO de la boucle avec ses contraintes SÉPARÉES (modèle quadratique
contraint, « CQM »), au lieu d'une matrice où les pénalités sont déjà mélangées à l'objectif.

C'est l'interface des solveurs industriels : les solveurs hybrides contraints (par exemple le solveur
CQM de D-Wave) et les solveurs classiques en variables binaires (CPLEX, Gurobi, SCIP) lisent un objectif
quadratique et une liste de contraintes ; ce sont EUX qui choisissent comment traiter les contraintes
(pénalités réglées automatiquement, branchement, etc.). Le format choisi est le format LP (texte), lu par
tous ces outils (dimod.lp.load côté D-Wave).

    minimiser    b^T Q b + c                       (objectif pondéré w C_D + (1 - w) SER, normalisé)
    sous         b^T H_k b + h0_k >= 0             (C_L,max, faces sans croisement, convergence d'XFOIL)
                 b binaire

    python3 -m contraintes.export_cqm           -> resultats/cqm_exemple.lp et cqm_exemple.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _terme(v, nom):
    return f"{'+' if v >= 0 else '-'} {abs(v):.10g} {nom}"


def _forme(Q, noms, objectif):
    """Partie linéaire et partie quadratique au format LP. Dans l'objectif, le crochet est divisé par 2
    (convention LP) : on y écrit donc 2 Q_ij."""
    n = Q.shape[0]
    lin = [_terme(Q[i, i], noms[i]) for i in range(n) if Q[i, i] != 0]
    quad = []
    f = 2.0 if objectif else 1.0
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                quad.append(_terme(f * Q[i, j], f"{noms[i]} * {noms[j]}"))
    s = " ".join(lin) if lin else "0 " + noms[0]
    if quad:
        s += " + [ " + " ".join(quad) + " ]" + (" / 2" if objectif else "")
    return s


def ecrire_lp(chemin, Q, c, contraintes, noms=None, titre=""):
    """contraintes : liste d'objets avec nom, H, h0 (contrainte b^T H b + h0 >= 0)."""
    n = Q.shape[0]
    noms = noms or [f"b{i}" for i in range(n)]
    lignes = [f"\\ {titre}", f"\\ constante de l'objectif (non écrite, sans effet sur l'optimum) : {c:.10g}",
              "Minimize", " obj: " + _forme(Q, noms, True), "Subject To"]
    for k, ct in enumerate(contraintes):
        nom = "".join(ch if ch.isalnum() else "_" for ch in ct.nom)
        lignes.append(f" c{k}_{nom}: " + _forme(np.asarray(ct.H), noms, False) + f" >= {-ct.h0:.10g}")
    lignes += ["Binary", " " + " ".join(noms), "End", ""]
    with open(chemin, "w") as f:
        f.write("\n".join(lignes))


def ecrire_json(chemin, Q, c, contraintes, meta=None):
    d = dict(meta=meta or {}, objectif=dict(Q=np.asarray(Q).tolist(), c=float(c)),
             contraintes=[dict(nom=ct.nom, H=np.asarray(ct.H).tolist(), h0=float(ct.h0), sens=">=", rhs=0.0)
                          for ct in contraintes])
    with open(chemin, "w") as f:
        json.dump(d, f)


def exemple():
    """Le QUBO d'une itération réelle de la seconde campagne CST (approche actuelle, graine 0, 150 profils),
    poids w = 0,5 : objectif, C_L,max, faces sans croisement et classifieur de convergence."""
    from optim.qubo_loop import fit_linear, fit_quadratic
    from quantique.qubo import surrogate_to_qubo
    from ia_qubo import probleme_cst as P
    from contraintes.classifieur import LogistiqueQuadZ
    from contraintes.formulations import Contrainte
    LV, BITS, KEYS = P.SPEC["levels"], P.SPEC["bits"], P.KEYS
    cache = P._charge_cache()
    seen = json.load(open(os.path.join(HERE, "resultats", "ia_qubo_echelle_brut.json")))["quadz_modele"][0][0][:150]
    rows = [cache[tuple(s)] for s in seen]
    I = np.array([[r[k] for k in KEYS] for r in rows])
    ok = np.array([r["feasible"] for r in rows])
    val = np.array([r["valid"] for r in rows])
    cd, ser, cl = (np.array([r[k] for r in rows], float) for k in ("CD_counts", "SER_db", "CLmax"))
    vl = val & np.isfinite(cl)
    w = 0.5
    coef = w * fit_quadratic(I[ok], cd[ok], levels=LV) / np.ptp(cd[ok]) + \
        (1 - w) * fit_quadratic(I[ok], ser[ok], levels=LV) / np.ptp(ser[ok])
    Q, c = surrogate_to_qubo(coef, BITS, LV)
    ctrs = [Contrainte.depuis_z("CLmax_min", fit_linear(I[vl], cl[vl], levels=LV), BITS, LV, seuil=P.CL_MAX_MIN)]
    ctrs += [Contrainte.depuis_z(n, cz, BITS, LV, seuil=sz) for n, cz, sz, _ in P.SPEC["geometrie_z"]]
    clf = LogistiqueQuadZ(LV).fit(I, val)
    ctrs.append(Contrainte.depuis_z("convergence_xfoil", clf.contrainte(0.1), BITS, LV))
    noms = [f"{k}_b{j}" for k, nb in zip(KEYS, BITS) for j in range(nb)]
    titre = "Boucle QUBO, problème CST 30 bits, graine 0, 150 profils simulés, w = 0,5 (lot L7)"
    ecrire_lp(os.path.join(HERE, "resultats", "cqm_exemple.lp"), Q, c, ctrs, noms, titre)
    ecrire_json(os.path.join(HERE, "resultats", "cqm_exemple.json"), Q, c, ctrs,
                dict(titre=titre, bits=list(BITS), variables=list(KEYS), noms=noms,
                     codage="indice x_i = sum_j 2^j b_ij, poids faible d'abord"))
    return Q, c, ctrs, noms


if __name__ == "__main__":
    Q, c, ctrs, noms = exemple()
    print(f"{Q.shape[0]} variables binaires, {len(ctrs)} contraintes "
          f"({sum(1 for ct in ctrs if ct.affine)} linéaires, {sum(1 for ct in ctrs if not ct.affine)} quadratique)")
    print("->", os.path.join(HERE, "resultats", "cqm_exemple.lp"))
